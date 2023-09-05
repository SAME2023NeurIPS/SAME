import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np
import torch.nn.functional as F
import hydra
from tqdm import tqdm
from torch import nn, optim
from torch_geometric.utils import dense_to_sparse
from omegaconf import OmegaConf
from dataset import get_dataset, get_dataloader
from gnnNets import get_gnnNets
from utils import check_dir, get_logger, PlotUtils
from baselines.baselines_utils import evaluate_related_preds_list
from baselines.methods import VGAE3MLP, Orphicx, causaleffect
from baselines.methods.orphicx import (
    gaeloss,
    get_orphicx_default_args,
    dense_to_sparse_batch,
    eval_model,
    GraphSampler,
)

IS_FRESH = False
# IS_FRESH = True


@hydra.main(config_path="../config", config_name="config")
def pipeline(config):
    cwd = os.path.dirname(os.path.abspath(__file__))
    pwd = os.path.dirname(cwd)
    config.datasets.dataset_root = os.path.join(pwd, "datasets")
    config.models.gnn_saving_path = os.path.join(pwd, "checkpoints")
    config.explainers.explanation_result_path = os.path.join(cwd, "results")
    config.log_path = os.path.join(cwd, "log")

    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.explainers.sparsity = float(config.sparsity)

    explainer_name = config.explainers.explainer_name
    log_file = (
        f"{explainer_name}_{config.datasets.dataset_name}_{config.models.gnn_name}.log"
    )
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    logger.debug(OmegaConf.to_yaml(config))

    if torch.cuda.is_available() and config.device_id >= 0:
        device = torch.device("cuda", index=config.device_id)
    else:
        device = torch.device("cpu")

    """Load dataset"""
    # bbbp warning
    dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()

    if config.models.param.graph_classification:
        dataloader_params = {
            "batch_size": config.models.param.batch_size,
            "random_split_flag": config.datasets.random_split_flag,
            "data_split_ratio": config.datasets.data_split_ratio,
            "seed": config.datasets.seed,
        }
        loader = get_dataloader(dataset, **dataloader_params)
        train_indices = loader["train"].dataset.indices
        test_indices = loader["test"].dataset.indices

        # if config.datasets.data_explain_cutoff > 0:
        #     test_indices = test_indices[: config.datasets.data_explain_cutoff]
        # TODO: Partial
        import random
        random.seed(config.datasets.seed)
        random.shuffle(test_indices)
        if 'graph_sst' in config.datasets.dataset_name.lower():
            print('Using 30 data instances only...')
            test_indices = sorted(test_indices, key=lambda x: dataset[x].num_nodes, reverse=True)
            test_indices = [x for x in test_indices if dataset[x].num_nodes == 16]
            test_indices = test_indices[10:40]

    else:
        train_indices = range(len(dataset))

    """Dataset specially pre-processed for OrphicX"""
    orphicx_data_saving_path = os.path.join(
        config.datasets.dataset_root,
        config.datasets.dataset_name,
        "orphicx_processed_dataset.pt",
    )
    if os.path.isfile(orphicx_data_saving_path):
        logger.info("Loading processed dataset")
        processed_dataset = torch.load(
            orphicx_data_saving_path, map_location=torch.device("cpu")
        )
    else:
        processed_dataset = GraphSampler(dataset, torch.arange(len(dataset)))
        torch.save(processed_dataset, orphicx_data_saving_path)

    train_indices = train_indices[:100]

    train_cutoff = int(len(train_indices) * 0.8)
    train_dataset = [processed_dataset[i] for i in train_indices[:train_cutoff]]
    val_dataset = [processed_dataset[i] for i in train_indices[train_cutoff:]]
    test_dataset = [processed_dataset[i] for i in test_indices]

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.explainers.param.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    """Model init"""
    model = get_gnnNets(
        input_dim=dataset.num_node_features,
        output_dim=dataset.num_classes,
        model_config=config.models,
    )

    state_dict = torch.load(
        os.path.join(
            config.models.gnn_saving_path,
            config.datasets.dataset_name,
            f"{config.models.gnn_name}_"
            f"{len(config.models.param.gnn_latent_dim)}l_best.pth",
        )
    )["net"]

    model.load_state_dict(state_dict)

    model.to(device)

    """Explainer init"""
    explanation_saving_path = os.path.join(
        config.explainers.explanation_result_path,
        config.datasets.dataset_name,
        config.models.gnn_name,
        explainer_name,
    )
    check_dir(explanation_saving_path)

    args = get_orphicx_default_args()
    args.encoder_output = config.explainers.param.encoder_output
    args.patient = config.explainers.param.patient

    cum_size = dataset.slices["x"]
    sizes = cum_size - cum_size.roll(1, 0)
    args.one_hot_label_dim = sizes.max().item()

    ceparams = {
        "Nalpha": args.Nalpha,
        "Nbeta": args.Nbeta,
        "K": args.K,
        "L": args.encoder_output - args.K,
        "z_dim": args.encoder_output,
        "M": dataset.num_classes,
    }

    vgae = VGAE3MLP(
        dataset.num_node_features + args.one_hot_label_dim,
        args.encoder_hidden1,
        args.encoder_hidden1,
        args.encoder_output,
        args.decoder_hidden1,
        args.decoder_hidden2,
        args.K,
        args.dropout,
    ).to(device)

    orphicx = Orphicx(
        vgae,
        model,
        device,
        num_causal_factors=args.K,
        explain_graph=config.models.param.graph_classification,
    )
    optimizer = optim.Adam(orphicx.explainer.parameters(), lr=args.lr)
    criterion = gaeloss

    """Training"""
    orphicx_saving_path = os.path.join(explanation_saving_path, f"{explainer_name}.pth")
    if not IS_FRESH and os.path.isfile(orphicx_saving_path):
        logger.info("Loading saved orphicx model...")
        state_dict = torch.load(orphicx_saving_path, map_location=torch.device("cpu"))
        orphicx.load_state_dict(state_dict)
    else:
        best_loss = 100
        patient = args.patient
        for epoch in tqdm(range(1, args.epoch + 1)):

            orphicx.explainer.train()
            train_losses = []
            for data in train_dataloader:
                sparse_batch = dense_to_sparse_batch(
                    data["feat"], data["sub_adj"], data["feat_mask"]
                ).to(device)
                for k in data:
                    data[k] = data[k].to(device)

                optimizer.zero_grad()
                mu, logvar = orphicx.explainer.encode(data["sub_feat"], data["sub_adj"])
                sample_mu = orphicx.explainer.reparameterize(mu, logvar)
                recovered = orphicx.explainer.dc(sample_mu)
                org_logit = orphicx.model(sparse_batch.x, sparse_batch.edge_index)
                org_probs = F.softmax(org_logit, dim=1)
                if args.coef_lambda:
                    nll_loss = (
                        args.coef_lambda * criterion(recovered, mu, logvar, data).mean()
                    )
                else:
                    nll_loss = 0
                alpha_mu = torch.zeros_like(sample_mu)
                alpha_mu[:, :, : args.K] = sample_mu[:, :, : args.K]
                alpha_adj = torch.sigmoid(orphicx.explainer.dc(alpha_mu))
                masked_alpha_adj = alpha_adj * data["sub_adj"]

                masked_alpha_adj_hard = (masked_alpha_adj > 0.5).float()

                masked_alpha_sparse_batch = dense_to_sparse_batch(
                    data["feat"], masked_alpha_adj_hard, data["feat_mask"]
                )

                alpha_logit = orphicx.model(
                    sparse_batch.x, masked_alpha_sparse_batch.edge_index
                )
                alpha_sparsity = masked_alpha_adj.mean((1, 2)) / data["sub_adj"].mean(
                    (1, 2)
                )

                if args.coef_causal:
                    causal_loss = []
                    NX = min(data["feat"].shape[0], args.NX)
                    NA = min(data["feat"].shape[0], args.NA)
                    for idx in random.sample(range(0, data["feat"].shape[0]), NX):
                        _causal_loss, _ = causaleffect.joint_uncond(
                            ceparams,
                            orphicx.explainer.dc,
                            orphicx.model,
                            data["sub_adj"][idx],
                            data["feat"][idx],
                            act=torch.sigmoid,
                            device=device,
                        )
                        causal_loss += [_causal_loss]
                        for A_idx in random.sample(
                            range(0, data["feat"].shape[0]), NA - 1
                        ):
                            if args.node_perm:
                                perm = torch.randperm(data["graph_size"][idx])
                                perm_adj = data["sub_adj"][idx].clone().detach()
                                perm_adj[: data["graph_size"][idx]] = perm_adj[perm]
                            else:
                                perm_adj = data["sub_adj"][A_idx]
                            _causal_loss, _ = causaleffect.joint_uncond(
                                ceparams,
                                orphicx.explainer.dc,
                                orphicx.model,
                                perm_adj,
                                data["feat"][idx],
                                act=torch.sigmoid,
                                device=device,
                            )
                            causal_loss += [_causal_loss]
                    causal_loss = args.coef_causal * torch.stack(causal_loss).mean()
                else:
                    causal_loss = 0

                if args.coef_kl:
                    klloss = args.coef_kl * F.kl_div(
                        F.log_softmax(alpha_logit, dim=1), org_probs
                    )
                else:
                    klloss = 0
                if args.coef_size:
                    size_loss = args.coef_size * alpha_sparsity.mean()
                else:
                    size_loss = 0

                loss = nll_loss + causal_loss + klloss + size_loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    orphicx.explainer.parameters(), args.max_grad_norm
                )
                optimizer.step()
                train_losses += [[nll_loss, causal_loss, klloss, size_loss]]

            if epoch % args.eval_epoch == 0:
                val_loss = eval_model(
                    args, orphicx, val_dataloader, device, criterion, ceparams
                )
                patient -= 1
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(orphicx.state_dict(), orphicx_saving_path)
                    patient = args.patient
                elif patient <= 0:
                    logger.info("Early stopping!")
                    break

                logger.info(
                    f"train_loss: {np.array(loss.item()).round(5)}, val_loss: {np.array(val_loss).round(5)}"
                )

        state_dict = torch.load(orphicx_saving_path)
        orphicx.load_state_dict(state_dict)

    """Evaluating"""
    plot_utils = PlotUtils(config.datasets.dataset_name, is_show=False)
    related_preds_list = []
    for i, data in enumerate(tqdm(test_dataloader)):
        orphicx.eval()
        idx = test_indices[i]
        sparse_batch = dense_to_sparse_batch(
            data["feat"], data["sub_adj"], data["feat_mask"]
        ).to(device)
        for k in data:
            data[k] = data[k].to(device)

        prediction = model(sparse_batch).softmax(dim=-1).argmax().item()
        edge_masks, hard_edge_masks, related_preds = orphicx(
            data,
            sparse_batch,
            num_classes=dataset.num_classes,
            sparsity=config.explainers.sparsity,
        )

        related_preds = related_preds[prediction]
        hard_edge_masks = hard_edge_masks[prediction]
        related_preds_list += [related_preds]

        if config.save_plot:
            logger.debug(f"Plotting example {idx}.")
            from utils import fidelity_normalize_and_harmonic_mean, to_networkx
            from baselines.baselines_utils import hard_edge_masks2coalition

            sparse_data = sparse_batch.to_data_list()[0]

            coalition = hard_edge_masks2coalition(sparse_data, hard_edge_masks, False)
            f = related_preds["origin"] - related_preds["maskout"]
            inv_f = related_preds["origin"] - related_preds["masked"]
            sp = related_preds["sparsity"]
            n_f, n_inv_f, h_f = fidelity_normalize_and_harmonic_mean(f, inv_f, sp)
            title_sentence = f"fide: {f:.3f}, inv-fide: {inv_f:.3f}, h-fide: {h_f:.3f}"

            if hasattr(dataset, "supplement"):
                words = dataset.supplement["sentence_tokens"][str(idx)]
            else:
                words = None

            explained_example_plot_path = os.path.join(
                explanation_saving_path, f"example_{idx}.png"
            )
            plot_utils.plot(
                to_networkx(sparse_data),
                coalition,
                x=sparse_data.x,
                words=words,
                title_sentence=title_sentence,
                figname=explained_example_plot_path,
            )

    metrics = evaluate_related_preds_list(related_preds_list, logger)
    metrics_str = ",".join([f"{m : .4f}" for m in metrics])
    print(metrics_str)


if __name__ == "__main__":
    import sys

    sys.argv.append("explainers=orphicx")
    pipeline()
