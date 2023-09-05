import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
from load_dataset import get_dataset, get_dataloader
from gnnNets import get_gnnNets
from utils import check_dir, get_logger, evaluate_scores_list, PlotUtils
from methods import GraphSVX

IS_FRESH = False


@hydra.main(config_path="../config", config_name="config")
def main(config):
    cwd = os.path.dirname(os.path.abspath(__file__))
    pwd = os.path.dirname(cwd)

    config.datasets.dataset_root = os.path.join(pwd, "datasets")
    config.models.gnn_saving_path = os.path.join(pwd, "checkpoints")
    config.explainers.explanation_result_path = os.path.join(cwd, "results")
    config.log_path = os.path.join(cwd, "log")

    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]

    explainer_name = f"{config.explainers.explainer_name}"
    log_file = (
        f"{explainer_name}_{config.datasets.dataset_name}_{config.models.gnn_name}.log"
    )
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    logger.debug(OmegaConf.to_yaml(config))

    if torch.cuda.is_available():
        device = torch.device("cuda", index=config.device_id)
    else:
        device = torch.device("cpu")

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

    if config.explainers.param.subgraph_building_method == "split":
        config.models.param.add_self_loop = False

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

    explanation_saving_path = os.path.join(
        config.explainers.explanation_result_path,
        config.datasets.dataset_name,
        config.models.gnn_name,
        explainer_name,
    )
    check_dir(explanation_saving_path)

    # Explain it with GraphSVX
    explainer = GraphSVX(
        dataset,
        model,
        device,
        subgraph_building_method=config.explainers.param.subgraph_building_method,
    )

    plot_utils = PlotUtils(config.datasets.dataset_name, is_show=False)
    scores_list = []
    for i, data in enumerate(tqdm(dataset[test_indices])):
        idx = test_indices[i]
        explained_example_path = os.path.join(
            explanation_saving_path, f"example_{idx}.pt"
        )
        if not IS_FRESH and os.path.isfile(explained_example_path):
            node_scores = torch.load(explained_example_path)
            logger.debug(f"Load example {idx}.")
        else:
            # use data.num_nodes as num_samples
            node_scores = explainer.explain(
                data,
                config.explainers.hops,
                config.explainers.num_samples,
                config.explainers.info,
                config.explainers.multiclass,
                config.explainers.fullempty,
                config.explainers.S,
                config.explainers.hv,
                config.explainers.feat,
                config.explainers.coal,
                config.explainers.g,
                config.explainers.regu,
            )

            torch.save(node_scores, explained_example_path)

        scores_list += [node_scores]
        if config.save_plot:
            logger.debug(f"Plotting example {idx}.")
            from utils import (
                scores2coalition,
                evaluate_coalition,
                fidelity_normalize_and_harmonic_mean,
                to_networkx,
            )

            coalition = scores2coalition(node_scores, config.explainers.sparsity)
            f, abs_f, inv_f, sp = evaluate_coalition(explainer, data, coalition)
            n_f, n_inv_f, h_f = fidelity_normalize_and_harmonic_mean(f, inv_f, sp)
            title_sentence = f"fide: {f:.3f}, inv-fide: {inv_f:.3f}, h-fide: {h_f:.3f}"

            if hasattr(dataset, "supplement"):
                words = dataset.supplement["sentence_tokens"][str(idx)]
            else:
                words = None

            explained_example_plot_path = os.path.join(
                explanation_saving_path, f"example_{idx}.png"
            )
            # plot_utils.plot(
            #     to_networkx(data),
            #     coalition,
            #     x=data.x,
            #     words=words,
            #     title_sentence=title_sentence,
            #     figname=explained_example_plot_path,
            # )

    metrics = evaluate_scores_list(
        explainer,
        dataset[test_indices],
        scores_list,
        config.explainers.sparsity,
        logger,
    )

    metrics_str = ",".join([f"{m : .4f}" for m in metrics])
    print(metrics_str)


if __name__ == "__main__":
    import sys

    sys.argv.append("explainers=graphsvx")
    main()
