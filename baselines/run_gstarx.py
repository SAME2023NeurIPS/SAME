import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import time
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from methods.gstarx import GStarX
from load_dataset import get_dataset, get_dataloader
from gnnNets import get_gnnNets
from utils import check_dir, get_logger, evaluate_scores_list, PlotUtils, Recorder

IS_FRESH = False


@hydra.main(config_path="../config", config_name="config")
def main(config):
    if not os.path.isdir(config.record_filename):
        os.makedirs(config.record_filename)
    config.record_filename = os.path.join(config.record_filename, f"{config.datasets.dataset_name}.json")

    recorder = Recorder(config.record_filename)

    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]

    explainer_name = config.explainers.explainer_name

    # log_file = (
    #     f"{explainer_name}_{config.datasets.dataset_name}_{config.models.gnn_name}.log"
    # )
    # logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    # logger.debug(OmegaConf.to_yaml(config))

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda", index=config.device_id)
    else:
        device = torch.device("cpu")

    # Load dataset
    dataset = get_dataset(
        dataset_root=config.datasets.dataset_root,
        dataset_name=config.datasets.dataset_name,
    )
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    dataloader_params = {
        "batch_size": config.models.param.batch_size,
        "random_split_flag": config.datasets.random_split_flag,
        "data_split_ratio": config.datasets.data_split_ratio,
        "seed": config.datasets.seed,
    }
    dataloader = get_dataloader(dataset, **dataloader_params)
    test_indices = dataloader["test"].dataset.indices
    if config.datasets.dataset_name == 'mutag':
        test_indices = list(range(len(dataset)))
    
    # if config.datasets.data_explain_cutoff > 0:
    #     test_indices = test_indices[: config.datasets.data_explain_cutoff]

    import random
    random.seed(config.datasets.seed)
    random.shuffle(test_indices)
    
    if 'graph_sst' in config.datasets.dataset_name.lower():
        print('Using 30 data instances only...')
        test_indices = sorted(test_indices, key=lambda x: dataset[x].num_nodes, reverse=True)
        test_indices = [x for x in test_indices if dataset[x].num_nodes == 16]
        test_indices = test_indices[10:40]

    # Load model
    model = get_gnnNets(
        input_dim=dataset.num_node_features,
        output_dim=dataset.num_classes,
        model_config=config.models,
    )

    state_dict = torch.load(
        os.path.join(
            config.models.gnn_saving_dir,
            config.datasets.dataset_name,
            f"{config.models.gnn_name}_"
            f"{len(config.models.param.gnn_latent_dim)}l_best.pth",
        )
    )["net"]

    model.load_state_dict(state_dict)

    model.to(device)

    explanation_saving_path = os.path.join(
        config.explainers.explanation_result_path,
        config.datasets.dataset_name,
        config.models.gnn_name,
        explainer_name,
    )

    check_dir(explanation_saving_path)
    # Test prediction accuracy and get average payoff (probability)
    preds = []
    rst = []
    for data in dataloader["test"]:
        data.to(device)
        pred = model(data).softmax(-1)
        preds += [pred]
        rst += [pred.argmax(-1) == data.y]
    preds = torch.concat(preds)
    rst = torch.concat(rst)
    payoff_avg = preds.mean(0).tolist()
    acc = rst.float().mean().item()
    # logger.debug("Predicted prob: " + ",".join([f"{p:.4f}" for p in payoff_avg]))
    # logger.debug(f"Test acc: {acc:.4f}")

    explainer = GStarX(
        model,
        device=device,
        max_sample_size=config.explainers.param.max_sample_size,
        tau=config.explainers.param.tau,
        payoff_type=config.explainers.param.payoff_type,
        payoff_avg=payoff_avg,
        subgraph_building_method=config.explainers.param.subgraph_building_method,
    )

    plot_utils = PlotUtils(config.datasets.dataset_name, is_show=False)
    scores_list = []
    start_time = time.time()
    for i, data in enumerate(tqdm(dataset[test_indices])):
        idx = test_indices[i]
        data.to(device)

        explained_example_path = os.path.join(
            explanation_saving_path, f"example_{idx}.pt"
        )
        if not IS_FRESH and os.path.isfile(explained_example_path):
            node_scores = torch.load(explained_example_path)
            # node_scores = explainer.explain(
            #     data,
            #     superadditive_ext=config.explainers.superadditive_ext,
            #     sample_method=config.explainers.sample_method,
            #     num_samples=config.explainers.num_samples,
            #     k=config.explainers.num_hops,
            # )

            torch.save(node_scores, explained_example_path)
        else:
            # use data.num_nodes as num_samples
            # continue
            node_scores = explainer.explain(
                data,
                superadditive_ext=config.explainers.superadditive_ext,
                sample_method=config.explainers.sample_method,
                num_samples=config.explainers.num_samples,
                k=config.explainers.num_hops,
            )

            torch.save(node_scores, explained_example_path)

        scores_list += [node_scores]

        if config.save_plot:
            from utils import (
                scores2coalition,
                evaluate_coalition,
                fidelity_normalize_and_harmonic_mean,
                to_networkx,
            )

            coalition = scores2coalition(node_scores, config.explainers.sparsity)
            f, abs_f, inv_f, sp = evaluate_coalition(explainer, data, coalition)
            n_f, n_inv_f, h_f = fidelity_normalize_and_harmonic_mean(f, inv_f, sp)
            title_sentence = f"fide: {f:.3f}, abs-fide: {abs_f:.3f}, h-fide: {h_f:.3f}"
            title_sentence = f'fide: {f:.4f}, spar: {sp:.4f}'

            if hasattr(dataset, "supplement"):
                words = dataset.supplement["sentence_tokens"][str(idx)]
            else:
                words = None

            explained_example_plot_path = os.path.join(
                explanation_saving_path, f"example_{idx}.png"
            )
            plot_utils.plot(
                to_networkx(data),
                coalition,
                x=data.x,
                words=words,
                title_sentence=title_sentence,
                figname=explained_example_plot_path,
            )
    end_time = time.time()
    metrics = evaluate_scores_list(
        explainer,
        dataset[test_indices],
        scores_list,
        config.explainers.sparsity,
    )

    sp_mean, f_mean, inv_f_mean, n_f_mean, n_inv_f_mean, h_f_mean = metrics

    # metrics_str = ",".join([f"{m : .4f}" for m in metrics])
    # print(metrics_str)
    # logger.info(f"Time in seconds: {end_time - start_time}\n"
    #             f"Avg Time: {(end_time - start_time)/len(test_indices)}")
    # print(f"Time in seconds: {end_time - start_time}")
    # print(f"Avg Time: {(end_time - start_time)/len(test_indices)}")

    experiment_data = {
        'fidelity': f_mean,
        'fidelity_inv': inv_f_mean,
        'h_fidelity': h_f_mean,
        'sparsity': sp_mean,
        'Time in seconds': end_time - start_time,
        'Average Time': (end_time - start_time)/len(test_indices)
    }
    
    recorder.append(experiment_settings=['gstarx', f"{config.explainers.sparsity}"],
                    experiment_data=experiment_data)

    recorder.save()


if __name__ == "__main__":
    import sys
    wkdir = os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

    sys.argv.append("explainers=gstarx")
    sys.argv.append(f"datasets.dataset_root={os.path.join(wkdir, 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(wkdir, 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_path={os.path.join(wkdir, 'results')}")
    sys.argv.append(f"record_filename={os.path.join(wkdir, 'result_jsons')}")
    main()
