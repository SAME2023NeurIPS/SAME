import os
import torch
import networkx as nx
from dig.xgraph.utils.compatibility import compatible_state_dict
from omegaconf import OmegaConf
from tqdm import tqdm
import hydra
import numpy as np
import time
from gnnNets import get_gnnNets
from load_dataset import get_dataset, get_dataloader
from fornode.mcts import MCTS, reward_func
from shapley import gnn_score, GnnNets_NC2value_func
from torch_geometric.utils import to_networkx, add_remaining_self_loops
from utils import PlotUtils, find_closest_node_result, eval_metric, Recorder, fidelity_normalize_and_harmonic_mean


@hydra.main(config_path="config", config_name="config")
def pipeline(config):
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.explainers.sparsity = config.sparsity
    config.models.param.add_self_loop = False

    if not os.path.isdir(config.record_filename):
        os.makedirs(config.record_filename)
    config.record_filename = os.path.join(config.record_filename, f"{config.datasets.dataset_name}.json")
    print(OmegaConf.to_yaml(config))
    recorder = Recorder(config.record_filename)

    dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes

    # loader = get_dataloader(dataset,
    #                         batch_size=config.models.param.batch_size,
    #                         random_split_flag=config.datasets.random_split_flag,
    #                         data_split_ratio=config.datasets.data_split_ratio,
    #                         seed=config.datasets.seed)
    # data_indices = loader['test'].dataset.indices

    data = dataset[0]
    node_indices = torch.where(data.test_mask * data.y != 0)[0]

    gnnNets = get_gnnNets(input_dim, output_dim, config.models)
    cwd = os.path.dirname(os.path.abspath(__file__))
    state_dict = compatible_state_dict(torch.load(os.path.join(cwd,
                                                               config.models.gnn_saving_dir,
                                                               config.datasets.dataset_name,
                                                               f"{config.models.gnn_name}_"
                                                               f"{len(config.models.param.gnn_latent_dim)}l_best.pth"
                                                               ))['net'])

    gnnNets.load_state_dict(state_dict)

    # gnnNets.to_device()
    # gnnNets.eval()
    save_dir = os.path.join(cwd, 'results',
                            f"{config.datasets.dataset_name}",
                            f"{config.models.gnn_name}",
                            f"Multi_{config.explainers.param.reward_method}")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    plotutils = PlotUtils(dataset_name=config.datasets.dataset_name)
    fidelity_score_list = []
    sparsity_score_list = []
    ori_fide_list = []
    fidelity_inv = []
    h_fides = []

    data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
    prob = gnnNets(data.clone()).squeeze().softmax(dim=-1)
    start_time = time.time()

    for node_idx in tqdm(node_indices):
        # find the paths and build the graph
        result_path = os.path.join(save_dir, f"node_{node_idx}_score.pt")

        # get data and prediction

        _, prediction = torch.max(prob, -1)
        prediction = prediction[node_idx].item()

        # build the graph for visualization
        graph = to_networkx(data, to_undirected=True)
        node_labels = {k: int(v) for k, v in enumerate(data.y)}
        nx.set_node_attributes(graph, node_labels, 'label')

        #  searching using gnn score
        mcts_state_map = MCTS(node_idx=node_idx, ori_graph=graph,
                              X=data.x, edge_index=data.edge_index,
                              num_hops=len(config.models.param.gnn_latent_dim),
                              n_rollout=config.explainers.param.rollout,
                              min_atoms=config.explainers.param.min_atoms,
                              c_puct=config.explainers.param.c_puct,
                              expand_atoms=config.explainers.param.expand_atoms)
        value_func = GnnNets_NC2value_func(gnnNets,
                                           node_idx=mcts_state_map.node_idx,
                                           target_class=prediction)
        score_func = reward_func(config.explainers.param, value_func)
        mcts_state_map.set_score_func(score_func)

        # get searching result
        if os.path.isfile(result_path):
            gnn_results = torch.load(result_path)
        else:
            gnn_results = mcts_state_map.mcts(verbose=True)
            torch.save(gnn_results, result_path)
        tree_node_x = find_closest_node_result(gnn_results, gnnNets=gnnNets, data=data, config=config)

        # calculate the metrics
        tree_node_x = tree_node_x[0]
        original_node_list = [i for i in tree_node_x.ori_graph.nodes]

        maskout_node_list = [i for i in tree_node_x.ori_graph.nodes
                            if i not in tree_node_x.coalition or i == mcts_state_map.node_idx]
        masked_node_list = [node for node in tree_node_x.ori_graph.nodes
                            if node in tree_node_x.coalition]
        
        original_score = gnn_score(original_node_list, tree_node_x.data,
                                   value_func=value_func, subgraph_building_method='split')
        maskout_score = gnn_score(maskout_node_list, tree_node_x.data,
                                 value_func=value_func, subgraph_building_method='split')
        masked_score = gnn_score(masked_node_list, tree_node_x.data, value_func,
                                subgraph_building_method=config.explainers.param.subgraph_building_method)
        # sparsity_score = 1 - len(tree_node_x.coalition)/tree_node_x.ori_graph.number_of_nodes()
        sparsity_score = sparsity(tree_node_x.coalition, tree_node_x.data, config.explainers.param.subgraph_building_method)

        fidelity_score, eval_score = eval_metric(original_score, maskout_score, sparsity_score)
        fidelity_score_list.append(fidelity_score)
        sparsity_score_list.append(sparsity_score)
        ori_fide = original_score - maskout_score
        ori_fide_list.append(ori_fide)
        
        inv_f = original_score - masked_score
        fidelity_inv.append(inv_f)
        _, _, h_f = fidelity_normalize_and_harmonic_mean(ori_fide, inv_f, sparsity_score)
        h_fides.append(h_f)

        # visualization
        subgraph_node_labels = nx.get_node_attributes(tree_node_x.ori_graph, name='label')
        subgraph_node_labels = torch.tensor([v for k, v in subgraph_node_labels.items()])
        plotutils.plot(tree_node_x.ori_graph, tree_node_x.coalition, y=subgraph_node_labels,
                       node_idx=mcts_state_map.node_idx,
                       figname=os.path.join(save_dir, f"node_{node_idx}.png"))

    end_time = time.time()
    experiment_data = {
        'fidelity': sum(ori_fide_list) / len(ori_fide_list),
        'fidelity_inv': np.mean(fidelity_inv),
        'h_fidelity': np.mean(h_fides),
        'sparsity': sum(sparsity_score_list) / len(sparsity_score_list),
        'fidelity_abs': sum(fidelity_score_list) / len(fidelity_score_list),
        'Time in seconds': end_time - start_time,
        'Average Time': (end_time - start_time)/len(node_indices)
    }

    recorder.append(experiment_settings=['same', f"{config.explainers.max_ex_size}"],
                    experiment_data=experiment_data)

    recorder.save()

    fidelity_scores = torch.tensor(fidelity_score_list)
    sparsity_scores = torch.tensor(sparsity_score_list)
    print(f"fidelity score: {fidelity_scores.mean().item()}, sparsity score: {sparsity_scores.mean().item()}")
    return fidelity_scores, sparsity_scores


def sparsity(coalition: list, data, subgraph_building_method='zero_filling'):
    if subgraph_building_method == 'zero_filling':
        return 1.0 - len(coalition) / data.num_nodes

    elif subgraph_building_method == 'split':
        row, col = data.edge_index
        node_mask = torch.zeros(data.x.shape[0])
        node_mask[coalition] = 1.0
        edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
        return 1.0 - (edge_mask.sum() / edge_mask.shape[0]).item()


if __name__ == '__main__':
    import sys

    cwd = os.path.dirname(os.path.abspath(__file__))

    sys.argv.append('explainers=same')
    sys.argv.append(f"datasets.dataset_root={os.path.join(cwd, 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(cwd, 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(cwd, 'results')}")
    sys.argv.append(f"record_filename={os.path.join(cwd, 'result_jsons')}")
    pipeline()
