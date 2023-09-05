import os
import time
import numpy as np
import time
import hydra
import torch
from dig.xgraph.utils.compatibility import compatible_state_dict
from omegaconf import OmegaConf
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
from tqdm import tqdm
from gnnNets import get_gnnNets
from load_dataset import get_dataset, get_dataloader
from SAME.methods.initialization_mcts import MCTS, reward_func
from torch_geometric.data import Batch
from shapley import GnnNets_GC2value_func, gnn_score
from utils import PlotUtils, find_closest_node_result, Recorder, eval_metric, fidelity_normalize_and_harmonic_mean

IS_FRESH = False

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
    plotutils = PlotUtils(dataset_name=config.datasets.dataset_name)
    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes

    if config.datasets.dataset_name == 'mutag':
        data_indices = list(range(len(dataset)))
    else:
        dataloader_params = {'batch_size': config.models.param.batch_size,
                             'random_split_flag': config.datasets.random_split_flag,
                             'data_split_ratio': config.datasets.data_split_ratio,
                             'seed': config.datasets.seed}
        loader = get_dataloader(dataset, **dataloader_params)
        train_indices = loader['train'].dataset.indices
        test_indices = loader['test'].dataset.indices
        if config.datasets.dataset_name == 'mutag':
            test_indices = list(range(len(dataset)))
        # TODO: Partial
        import random
        random.seed(config.datasets.seed)
        random.shuffle(test_indices)
        if 'graph_sst' in config.datasets.dataset_name.lower():
            print('Using 30 data instances only...')
            test_indices = sorted(test_indices, key=lambda x: dataset[x].num_nodes, reverse=True)
            test_indices = [x for x in test_indices if dataset[x].num_nodes == 16]
            test_indices = test_indices[10:40]

    cwd = os.path.dirname(os.path.abspath(__file__))
    gnnNets = get_gnnNets(input_dim, output_dim, config.models)
    state_dict = compatible_state_dict(torch.load(os.path.join(cwd,
                                                               config.models.gnn_saving_dir,
                                                               config.datasets.dataset_name,
                                                               f"{config.models.gnn_name}_"
                                                               f"{len(config.models.param.gnn_latent_dim)}l_best.pth"
                                                               ))['net'])

    gnnNets.load_state_dict(state_dict)
    
    save_dir = os.path.join(cwd, 'results',
                            f"{config.datasets.dataset_name}",
                            f"{config.models.gnn_name}",
                            f"Multi_{config.explainers.param.reward_method}")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    abs_fidelity_score_list = []
    sparsity_score_list = []
    ori_fide_list = []
    inv_fide_list = []
    h_fides = []
    start_time = time.time()
    for i in tqdm(data_indices):
        # get data and prediction
        data = dataset[i]
        data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        probs = gnnNets(data.x, data.edge_index).squeeze().softmax(dim=-1)
        prediction = probs.squeeze().argmax(-1).item()
        original_score = probs[prediction]

        # get the reward func
        value_func = GnnNets_GC2value_func(gnnNets, target_class=data.y)
        payoff_func = reward_func(config.explainers.param, value_func)

        # find the paths and build the graph
        result_path = os.path.join(save_dir, f"example_{i}.pt")

        # mcts for l_shapely
        max_ex_size = np.ceil(data.num_nodes * (1-config.explainers.sparsity))
        mcts_state_map = MCTS(data.x, data.edge_index,
                              score_func=payoff_func,
                              n_rollout=config.explainers.param.rollout,
                              min_atoms=max_ex_size, 
                              c_puct=config.explainers.param.c_puct,
                              expand_atoms=config.explainers.param.expand_atoms,
                              high2low=config.explainers.param.high2low)

        if os.path.isfile(result_path) and not IS_FRESH:
            results = torch.load(result_path)
            print(f"Load Example {i}")
        else:
            results = mcts_state_map.mcts(verbose=True)
            torch.save(results, result_path)

        final_result_path = os.path.join(save_dir, f"example_{i}_final.pt")
        if os.path.isfile(final_result_path):
            final_results = torch.load(final_result_path)   # dict
            if final_results.get(config.explainers.sparsity) is not None:
                final_results = final_results.get(config.explainers.sparsity) # list
                print(f"Load Example {i} with final result.")
            else:
                new_final_results = find_closest_node_result(results, max_nodes=max_ex_size, gnnNets=gnnNets,
                                                            data=data, config=config).coalition
                final_results[config.explainers.sparsity] = new_final_results   # dict
                torch.save(final_results, final_result_path)
                final_results = new_final_results   # list
        else:
            # l sharply score
            final_results = find_closest_node_result(results, max_nodes=max_ex_size, gnnNets=gnnNets,
                                                     data=data, config=config).coalition
            tmp = dict()
            tmp[config.explainers.sparsity] = final_results
            torch.save(tmp, final_result_path)
        
        graph_node_exist = final_results
        graph_node_x = data

        maskout_node_list = [node for node in list(range(graph_node_x.x.shape[0]))
                                if node not in graph_node_exist]
        mask_node_list = [node for node in list(range(graph_node_x.x.shape[0]))
                            if node in graph_node_exist]
        

        maskout_score = gnn_score(maskout_node_list, graph_node_x, value_func,
                                    subgraph_building_method=config.explainers.param.subgraph_building_method)
        masked_score = gnn_score(mask_node_list, graph_node_x, value_func,
                                subgraph_building_method=config.explainers.param.subgraph_building_method)

        sparsity_score = sparsity(graph_node_exist, graph_node_x, config.explainers.param.subgraph_building_method)

        abs_fidelity_score, eval_score = eval_metric(original_score, maskout_score, sparsity_score)
        ori_fide = (original_score - maskout_score).item()
        ori_fide_list.append(ori_fide)
        abs_fidelity_score_list.append(abs_fidelity_score)
        sparsity_score_list.append(sparsity_score)
       
        inv_f = (original_score - masked_score).item()
        inv_fide_list.append(inv_f)
        _, _, h_f = fidelity_normalize_and_harmonic_mean(ori_fide, inv_f, sparsity_score)
        h_fides.append(h_f)
        
        # visualization
        if hasattr(dataset, 'supplement'):
            words = dataset.supplement['sentence_tokens'][str(i)]
            plotutils.plot(mcts_state_map.graph, graph_node_exist, words=words,
                           figname=os.path.join(save_dir, f"example_{i}.png"),
                           title_sentence=f'fidelity: {ori_fide:.4f}, sparsity: {sparsity_score:.4f}')
        else:
            plotutils.plot(mcts_state_map.graph, graph_node_exist, x=graph_node_x.x,
                           figname=os.path.join(save_dir, f"example_{i}.png"),
                           title_sentence=f'fidelity: {ori_fide:.4f}, sparsity: {sparsity_score:.4f}')
            

    end_time = time.time()
    experiment_data = {
        'fidelity': np.mean(ori_fide_list),
        'inv_fidelity': np.mean(inv_fide_list),
        'h_fidelity': np.mean(h_fides),
        'sparsity': np.mean(sparsity_score_list),
        'STD of sparsity': np.std(sparsity_score_list),
        'fidelity_abs': np.mean(abs_fidelity_score_list),
        'STD of fidelity_abs': np.std(abs_fidelity_score_list),
        'Time in seconds': end_time - start_time,
        'Average Time': (end_time - start_time)/len(data_indices)
    }

    recorder.append(experiment_settings=['same', f"{config.explainers.sparsity}"],
                    experiment_data=experiment_data)

    recorder.save()

    fidelity_scores = torch.tensor(ori_fide_list)
    sparsity_scores = torch.tensor(sparsity_score_list)
    
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
