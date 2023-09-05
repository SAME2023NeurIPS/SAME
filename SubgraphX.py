import os
import argparse
import numpy as np
import torch
import time
import hydra
from omegaconf import OmegaConf
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import check_dir, Recorder, eval_metric, PlotUtils, fidelity_normalize_and_harmonic_mean
from gnnNets import get_gnnNets
from load_dataset import get_dataset, get_dataloader

from dig.xgraph.method import SubgraphX
from dig.xgraph.dataset import SynGraphDataset
# from dig.xgraph.method.subgraphx import PlotUtils
from dig.xgraph.evaluation import XCollector
from dig.xgraph.method.subgraphx import find_closest_node_result
from dig.xgraph.utils.compatibility import compatible_state_dict

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

    abs_fides = []
    fides_ori = []
    spars = []
    h_f_list = []
    inv_f_list = []

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    dataset = get_dataset(config.datasets.dataset_root,
                          config.datasets.dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:
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
        test_indices = test_indices[:30] 

    model = get_gnnNets(input_dim=dataset.num_node_features,
                        output_dim=dataset.num_classes,
                        model_config=config.models)

    state_dict = compatible_state_dict(torch.load(os.path.join(
        config.models.gnn_saving_dir,
        config.datasets.dataset_name,
        f"{config.models.gnn_name}_"
        f"{len(config.models.param.gnn_latent_dim)}l_best.pth"
    ))['net'])

    model.load_state_dict(state_dict)

    explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                          config.datasets.dataset_name,
                                          config.models.gnn_name,
                                          config.explainers.param.reward_method)
    check_dir(explanation_saving_dir)
    plot_utils = PlotUtils(dataset_name=config.datasets.dataset_name, is_show=False)    

    start_time = time.time()
    if config.models.param.graph_classification:
        subgraphx = SubgraphX(model,
                              dataset.num_classes,
                              device,
                              explain_graph=config.models.param.graph_classification,
                              verbose=config.explainers.param.verbose,
                              c_puct=config.explainers.param.c_puct,
                              rollout=config.explainers.param.rollout,
                              high2low=config.explainers.param.high2low,
                              min_atoms=config.explainers.param.min_atoms,
                              expand_atoms=config.explainers.param.expand_atoms,
                              reward_method=config.explainers.param.reward_method,
                              subgraph_building_method=config.explainers.param.subgraph_building_method,
                              save_dir=explanation_saving_dir)
        index = 0
        x_collector = XCollector()
        for i, data in enumerate(dataset[test_indices]):
            # if test_indices[i] != 173:
            #     continue
            index += 1
            # if index == 1:
            #     continue
            data.to(device)
            data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            saved_MCTSInfo_list = None
            prediction = model(data).argmax(-1).item()
            if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt')) and not IS_FRESH:
                pass
                saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))
                print(f"load example {test_indices[i]}.")
            # else:
            #     continue

            max_nodes = np.ceil(data.num_nodes * (1-config.explainers.sparsity))
            explain_result, related_preds = \
                subgraphx.explain(data.x, data.edge_index,
                                  max_nodes=max_nodes,
                                  label=prediction,
                                  saved_MCTSInfo_list=saved_MCTSInfo_list)

            torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))

            spar = related_preds["sparsity"]
            f, score = eval_metric(torch.tensor(related_preds["origin"]), torch.tensor(related_preds["maskout"]), spar)

            fo = torch.tensor(related_preds["origin"]) - torch.tensor(related_preds["maskout"])
            inv_f = torch.tensor(related_preds["origin"]) - torch.tensor(related_preds["masked"])
            fides_ori.append(fo.item())
            abs_fides.append(f)
            spars.append(spar)
            
            _, _, h_f = fidelity_normalize_and_harmonic_mean(fo.item(), inv_f.item(), spar)
            inv_f_list.append(inv_f.item())
            h_f_list.append(h_f)

            title_sentence = f'fide: {f:.4f}, ' \
                             f'spar: {spar:.4f}'

            explain_result = subgraphx.read_from_MCTSInfo_list(explain_result)
            explanation = find_closest_node_result(explain_result, max_nodes=max_nodes)

            if isinstance(dataset, SynGraphDataset):
                explanation = find_closest_node_result(explain_result, max_nodes=max_nodes)
                edge_mask = data.edge_index[0].cpu().apply_(lambda x: x in explanation.coalition).bool() & \
                            data.edge_index[1].cpu().apply_(lambda x: x in explanation.coalition).bool()
                edge_mask = edge_mask.float().numpy()
                motif_edge_mask = dataset.gen_motif_edge_mask(data).float().cpu().numpy()
                accuracy = accuracy_score(edge_mask, motif_edge_mask)
                roc_auc = roc_auc_score(edge_mask, motif_edge_mask)
                related_preds['accuracy'] = roc_auc

            if hasattr(dataset, 'supplement'):
                words = dataset.supplement['sentence_tokens'][str(test_indices[i])]
            else:
                words = None

            predict_true = 'True' if prediction == data.y.item() else "False"

            subgraphx.visualization(explain_result,
                                    max_nodes=max_nodes,
                                    plot_utils=plot_utils,
                                    title_sentence=title_sentence,
                                    vis_name=os.path.join(explanation_saving_dir,
                                                          f'example_{test_indices[i]}_'
                                                          f'prediction_{prediction}_'
                                                          f'label_{data.y.item()}_'
                                                          f'pred_{predict_true}.png'),
                                    words=words)

            explain_result = [explain_result]
            related_preds = [related_preds]
            x_collector.collect_data(explain_result, related_preds, label=0)

    else:
        x_collector = XCollector()
        data = dataset.data
        data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
        test_indices = node_indices
        predictions = model(data).argmax(-1)

        subgraphx = SubgraphX(model,
                              dataset.num_classes,
                              device,
                              explain_graph=config.models.param.graph_classification,
                              verbose=config.explainers.param.verbose,
                              c_puct=config.explainers.param.c_puct,
                              rollout=config.explainers.param.rollout,
                              high2low=config.explainers.param.high2low,
                              min_atoms=config.explainers.param.min_atoms,
                              expand_atoms=config.explainers.param.expand_atoms,
                              reward_method=config.explainers.param.reward_method,
                              subgraph_building_method=config.explainers.param.subgraph_building_method,
                              save_dir=explanation_saving_dir)

        for node_idx in node_indices:
            data.to(device)
            saved_MCTSInfo_list = None

            if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{node_idx}.pt')) and not IS_FRESH:
                saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir,
                                                              f'example_{node_idx}.pt'))
                print(f"load example {node_idx}.")

            explain_result, related_preds = \
                subgraphx.explain(data.x, data.edge_index,
                                  node_idx=node_idx,
                                  max_nodes=config.explainers.max_ex_size,
                                  label=predictions[node_idx].item(),
                                  saved_MCTSInfo_list=saved_MCTSInfo_list)

            torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{node_idx}.pt'))

            spar = related_preds["sparsity"]
            f, score = eval_metric(related_preds["origin"], related_preds["maskout"], spar)
            fo = related_preds["origin"] - related_preds["maskout"]

            title_sentence = f'fide: {(related_preds["origin"] - related_preds["maskout"]):.3f}, ' \
                             f'fide_inv: {(related_preds["origin"] - related_preds["masked"]):.3f}, ' \
                             f'spar: {related_preds["sparsity"]:.3f}'
            title_sentence = f'fide: {f:.4f}, ' \
                             f'spar: {spar:.4f}, ' \
                             f'score: {score:.4f}, gnn: {related_preds["origin"]:.4f}'
            fides_ori.append(fo)
            abs_fides.append(f)
            spars.append(spar)
            
            _, _, h_f = fidelity_normalize_and_harmonic_mean(fo.item(), inv_f.item(), spar)
            inv_f_list.append(inv_f.item())
            h_f_list.append(h_f)

            explain_result = subgraphx.read_from_MCTSInfo_list(explain_result)

            subgraphx.visualization(explain_result,
                                    y=data.y,
                                    max_nodes=config.explainers.max_ex_size,
                                    plot_utils=plot_utils,
                                    title_sentence=title_sentence,
                                    vis_name=os.path.join(explanation_saving_dir,
                                                          f'example_{node_idx}.png'))

            explain_result = [explain_result]
            related_preds = [related_preds]
            x_collector.collect_data(explain_result, related_preds, label=0)

    end_time = time.time()
    
    experiment_data = {
        'fidelity': np.mean(fides_ori),
        'fidelity_inv': x_collector.fidelity_inv,
        'h_fidelity': np.mean(h_f_list),
        'sparsity': np.mean(spars),
        'STD of sparsity': np.std(spars),
        'fidelity_abs': np.mean(abs_fides),
        'STD of fidelity_abs': np.std(abs_fides),
        'Time in seconds': end_time - start_time,
        'Average Time': (end_time - start_time)/len(test_indices)
    }

    recorder.append(experiment_settings=['subgraphx', f"{config.explainers.sparsity}"],
                    experiment_data=experiment_data)

    recorder.save()


if __name__ == '__main__':
    import sys
    
    cwd = os.path.dirname(os.path.abspath(__file__))   
        
    sys.argv.append('explainers=subgraphx')
    sys.argv.append(f"datasets.dataset_root={os.path.join(cwd, 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(cwd, 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(cwd, 'results')}")
    sys.argv.append(f"record_filename={os.path.join(cwd, 'result_jsons')}")
    pipeline()
