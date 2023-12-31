import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
import time
from gnnNets import get_gnnNets
from load_dataset import get_dataset, get_dataloader
from utils import check_dir, fix_random_seed, Recorder, perturb_input, eval_metric, PlotUtils, to_networkx, fidelity_normalize_and_harmonic_mean

from dig.xgraph.method import GNNExplainer
from dig.xgraph.evaluation import XCollector
from dig.xgraph.utils.compatibility import compatible_state_dict
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, to_undirected
IS_FRESH = False

@hydra.main(config_path="../config", config_name="config")
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

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    # bbbp warning
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
        test_indices = loader['test'].dataset.indices
        if config.datasets.dataset_name == 'mutag':
            test_indices = list(range(len(dataset)))

        import random
        random.seed(config.datasets.seed)
        random.shuffle(test_indices)
        if 'graph_sst' in config.datasets.dataset_name.lower():
            print('Using 30 data instances only...')
            test_indices = sorted(test_indices, key=lambda x: dataset[x].num_nodes, reverse=True)
            test_indices = [x for x in test_indices if dataset[x].num_nodes == 16]
            test_indices = test_indices[10:40]
            
    else:
        node_indices_mask = (dataset.data.y != 0) * dataset.data.test_mask
        node_indices = torch.where(node_indices_mask)[0]
        test_indices = node_indices

    model = get_gnnNets(input_dim=dataset.num_node_features,
                        output_dim=dataset.num_classes,
                        model_config=config.models)

    state_dict = compatible_state_dict(torch.load(os.path.join(
        config.models.gnn_saving_dir,
        config.datasets.dataset_name,
        f"{config.models.gnn_name}_"
        f"{len(config.models.param.gnn_latent_dim)}l_best.pth"
    ))['net'])

    fides_abs = []
    fides_ori = []
    inv_fide_list = []
    h_fides = []
    spars = []
    scores = []

    model.load_state_dict(state_dict)
    model.to(device)
    
    plotutils = PlotUtils(dataset_name=config.datasets.dataset_name)
    explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                          config.datasets.dataset_name,
                                          config.models.gnn_name,
                                          'GNNExplainer')
    check_dir(explanation_saving_dir)
            
    gnn_explainer = GNNExplainer(model,
                                 epochs=config.explainers.param.epochs,
                                 lr=config.explainers.param.lr,
                                 coff_size=config.explainers.param.coff_size,
                                 coff_ent=config.explainers.param.coff_ent,
                                 explain_graph=config.models.param.graph_classification)
    gnn_explainer_perturb = GNNExplainer(model,
                                 epochs=config.explainers.param.epochs,
                                 lr=config.explainers.param.lr,
                                 coff_size=config.explainers.param.coff_size,
                                 coff_ent=config.explainers.param.coff_ent,
                                 explain_graph=config.models.param.graph_classification)
    gnn_explainer.device = device

    index = 0
    x_collector = XCollector()
    start_time = time.time()
    if config.models.param.graph_classification:
        for i, data in tqdm(enumerate(dataset[test_indices])):
            index += 1
            data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            data.to(device)
            prediction = model(data).argmax(-1).item()

            if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt')) and not IS_FRESH:
                edge_masks = torch.load(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))
                edge_masks = [edge_mask.to(device) for edge_mask in edge_masks]
                print(f"load example {test_indices[i]}.")
                edge_masks, hard_edge_masks, related_preds = \
                    gnn_explainer(data.x, data.edge_index,
                                  sparsity=config.explainers.sparsity,
                                  num_classes=dataset.num_classes,
                                  edge_masks=edge_masks)

            else:
                edge_masks, hard_edge_masks, related_preds = \
                    gnn_explainer(data.x, data.edge_index,
                                  sparsity=config.explainers.sparsity,
                                  num_classes=dataset.num_classes)
                edge_masks = [edge_mask.to('cpu') for edge_mask in edge_masks]
                torch.save(edge_masks, os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))

            if hasattr(dataset, "supplement"):
                words = dataset.supplement["sentence_tokens"][str(test_indices[i])]
            else:
                words = None
            
            mask = edge_masks[prediction].clone().detach().cpu()
            edge_index, edge_attr = remove_self_loops(data.edge_index.clone().cpu(), mask)
            edge_index, edge_attr = to_undirected(edge_index, edge_attr)
            mask = np.sort(edge_attr)
            threshold = mask[int(len(mask)*(config.explainers.sparsity))]
            
            edge_attr = torch.tensor(edge_attr)
            
            binary_mask = torch.where(edge_attr > threshold, torch.ones_like(edge_attr), torch.zeros_like(edge_attr))
            edge_indices = torch.nonzero(torch.tensor(binary_mask), as_tuple=False).t()
            edge_indices = torch.tensor(edge_indices,dtype=torch.long)
            node_mask = torch.unique(edge_index[:, edge_indices].flatten())
            
            node_list = [x.item() for x in node_mask]
            
            x_collector.collect_data(hard_edge_masks, related_preds, label=prediction)
            abs_fide, score = eval_metric(related_preds[prediction]["origin"], related_preds[prediction]["maskout"],
                                          related_preds[prediction]["sparsity"])
            inv_f = related_preds[prediction]["origin"] - related_preds[prediction]["masked"]
            ori_f = related_preds[prediction]["origin"] - related_preds[prediction]["maskout"]
            sp = related_preds[prediction]["sparsity"]

            fides_abs.append(abs_fide)
            fides_ori.append(ori_f)
            inv_fide_list.append(inv_f)
            _, _, h_f = fidelity_normalize_and_harmonic_mean(ori_f, inv_f, sp)
            h_fides.append(h_f)

            scores.append(score)
            spars.append(sp)
            
            title_sentence = f'fide: {fides_ori[-1]:.4f}, spar: {spars[-1]:.4f}'
            
            if config.save_plot:
                plotutils.plot(to_networkx(data, remove_self_loops=True), node_list, words=words, x=data.x.detach(),
                            figname=os.path.join(explanation_saving_dir, f"example_{test_indices[i]}.png"),
                            title_sentence=title_sentence)

    else:
        data = dataset.data
        data.edge_index = add_remaining_self_loops(data.edge_index)[0]
        data.to(device)
        prediction = model(data).argmax(-1)
        for node_idx in node_indices:
            # if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{node_idx}.pt')):
            #     edge_masks = torch.load(os.path.join(explanation_saving_dir, f'example_{node_idx}.pt'))
            #     edge_masks = [edge_mask.to(device) for edge_mask in edge_masks]
            #     print(f"load example {node_idx}.")
            #     edge_masks, hard_edge_masks, related_preds = \
            #         gnn_explainer(data.x, data.edge_index,
            #                       node_idx=node_idx,
            #                       sparsity=config.explainers.sparsity,
            #                       num_classes=dataset.num_classes,
            #                       edge_masks=edge_masks)
            # else:
            edge_masks, hard_edge_masks, related_preds = \
                gnn_explainer(data.x, data.edge_index,
                              node_idx=node_idx,
                              sparsity=config.explainers.sparsity,
                              num_classes=dataset.num_classes)

            edge_masks = [edge_mask.to('cpu') for edge_mask in edge_masks]
            torch.save(edge_masks, os.path.join(explanation_saving_dir, f'example_{node_idx}.pt'))

            x_collector.collect_data(edge_masks, related_preds, label=prediction[node_idx].item())
            abs_fide, score = eval_metric(related_preds[prediction[node_idx].item()]["origin"],
                                      related_preds[prediction[node_idx].item()]["maskout"],
                                      related_preds[prediction[node_idx].item()]["sparsity"])
            
            inv_f = related_preds[prediction[node_idx].item()]["origin"] - related_preds[prediction[node_idx].item()]["masked"]
            ori_f = related_preds[prediction[node_idx].item()]["origin"] - related_preds[prediction[node_idx].item()]["maskout"]
            sp = related_preds[prediction[node_idx].item()]["sparsity"]

            fides_abs.append(abs_fide)
            fides_ori.append(ori_f)
            inv_fide_list.append(inv_f)
            _, _, h_f = fidelity_normalize_and_harmonic_mean(ori_f, inv_f, sp)
            h_fides.append(h_f)
            fides_abs.append(ori_f)
            scores.append(score)
            spars.append(sp)

    end_time = time.time()

    experiment_data = {
        'fidelity': np.mean(fides_ori),
        'inv_fidelity': np.mean(inv_fide_list),
        'h_fidelity': np.mean(h_fides),
        'sparsity': np.mean(spars),
        'STD of sparsity': np.std(spars),       
        'Time in seconds': end_time - start_time,
        'Average Time': (end_time - start_time)/len(test_indices)
    }

    recorder.append(experiment_settings=['gnn_explainer', f"{config.explainers.sparsity}"],
                    experiment_data=experiment_data)

    recorder.save()


if __name__ == '__main__':
    import sys
    wkdir = os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    sys.argv.append('explainers=gnn_explainer')
    sys.argv.append(f"datasets.dataset_root={os.path.join(wkdir, 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(wkdir, 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(wkdir, 'results')}")
    sys.argv.append(f"record_filename={os.path.join(wkdir, 'result_jsons')}")
    pipeline()
