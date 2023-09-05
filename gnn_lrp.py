import os
import torch
import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, to_undirected
import time
from gnnNets import get_gnnNets
from load_dataset import get_dataset, get_dataloader
from utils import check_dir, fix_random_seed, Recorder, perturb_input, eval_metric, PlotUtils, to_networkx
from dig.xgraph.evaluation import XCollector
from dig.xgraph.method import GNN_LRP
from dig.xgraph.dataset import SynGraphDataset
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
    # fix_random_seed(config.random_seed)
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

    model.load_state_dict(state_dict)

    model.to(device)
    fides_abs = []
    fides_ori = []
    spars = []
    scores = []

    plotutils = PlotUtils(dataset_name=config.datasets.dataset_name)
    explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                          config.datasets.dataset_name,
                                          config.models.gnn_name,
                                          'GNNLRP')
    check_dir(explanation_saving_dir)      
            
    gnnlrp_explainer = GNN_LRP(model, explain_graph=config.models.param.graph_classification)
    gnnlrp_explainer_perturb = GNN_LRP(model, explain_graph=config.models.param.graph_classification)

    index = 0
    x_collector = XCollector()
    start_time = time.time()
    if config.models.param.graph_classification:
        for i, data in enumerate(dataset[test_indices]):
            index += 1
            data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            data.to(device)
            if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt')) and not IS_FRESH:
                walks = torch.load(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))
                walks = {k: v.to(device) for k, v in walks.items()}
                print(f"load example {test_indices[i]}.")
                walks, masks, related_preds = \
                    gnnlrp_explainer(data.x, data.edge_index,
                                     sparsity=config.explainers.sparsity,
                                     num_classes=dataset.num_classes,
                                     walks=walks)
            else:
                print(f"GNNLRP explain example {test_indices[i]}.")
                walks, masks, related_preds = \
                    gnnlrp_explainer(data.x, data.edge_index,
                                     sparsity=config.explainers.sparsity,
                                     num_classes=dataset.num_classes)

                walks = {k: v.to('cpu') for k, v in walks.items()}
                torch.save(walks, os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))
            prediction = model(data).argmax(-1).item()

            if isinstance(dataset, SynGraphDataset):
                motif_edge_mask = dataset.gen_motif_edge_mask(data)
                roc_aucs = [roc_auc_score(motif_edge_mask.float().cpu().numpy(), edge_mask.float().cpu().numpy())
                            for edge_mask in masks]
                for target_label, related_pred in enumerate(related_preds):
                    related_preds[target_label]['accuracy'] = roc_aucs[target_label]
                    
            if hasattr(dataset, "supplement"):
                words = dataset.supplement["sentence_tokens"][str(test_indices[i])]
            else:
                words = None
                
            mask = masks[data.y].clone().detach().cpu()
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
            
            x_collector.collect_data(masks, related_preds, label=prediction)
            fide, score = eval_metric(related_preds[prediction]["origin"], related_preds[prediction]["maskout"],
                                      related_preds[prediction]["sparsity"])
            fides_abs.append(fide)
            fides_ori.append(related_preds[prediction]["origin"] - related_preds[prediction]["maskout"])
            scores.append(score)
            spars.append(related_preds[prediction]["sparsity"])
            
            title_sentence = f'fide: {fides_ori[-1]:.4f}, spar: {spars[-1]:.4f}'
            plotutils.plot(to_networkx(data, remove_self_loops=True), node_list, words=words, x=data.x.detach(),
                figname=os.path.join(explanation_saving_dir, f"example_{test_indices[i]}.png"),
                title_sentence=title_sentence)

    else:
        data = dataset.data
        data.edge_index = add_remaining_self_loops(data.edge_index)[0]
        data.to(device)
        prediction = model(data).argmax(-1)
        try:
            for node_idx in node_indices:
                if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{node_idx}.pt')):
                    walks = torch.load(os.path.join(explanation_saving_dir, f'example_{node_idx}.pt'))
                    walks = {k: v.to(device) for k, v in walks.items()}
                    print(f"load example {node_idx}.")
                    walks, masks, related_preds = \
                        gnnlrp_explainer(data.x, data.edge_index,
                                         node_idx=node_idx,
                                         sparsity=config.explainers.sparsity,
                                         num_classes=dataset.num_classes,
                                         walks=walks)
                else:
                    walks, masks, related_preds = \
                        gnnlrp_explainer(data.x, data.edge_index,
                                         node_idx=node_idx,
                                         sparsity=config.explainers.sparsity,
                                         num_classes=dataset.num_classes)
                    walks = {k: v.to('cpu') for k, v in walks.items()}
                    torch.save(walks, os.path.join(explanation_saving_dir, f'example_{node_idx}.pt'))

                if isinstance(dataset, SynGraphDataset):
                    motif_edge_mask = dataset.gen_motif_edge_mask(data, node_idx=node_idx)
                    motif_edge_mask = motif_edge_mask[gnnlrp_explainer.hard_edge_mask].float()
                    masks = [edge_mask[gnnlrp_explainer.hard_edge_mask] for edge_mask in masks]
                    if not (motif_edge_mask == 1).all() and motif_edge_mask.shape[0] < 1000:
                        roc_aucs = [roc_auc_score(motif_edge_mask.cpu().numpy(), edge_mask.cpu().numpy())
                                    for edge_mask in masks]
                        for target_label, related_pred in enumerate(related_preds):
                            related_preds[target_label]['accuracy'] = roc_aucs[target_label]

                        # for the stability metric
                        all_roc_aucs = []
                        for label_idx in range(dataset.num_classes):
                            all_roc_aucs.append([])
                        perturb_input_list = perturb_input(data, gnnlrp_explainer.hard_edge_mask, gnnlrp_explainer.subset)
                        for perturb_data in perturb_input_list:
                            new_prediction = model(perturb_data)[gnnlrp_explainer.new_node_idx].argmax(dim=-1).item()
                            perturb_walks, perturb_masks, perturb_related_preds = \
                                gnnlrp_explainer_perturb(perturb_data.x,
                                                         perturb_data.edge_index,
                                                         node_idx=gnnlrp_explainer.new_node_idx,
                                                         sparsity=config.explainers.sparsity,
                                                         num_classes=dataset.num_classes)

                            perturb_motif_edge_mask = torch.cat([
                                motif_edge_mask, torch.zeros(perturb_data.edge_index.shape[1] - motif_edge_mask.shape[0])],
                                dim=0)
                            if not (perturb_motif_edge_mask == 1).all():
                                perturb_roc_aucs = [
                                    roc_auc_score(perturb_motif_edge_mask.cpu().numpy(), edge_mask.cpu().numpy())
                                    for edge_mask in perturb_masks]
                                for label_idx in range(dataset.num_classes):
                                    all_roc_aucs[label_idx].append(
                                        related_preds[label_idx]['accuracy'] - perturb_roc_aucs[label_idx])

                        for target_label, related_pred in enumerate(related_preds):
                            related_preds[target_label]['stability'] = torch.tensor(
                                all_roc_aucs[target_label]).mean().item()

                x_collector.collect_data(masks, related_preds, label=prediction[node_idx].item())
                fide, score = eval_metric(related_preds[prediction[node_idx].item()]["origin"],
                                          related_preds[prediction[node_idx].item()]["maskout"],
                                          related_preds[prediction[node_idx].item()]["sparsity"])
                fides_abs.append(fide)
                fides_ori.append(
                    related_preds[prediction[node_idx].item()]["origin"] - related_preds[prediction[node_idx].item()][
                        "maskout"])
                scores.append(score)
                spars.append(related_preds[prediction[node_idx].item()]["sparsity"])
        except e:
            print(e)
            pass

    end_time = time.time()

    experiment_data = {
        'fidelity': np.mean(fides_ori),
        'STD of fidelity': np.std(fides_ori),
        'sparsity': np.mean(spars),
        'STD of sparsity': np.std(spars),
        'fidelity_abs': np.mean(fides_abs),
        'STD of fidelity_abs': np.std(fides_abs),
        'Time in seconds': end_time - start_time,
        'Average Time': (end_time - start_time)/len(test_indices)
    }

    recorder.append(experiment_settings=['gnn_lrp', f"{config.explainers.sparsity}"],
                    experiment_data=experiment_data)

    recorder.save()


if __name__ == '__main__':
    import sys
    sys.argv.append('explainers=gnn_lrp')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")
    sys.argv.append(f"record_filename={os.path.join(os.path.dirname(__file__), 'result_jsons')}")
    pipeline()
