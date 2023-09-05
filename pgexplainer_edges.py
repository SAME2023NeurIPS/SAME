import os
import torch
import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
import time
from dig.xgraph.method import PGExplainer
from dig.xgraph.evaluation import XCollector
from dig.xgraph.method.base_explainer import ExplainerBase
from dig.xgraph.utils.compatibility import compatible_state_dict

from gnnNets import get_gnnNets
from load_dataset import get_dataset, get_dataloader, SynGraphDataset
from utils import check_dir, fix_random_seed, Recorder, perturb_input, eval_metric, PlotUtils, to_networkx

from torch import Tensor
from typing import List, Dict, Tuple
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, to_undirected
IS_FRESH = False

class PGExplainer_edges(ExplainerBase):
    def __init__(self, pgexplainer, model, molecule: bool):
        super().__init__(model=model,
                         explain_graph=pgexplainer.explain_graph,
                         molecule=molecule)
        self.explainer = pgexplainer

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs)\
            -> Tuple[List, List, List[Dict]]:
        # set default subgraph with 10 edges

        pred_label = kwargs.get('pred_label')
        num_classes = kwargs.get('num_classes')
        self.model.eval()
        self.explainer.__clear_masks__()

        x = x.to(self.device)
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = edge_index.to(self.device)

        if self.explain_graph:
            embed = self.model.get_emb(x, edge_index)
            _, edge_mask = self.explainer.explain(x,
                                                  edge_index,
                                                  embed=embed,
                                                  tmp=1.0,
                                                  training=False)
            # edge_masks
            edge_masks = [edge_mask for _ in range(num_classes)]
            # Calculate mask
            hard_edge_masks = [self.control_sparsity(edge_mask, sparsity=kwargs.get('sparsity')).sigmoid()
                               for _ in range(num_classes)]

            self.__clear_masks__()
            self.__set_masks__(x, edge_index)
            with torch.no_grad():
                if self.explain_graph:
                    related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks)

            self.__clear_masks__()

        else:
            node_idx = kwargs.get('node_idx')
            sparsity = kwargs.get('sparsity')
            assert kwargs.get('node_idx') is not None, "please input the node_idx"
            select_edge_index = torch.arange(0, edge_index.shape[1])
            subgraph_x, subgraph_edge_index, _, subset, kwargs = \
                self.explainer.get_subgraph(node_idx, x, edge_index, select_edge_index=select_edge_index)
            select_edge_index = kwargs['select_edge_index']
            self.select_edge_mask = edge_index.new_empty(edge_index.size(1),
                                                         device=self.device,
                                                         dtype=torch.bool)
            self.select_edge_mask.fill_(False)
            self.select_edge_mask[select_edge_index] = True
            self.hard_edge_mask = edge_index.new_empty(subgraph_edge_index.size(1),
                                                       device=self.device,
                                                       dtype=torch.bool)
            self.hard_edge_mask.fill_(True)
            self.subset = subset
            self.new_node_idx = torch.where(subset == node_idx)[0]

            subgraph_embed = self.model.get_emb(subgraph_x, subgraph_edge_index)
            _, subgraph_edge_mask = self.explainer.explain(subgraph_x,
                                                           subgraph_edge_index,
                                                           embed=subgraph_embed,
                                                           tmp=1.0,
                                                           training=False,
                                                           node_idx=self.new_node_idx)

            # edge_masks
            edge_masks = [subgraph_edge_mask for _ in range(num_classes)]
            # Calculate mask
            hard_edge_masks = [
                self.control_sparsity(subgraph_edge_mask, sparsity=sparsity).sigmoid()
                for _ in range(num_classes)]

            self.__clear_masks__()
            self.__set_masks__(subgraph_x, subgraph_edge_index)
            with torch.no_grad():
                related_preds = self.eval_related_pred(
                    subgraph_x, subgraph_edge_index, hard_edge_masks, node_idx=self.new_node_idx)

            self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds


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
            train_indices = list(range(len(dataset)))
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
        train_indices = range(len(dataset))

    model = get_gnnNets(input_dim=dataset.num_node_features,
                        output_dim=dataset.num_classes,
                        model_config=config.models)
    eval_model = get_gnnNets(input_dim=dataset.num_node_features,
                             output_dim=dataset.num_classes,
                             model_config=config.models)

    state_dict = torch.load(os.path.join(config.models.gnn_saving_dir,
                                         config.datasets.dataset_name,
                                         f"{config.models.gnn_name}_"
                                         f"{len(config.models.param.gnn_latent_dim)}l_best.pth"))['net']

    state_dict = compatible_state_dict(state_dict)
    model.load_state_dict(state_dict)
    eval_model.load_state_dict(state_dict)

    model.to(device)
    eval_model.to(device)

    fides_abs = []
    fides_ori = []
    spars = []
    scores = []
    
    plotutils = PlotUtils(dataset_name=config.datasets.dataset_name)
    explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                          config.datasets.dataset_name,
                                          config.models.gnn_name,
                                          'PGExplainer')
    check_dir(explanation_saving_dir)

    if config.models.param.graph_classification:
        input_dim = config.models.param.gnn_latent_dim[-1] * 2
    else:
        input_dim = config.models.param.gnn_latent_dim[-1] * 3

    pgexplainer = PGExplainer(model,
                              in_channels=input_dim,
                              device=device,
                              explain_graph=config.models.param.graph_classification,
                              epochs=config.explainers.param.ex_epochs,
                              lr=config.explainers.param.ex_learning_rate,
                              coff_size=config.explainers.param.coff_size,
                              coff_ent=config.explainers.param.coff_ent,
                              sample_bias=config.explainers.param.sample_bias,
                              t0=config.explainers.param.t0,
                              t1=config.explainers.param.t1)

    pgexplainer_saving_path = os.path.join(config.explainers.explainer_saving_dir,
                                           config.datasets.dataset_name,
                                           config.explainers.explainer_saving_name)

    train_time = None
    if os.path.isfile(pgexplainer_saving_path) and not IS_FRESH:
        print("Load saved PGExplainer model...")
        state_dict = torch.load(pgexplainer_saving_path)
        state_dict = compatible_state_dict(state_dict)
        pgexplainer.load_state_dict(state_dict)
    else:
        start_time = time.time()
        if config.models.param.graph_classification:
            pgexplainer.train_explanation_network(dataset[train_indices])
        else:
            pgexplainer.train_explanation_network(dataset)
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        state_dict = torch.load(pgexplainer_saving_path)
        state_dict = compatible_state_dict(state_dict)
        pgexplainer.load_state_dict(state_dict)
        end_time = time.time()
        train_time = end_time - start_time
        print(train_time)
        return

    index = 0
    x_collector = XCollector()
    pgexplainer_edges = PGExplainer_edges(pgexplainer=pgexplainer,
                                          model=eval_model,
                                          molecule=True)
    pgexplainer_edges.device = pgexplainer.device
    start_time = time.time()
    if config.models.param.graph_classification:
        for i, data in enumerate(dataset[test_indices]):
            index += 1
            #data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            data.to(device)
            prediction = model(data).softmax(dim=-1).argmax().item()
            edge_masks, hard_edge_masks, related_preds = \
                pgexplainer_edges(data.x, data.edge_index,
                                  num_classes=dataset.num_classes,
                                  sparsity=config.explainers.sparsity)

            if isinstance(dataset, SynGraphDataset):
                motif_edge_mask = dataset.gen_motif_edge_mask(data)
                roc_aucs = [roc_auc_score(motif_edge_mask.cpu().numpy(), edge_mask.detach().cpu().numpy())
                            for edge_mask in edge_masks]
                for target_label, related_pred in enumerate(related_preds):
                    related_preds[target_label]['accuracy'] = roc_aucs[target_label]
                    
            if hasattr(dataset, "supplement"):
                words = dataset.supplement["sentence_tokens"][str(test_indices[i])]
            else:
                words = None         
            
            idx = test_indices[i]
            related_preds = related_preds[prediction]
            hard_edge_masks = hard_edge_masks[prediction]
            from utils import to_networkx
            from baselines_utils import hard_edge_masks2coalition, fidelity_normalize_and_harmonic_mean

            f = related_preds["origin"] - related_preds["maskout"]
            inv_f = related_preds["origin"] - related_preds["masked"]
            sp = related_preds["sparsity"]
            n_f, n_inv_f, h_f = fidelity_normalize_and_harmonic_mean(f, inv_f, sp)
            title_sentence = f"fide: {f:.3f}, inv-fide: {inv_f:.3f}, h-fide: {h_f:.3f}"
            title_sentence = f'fide: {f:.4f}, spar: {sp:.4f}'

            if hasattr(dataset, "supplement"):
                words = dataset.supplement["sentence_tokens"][str(idx)]
            else:
                words = None

            explained_example_plot_path = os.path.join(
                explanation_saving_dir, f"example_{idx}.png"
            )

            fide, score = eval_metric(related_preds["origin"], related_preds["maskout"], related_preds["sparsity"])
            fides_abs.append(fide)
            fides_ori.append(f)
            scores.append(score)
            spars.append(sp)

    else:
        data = dataset.data
        data.edge_index = add_remaining_self_loops(data.edge_index)[0]
        data.to(device)

        node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
        test_indices = node_indices
        predictions = model(data).softmax(dim=-1).argmax(dim=-1)
        for node_idx in node_indices:
            index += 1
            with torch.no_grad():
                edge_masks, hard_edge_masks, related_preds = \
                    pgexplainer_edges(data.x, data.edge_index,
                                      node_idx=node_idx,
                                      num_classes=dataset.num_classes,
                                      sparsity=config.explainers.sparsity,
                                      pred_label=predictions[node_idx].item())
                edge_masks = [mask.detach() for mask in edge_masks]

                # if isinstance(dataset, SynGraphDataset):
                #     motif_edge_mask = dataset.gen_motif_edge_mask(data, node_idx=node_idx)
                #     motif_edge_mask = motif_edge_mask[pgexplainer_edges.select_edge_mask].float()
                #     if not (motif_edge_mask == 1).all():
                #         roc_aucs = [roc_auc_score(motif_edge_mask.cpu().numpy(), edge_mask.detach().cpu().numpy())
                #                     for edge_mask in edge_masks]
                #         for target_label, related_pred in enumerate(related_preds):
                #             related_preds[target_label]['accuracy'] = roc_aucs[target_label]

                #         # for the stability metric
                #         all_roc_aucs = []
                #         perturb_input_list = perturb_input(data,
                #                                            pgexplainer_edges.select_edge_mask,
                #                                            pgexplainer_edges.subset)
                #         new_node_idx = pgexplainer_edges.new_node_idx

                #         # take the original PGExplainer forward to calculate the stability metric
                #         for perturb_data in perturb_input_list:
                #             new_prediction = model(perturb_data)[new_node_idx].argmax(dim=-1)
                #             perturb_edge_masks, perturb_hard_edge_masks, perturb_related_preds = \
                #                 pgexplainer_edges(perturb_data.x,
                #                                   perturb_data.edge_index,
                #                                   node_idx=new_node_idx,
                #                                   sparsity=config.explainers.sparsity,
                #                                   num_classes=dataset.num_classes)
                #             perturb_motif_edge_mask = torch.cat(
                #                 [motif_edge_mask, torch.zeros(perturb_data.edge_index.shape[1] - motif_edge_mask.shape[0])],
                #                 dim=0)
                #             if not (perturb_motif_edge_mask == 1).all():
                #                 all_roc_aucs.append(
                #                     roc_auc_score(perturb_motif_edge_mask[pgexplainer_edges.select_edge_mask].cpu().numpy(),
                #                                   perturb_edge_masks[0].cpu().numpy()))

                #         for target_label, related_pred in enumerate(related_preds):
                #             related_preds[target_label]['stability'] = \
                #                 related_preds[target_label]['accuracy'] - torch.tensor(all_roc_aucs).mean().item()

            x_collector.collect_data(edge_masks, related_preds, label=predictions[node_idx].item())
            fide, score = eval_metric(related_preds[predictions[node_idx].item()]["origin"], related_preds[predictions[node_idx].item()]["maskout"],
                                      related_preds[predictions[node_idx].item()]["sparsity"])
            fides_abs.append(fide)
            fides_ori.append(related_preds[predictions[node_idx].item()]["origin"] - related_preds[predictions[node_idx].item()]["maskout"])
            scores.append(score)
            spars.append(related_preds[predictions[node_idx].item()]["sparsity"])

    end_time = time.time()

    experiment_data = {
        'fidelity': np.mean(fides_ori),
        'sparsity': np.mean(spars),
        'STD of sparsity': np.std(spars),
        'fidelity_abs': np.mean(fides_abs),
        'STD of fidelity_abs': np.std(fides_abs),
        'Training time': train_time,
        'Time in seconds': end_time - start_time,
        'Average Time': (end_time - start_time)/len(test_indices)
    }

    recorder.append(experiment_settings=['pgexplainer', f"{config.explainers.sparsity}"],
                    experiment_data=experiment_data)

    recorder.save()


if __name__ == '__main__':
    import sys
    sys.argv.append('explainers=pgexplainer')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explainer_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")
    sys.argv.append(f"record_filename={os.path.join(os.path.dirname(__file__), 'result_jsons')}")
    pipeline()
