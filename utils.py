import json
import os
from abc import ABC
from collections import Counter
import pytz
import logging
import copy
from datetime import datetime
import numpy as np
import networkx as nx
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
from typing import Union, List
from textwrap import wrap
import random
import numpy as np
import torch
from methods.initialization_mcts import reward_func, MCTSNode
from methods.exploration_mcts import exploration_MCTS
from shapley import GnnNets_GC2value_func, GnnNets_NC2value_func
from torch_geometric.utils import subgraph, to_dense_adj, k_hop_subgraph
from torch_geometric.data import Data, Batch, Dataset, DataLoader

# For associated game
from itertools import combinations
from scipy.sparse.csgraph import connected_components as cc



def find_closest_node_result(results, max_nodes=5, **kwargs):
    """ return the highest reward graph node constraining to the subgraph size """
    gamma = kwargs.get('config').explainers.param.single_explanation_size
    g = results[0].ori_graph
    b = results[0].P
    data = kwargs.get('data')
    # result_node = []
    _results = [tmp for tmp in results if len(tmp.coalition) <= gamma]
    results = _results if len(_results) > 0 else results[0]
    results = sorted(results, key=lambda x: x.P, reverse=True)
    K = kwargs.get('config').explainers.param.candidate_size
    max_i = min(K, len(results))
    
    if kwargs.get('config').models.param.graph_classification:
        value_func = GnnNets_GC2value_func(kwargs.get('gnnNets'), target_class=data.y)
        score_func = reward_func(kwargs.get('config').explainers.param, value_func)
    else:
        value_func = GnnNets_NC2value_func(kwargs.get('gnnNets'), target_class=kwargs.get('target_class'), node_idx=kwargs.get('node_idx'))
        score_func = reward_func(kwargs.get('config').explainers.param, value_func)
    
    method = kwargs.get('config').explainers.param.explanation_exploration_method
    if method.lower() == 'permutation':
        def dfs(coalition: list, current, num_of_g=0):
            num_of_g = num_of_g + 1
            co = coalition # list(set(coalition))
            if len(co) > max_nodes:
                return
            if num_of_g >= 2:
                n = MCTSNode(co, data=data, ori_graph=g)
                n.P = score_func(n.coalition, data)
                results.append(n)
                # return
            for i in range(current, max_i):
                tmp = coalition.copy()
                tmp.extend(results[i].coalition)
                tmp = list(set(tmp))
                if Counter(co) == Counter(tmp):
                    continue
                dfs(tmp, i+1, num_of_g)
            pass

        dfs([], 0)
        results = sorted(results, key=lambda x: x.P, reverse=True)
    elif method.lower() == 'mcts':
        mcts_state_map = exploration_MCTS(data.x, data.edge_index, results[:max_i]
                                          score_func=score_func,
                                          n_rollout=10,
                                          explanation_size=max_nodes, 
                                          c_puct=config.explainers.param.c_puct)
        results = mcts_state_map.mcts(verbose=True)
        pass
    
    result_node = results[0]
    return result_node


def eval_metric(original_score, gnn_score, sparsity):
    fidelity_score = (original_score - gnn_score) / original_score
    if isinstance(fidelity_score, torch.Tensor):
        fidelity_score = fidelity_score.item()
    score = fidelity_score * sparsity
    return fidelity_score, score


def fidelity_normalize_and_harmonic_mean(fidelity, inv_fidelity, sparsity):
    """
    The idea is similar to the F1 score, two measures are summarized to one through harmonic mean.

    Step1: normalize both scores with sparsity
        norm_fidelity = fidelity * sparsity
        norm_inv_fidelity = inv_fidelity * (1 - sparsity)
    Step2: rescale both normalized scores from [-1, 1] to [0, 1]
        rescaled_fidelity = (1 + norm_fidelity) / 2
        rescaled_inv_fidelity = (1 - norm_inv_fidelity) / 2
    Step3: take the harmonic mean of two rescaled scores
        2 / (1/rescaled_fidelity + 1/rescaled_inv_fidelity)

    Simplifying these three steps gives the formula
    """
    norm_fidelity = fidelity * sparsity
    norm_inv_fidelity = inv_fidelity * (1 - sparsity)
    harmonic_fidelity = (
        (1 + norm_fidelity)
        * (1 - norm_inv_fidelity)
        / (2 + norm_fidelity - norm_inv_fidelity)
    )
    return norm_fidelity, norm_inv_fidelity, harmonic_fidelity


def fix_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def perturb_input(data, hard_edge_mask, subset):
    """add 2 additional empty node into the motif graph"""
    num_add_node = 2
    num_perturb_graph = 10
    subgraph_x = data.x[subset]
    subgraph_edge_index = data.edge_index[:, hard_edge_mask]
    row, col = data.edge_index

    mappings = row.new_full((data.num_nodes,), -1)
    mappings[subset] = torch.arange(subset.size(0), device=row.device)
    subgraph_edge_index = mappings[subgraph_edge_index]

    subgraph_y = data.y[subset]

    num_node_subgraph = subgraph_x.shape[0]

    # add two nodes to the subgraph, the node features are all 0.1
    subgraph_x = torch.cat([subgraph_x,
                            torch.ones(2, subgraph_x.shape[1]).to(subgraph_x.device)],
                           dim=0)
    subgraph_y = torch.cat([subgraph_y,
                            torch.zeros(num_add_node).type(torch.long).to(subgraph_y.device)], dim=0)

    perturb_input_list = []
    for _ in range(num_perturb_graph):
        to_node = torch.randint(0, num_node_subgraph, (num_add_node,))
        frm_node = torch.arange(num_node_subgraph, num_node_subgraph + num_add_node, 1)
        add_edges = torch.cat([torch.stack([to_node, frm_node], dim=0),
                               torch.stack([frm_node, to_node], dim=0),
                               torch.stack([frm_node, frm_node], dim=0)], dim=1)
        perturb_subgraph_edge_index = torch.cat([subgraph_edge_index,
                                                 add_edges.to(subgraph_edge_index.device)], dim=1)
        perturb_input_list.append(Data(x=subgraph_x, edge_index=perturb_subgraph_edge_index, y=subgraph_y))

    return perturb_input_list


class Recorder(ABC):
    def __init__(self, recorder_filename):
        # init the recorder
        self.recorder_filename = recorder_filename
        if os.path.isfile(recorder_filename):
            with open(recorder_filename, 'r') as f:
                self.recorder = json.load(f)
        else:
            self.recorder = {}
            check_dir(os.path.dirname(recorder_filename))

    @classmethod
    def load_and_change_dict(cls, ori_dict, experiment_settings, experiment_data):
        key = experiment_settings[0]
        if key not in ori_dict.keys():
            ori_dict[key] = {}
        if len(experiment_settings) == 1:
            ori_dict[key] = experiment_data
        else:
            ori_dict[key] = cls.load_and_change_dict(ori_dict[key],
                                                     experiment_settings[1:],
                                                     experiment_data)
        return ori_dict

    def append(self, experiment_settings, experiment_data):
        ex_dict = self.recorder

        self.recorder = self.load_and_change_dict(ori_dict=ex_dict,
                                                  experiment_settings=experiment_settings,
                                                  experiment_data=experiment_data)

    def save(self):
        with open(self.recorder_filename, 'w') as f:
            json.dump(self.recorder, f, indent=2)

def to_networkx(
    data,
    node_index=None,
    node_attrs=None,
    edge_attrs=None,
    to_undirected=False,
    remove_self_loops=False,
):
    r"""
    Extend the PyG to_networkx with extra node_index argument, so subgraphs can be plotted with correct ids

    Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)


        node_index (iterable): Pass in it when there are some nodes missing.
                 max(node_index) == max(data.edge_index)
                 len(node_index) == data.num_nodes
    """
    import networkx as nx

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    if node_index is not None:
        """
        There are some nodes missing. The max(data.edge_index) > data.x.shape[0]
        """
        G.add_nodes_from(node_index)
    else:
        G.add_nodes_from(range(data.num_nodes))

    node_attrs, edge_attrs = node_attrs or [], edge_attrs or []

    values = {}
    for key, item in data(*(node_attrs + edge_attrs)):
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)

        for key in edge_attrs:
            G[u][v][key] = values[key][i]

    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)


def timetz(*args):
    tz = pytz.timezone("US/Pacific")
    return datetime.now(tz).timetuple()


def get_logger(log_path, log_file, console_log=False, log_level=logging.INFO):
    check_dir(log_path)

    tz = pytz.timezone("US/Pacific")
    logger = logging.getLogger(__name__)
    logger.propagate = False  # avoid duplicate logging
    logger.setLevel(log_level)

    # Clean logger first to avoid duplicated handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    file_handler = logging.FileHandler(os.path.join(log_path, log_file))
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%b%d %H-%M-%S")
    formatter.converter = timetz
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def get_graph_build_func(build_method):
    if build_method.lower() == "zero_filling":
        return graph_build_zero_filling
    elif build_method.lower() == "split":
        return graph_build_split
    elif build_method.lower() == "remove":
        return graph_build_remove
    else:
        raise NotImplementedError


"""
Graph building/Perturbation
`graph_build_zero_filling` and `graph_build_split` are adapted from the DIG library
"""


def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through masking the unselected nodes with zero features"""
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index


def graph_build_split(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through spliting the selected nodes from the original graph"""
    ret_X = X
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return ret_X, ret_edge_index


def graph_build_remove(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through removing the unselected nodes from the original graph"""
    ret_X = X[node_mask == 1]
    ret_edge_index, _ = subgraph(node_mask.bool(), edge_index, relabel_nodes=True)
    return ret_X, ret_edge_index


"""
Associated game of the HN value
Implementated using sparse tensor
"""


def get_ordered_coalitions(n):
    coalitions = sum(
        [[set(c) for c in combinations(range(n), k)] for k in range(1, n + 1)], []
    )
    return coalitions


def get_associated_game_matrix_M(coalitions, n, tau):
    indices = []
    values = []
    for i, s in enumerate(coalitions):
        for j, t in enumerate(coalitions):
            if i == j:
                indices += [[i, j]]
                values += [1 - (n - len(s)) * tau]
            elif len(s) + 1 == len(t) and s.issubset(t):
                indices += [[i, j]]
                values += [tau]
            elif len(t) == 1 and not t.issubset(s):
                indices += [[i, j]]
                values += [-tau]

    indices = torch.Tensor(indices).t()
    size = (2**n - 1, 2**n - 1)
    M = torch.sparse_coo_tensor(indices, values, size)
    return M


def get_associated_game_matrix_P(coalitions, n, adj):
    indices = []
    for i, s in enumerate(coalitions):
        idx_s = torch.LongTensor(list(s))
        num_cc, labels = cc(adj[idx_s, :][:, idx_s])
        cc_s = []
        for k in range(num_cc):
            cc_idx_s = (labels == k).nonzero()[0]
            cc_s += [set((idx_s[cc_idx_s]).tolist())]
        for j, t in enumerate(coalitions):
            if t in cc_s:
                indices += [[i, j]]

    indices = torch.Tensor(indices).t()
    values = [1.0] * indices.shape[-1]
    size = (2**n - 1, 2**n - 1)

    P = torch.sparse_coo_tensor(indices, values, size)
    return P


def get_limit_game_matrix(H, exp_power=7, tol=1e-3, is_sparse=True):
    """
    Speed up the power computation by
    1. Use sparse matrices
    2. Put all tensors on cuda
    3. Compute powers exponentially rather than linearly
        i.e. H -> H^2 -> H^4 -> H^8 -> H^16 -> ...
    """
    i = 0
    diff_norm = tol + 1
    while i < exp_power and diff_norm > tol:
        if is_sparse:
            H_tilde = torch.sparse.mm(H, H)
        else:
            H_tilde = torch.mm(H, H)
        diff_norm = (H_tilde - H).norm()
        H = H_tilde
        i += 1
    return H_tilde


"""
khop or random sampling to generate subgraphs
"""


def sample_subgraph(
    data, max_sample_size, sample_method, target_node=None, k=0, adj=None
):
    if sample_method == "khop":
        # pick nodes within k-hops of target node. Hop by hop until reach max_sample_size
        if adj is None:
            adj = (
                to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
                .detach()
                .cpu()
            )

        adj_self_loop = adj + torch.eye(data.num_nodes)
        k_hop_adj = adj_self_loop
        sampled_nodes = set()
        m = max_sample_size
        l = 0
        while k > 0 and l < m:
            k_hop_nodes = k_hop_adj[target_node].nonzero().view(-1).tolist()
            next_hop_nodes = list(set(k_hop_nodes) - sampled_nodes)
            sampled_nodes.update(next_hop_nodes[: m - l])
            l = len(sampled_nodes)
            k -= 1
            k_hop_adj = torch.mm(k_hop_adj, adj_self_loop)
        sampled_nodes = torch.tensor(list(sampled_nodes))

    elif sample_method == "random":  # randomly pick #max_sample_size nodes
        sampled_nodes = torch.randperm(data.num_nodes)[:max_sample_size]
    else:
        ValueError("Unknown sample method")

    sampled_x = data.x[sampled_nodes]
    sampled_edge_index, _ = subgraph(
        sampled_nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
    )
    sampled_data = Data(x=sampled_x, edge_index=sampled_edge_index)
    sampled_adj = adj[sampled_nodes, :][:, sampled_nodes]

    return sampled_nodes, sampled_data, sampled_adj


"""
Payoff computation
"""


def get_char_func(model, target_class, payoff_type="norm_prob", payoff_avg=None):
    def char_func(data, for_node=None):
        with torch.no_grad():
            logits = model(data=data)
            # print(logits.softmax(dim=-1).shape)
            if for_node is not None:
                idx = [(x % for_node == 0) for x in range(logits.shape[0])]
                logits = logits[idx, :]
                # print(logits.shape[0])
            if payoff_type == "raw":
                payoff = logits[:, target_class]
            elif payoff_type == "prob":
                payoff = logits.softmax(dim=-1)[:, target_class]
            elif payoff_type == "norm_prob":
                prob = logits.softmax(dim=-1)[:, target_class]
                payoff = prob - payoff_avg[target_class]
            elif payoff_type == "log_prob":
                payoff = logits.log_softmax(dim=-1)[:, target_class]
            else:
                raise ValueError("unknown payoff type")
        return payoff

    return char_func


class MaskedDataset(Dataset):
    def __init__(self, data, mask, subgraph_building_func):
        super().__init__()

        self.num_nodes = data.num_nodes
        self.x = data.x
        self.edge_index = data.edge_index
        self.device = data.x.device
        self.y = data.y

        if not torch.is_tensor(mask):
            mask = torch.tensor(mask)

        self.mask = mask.type(torch.float32).to(self.device)
        self.subgraph_building_func = subgraph_building_func

    def __len__(self):
        return self.mask.shape[0]

    def __getitem__(self, idx):
        masked_x, masked_edge_index = self.subgraph_building_func(
            self.x, self.edge_index, self.mask[idx]
        )
        masked_data = Data(x=masked_x, edge_index=masked_edge_index)
        return masked_data


def get_coalition_payoffs(data, coalitions, char_func, subgraph_building_func, for_node=None):
    n = data.num_nodes
    masks = []
    for coalition in coalitions:
        mask = torch.zeros(n)
        mask[list(coalition)] = 1.0
        masks += [mask]

    coalition_mask = torch.stack(masks, axis=0)
    # print(coalition_mask.shape)  [1023, 10]
    masked_dataset = MaskedDataset(data, coalition_mask, subgraph_building_func)
    masked_dataloader = DataLoader(
        masked_dataset, batch_size=256, shuffle=False, num_workers=0
    )
    
    pos = None
    if for_node is not None:
        pos = 0
        for x in coalition:
            if x == for_node:
                break
            pos += 1
    

    masked_payoff_list = []
    for masked_data in masked_dataloader:
        masked_payoff_list.append(char_func(masked_data, pos))

    # print(masked_payoff_list[0].shape)
    masked_payoffs = torch.cat(masked_payoff_list, dim=0)
    # print(masked_payoffs.shape)
    return masked_payoffs


"""
Superadditive extension
"""


class TrieNode:
    def __init__(self, player, payoff=0, children=[]):
        self.player = player
        self.payoff = payoff
        self.children = children


class CoalitionTrie:
    def __init__(self, coalitions, n, v):
        self.n = n
        self.root = self.get_node(None, 0)
        for i, c in enumerate(coalitions):
            self.insert(c, v[i])

    def get_node(self, player, payoff):
        return TrieNode(player, payoff, [None] * self.n)

    def insert(self, coalition, payoff):
        curr = self.root
        for player in coalition:
            if curr.children[player] is None:
                curr.children[player] = self.get_node(player, 0)
            curr = curr.children[player]
        curr.payoff = payoff

    def search(self, coalition):
        curr = self.root
        for player in coalition:
            if curr.children[player] is None:
                return None
            curr = curr.children[player]
        return curr.payoff

    def visualize(self):
        self._visualize(self.root, 0)

    def _visualize(self, node, level):
        if node:
            print(f"{'-'*level}{node.player}:{node.payoff}")
            for child in node.children:
                self._visualize(child, level + 1)


def superadditive_extension(n, v):
    """
    n (int): number of players
    v (list of floats): dim = 2 ** n - 1, each entry is a payoff
    """
    coalition_sets = get_ordered_coalitions(n)
    coalition_lists = [sorted(list(c)) for c in coalition_sets]
    coalition_trie = CoalitionTrie(coalition_lists, n, v)
    v_ext = v[:]
    for i, coalition in enumerate(coalition_lists):
        partition_payoff = []
        for part in set_partitions(coalition, 2):
            subpart_payoff = []
            for subpart in part:
                subpart_payoff += [coalition_trie.search(subpart)]
            partition_payoff += [sum(subpart_payoff)]
        v_ext[i] = max(partition_payoff + [v[i]])
        coalition_trie.insert(coalition, v_ext[i])
    return v_ext


"""
Evaluation functions
"""


def scores2coalition(scores, sparsity):
    scores_tensor = torch.tensor(scores)
    top_idx = scores_tensor.argsort(descending=True).tolist()
    cutoff = int(len(scores) * (1 - sparsity))
    cutoff = min(cutoff, (scores_tensor > 0).sum().item())
    coalition = top_idx[:cutoff]
    return coalition


def evaluate_coalition(explainer, data, coalition, node_idx=None, config=None):
    device = explainer.device
    data = data.to(device)
    pred_prob = explainer.model(data).softmax(dim=-1)
    if node_idx is None:
        target_class = pred_prob.argmax(-1).item()
        original_prob = pred_prob[:, target_class].item()
    else:
        _, predictions = torch.max(pred_prob, -1)
        target_class = predictions[node_idx]
        original_prob = pred_prob[node_idx, target_class].item()
        

    num_nodes = data.num_nodes
    if len(coalition) == num_nodes:
        # Edge case: pick the graph itself as the explanation, for synthetic data
        masked_prob = original_prob
        maskout_prob = 0
    elif len(coalition) == 0:
        # Edge case: pick the empty set as the explanation, for synthetic data
        masked_prob = 0
        maskout_prob = original_prob
    else:
        if node_idx is not None:
            # graph = to_networkx(data)
            
            ego_nodes, ego_edge, mapping, _ = k_hop_subgraph(node_idx, 
                                                   config.datasets.local_radius-1, 
                                                   data.edge_index, 
                                                   relabel_nodes=True)
            
            node_idx = mapping.item()
            ego_coali = [torch.argmax((x == ego_nodes).int()).item() for x in coalition]
            # ego_coali = [x for x in ego_coali if x != mapping]
            num_nodes = len(ego_nodes)
            mask = torch.zeros(num_nodes).type(torch.float32).to(device)
            mask[ego_coali] = 1.0
            # print(ego_coali)
            # print(ego_nodes)
            # raise 
            masked_x, masked_edge_index = explainer.subgraph_building_func(
                data.x[ego_nodes,:], ego_edge, mask
            )
            masked_data = Data(x=masked_x, edge_index=masked_edge_index).to(device) 
        else:
            mask = torch.zeros(num_nodes).type(torch.float32).to(device)
            mask[coalition] = 1.0
            masked_x, masked_edge_index = explainer.subgraph_building_func(
                data.x, data.edge_index, mask
            )
            masked_data = Data(x=masked_x, edge_index=masked_edge_index).to(device) 
        
        if node_idx is None:
            masked_prob = (
                explainer.model(masked_data).softmax(dim=-1)[:, target_class].item()
            )
        else:
            masked_prob = (
                explainer.model(masked_data).softmax(dim=-1)[node_idx, target_class].item()
            )
    
        if node_idx is None:
            maskout_x, maskout_edge_index = explainer.subgraph_building_func(
                data.x, data.edge_index, 1 - mask
            )
            maskout_data = Data(x=maskout_x, edge_index=maskout_edge_index).to(device)
            maskout_prob = (
                explainer.model(maskout_data).softmax(dim=-1)[:, target_class].item()
            )
        else:
            maskout_x, maskout_edge_index = explainer.subgraph_building_func(
                data.x[ego_nodes,:], ego_edge, 1 - mask
            )
            maskout_data = Data(x=maskout_x, edge_index=maskout_edge_index).to(device)
            maskout_prob = (
                explainer.model(maskout_data).softmax(dim=-1)[node_idx, target_class].item()
            )
        

    fidelity = original_prob - maskout_prob
    abs_fidelity = fidelity / original_prob
    inv_fidelity = original_prob - masked_prob
    sparsity = 1 - len(coalition) / num_nodes
    return fidelity, abs_fidelity, inv_fidelity, sparsity

def evaluate_scores_list(explainer, data_list, scores_list, sparsity, logger=None):
    """
    Evaluate the node importance scoring methods, where each node has an associated score,
    i.e. GStarX and GraphSVX.

    Args:
    data_list (list of PyG data)
    scores_list (list of lists): each entry is a list with scores of nodes in a graph

    """

    assert len(data_list) == len(scores_list)

    f_list = []
    abs_f_list = []
    inv_f_list = []
    n_f_list = []
    n_inv_f_list = []
    sp_list = []
    h_f_list = []
    for i, data in enumerate(data_list):
        node_scores = scores_list[i]
        coalition = scores2coalition(node_scores, sparsity)
        f, abs_f, inv_f, sp = evaluate_coalition(explainer, data, coalition)
        n_f, n_inv_f, h_f = fidelity_normalize_and_harmonic_mean(f, inv_f, sp)

        f_list += [f]
        abs_f_list += [abs_f]
        inv_f_list += [inv_f]
        n_f_list += [n_f]
        n_inv_f_list += [n_inv_f]
        sp_list += [sp]
        h_f_list += [h_f]

    f_mean = np.mean(f_list).item()
    abs_f_mean = np.mean(abs_f_list).item()
    inv_f_mean = np.mean(inv_f_list).item()
    n_f_mean = np.mean(n_f_list).item()
    n_inv_f_mean = np.mean(n_inv_f_list).item()
    sp_mean = np.mean(sp_list).item()
    h_f_mean = np.mean(h_f_list).item()

    if logger is not None:
        logger.info(
            f"Fidelity Mean: {f_mean:.5f}\n"
            f"Abs Fidelity Mean: {abs_f_mean:.5f}\n"
            f"Inv-Fidelity Mean: {inv_f_mean:.5f}\n"
            f"Norm-Fidelity Mean: {n_f_mean:.5f}\n"
            f"Norm-Inv-Fidelity Mean: {n_inv_f_mean:.5f}\n"
            f"Sparsity Mean: {sp_mean:.5f}\n"
            f"Harmonic-Fidelity Mean: {h_f_mean:.5f}\n"
        )

    return sp_mean, f_mean, inv_f_mean, n_f_mean, n_inv_f_mean, h_f_mean


"""
Visualization
"""


def coalition2subgraph(coalition, data, relabel_nodes=True):
    sub_data = copy.deepcopy(data)
    node_mask = torch.zeros(data.num_nodes)
    node_mask[coalition] = 1

    sub_data.x = data.x[node_mask == 1]
    sub_data.edge_index, _ = subgraph(
        node_mask.bool(), data.edge_index, relabel_nodes=relabel_nodes
    )
    return sub_data

"""
Adapted from SubgraphX DIG implementation
https://github.com/divelab/DIG/blob/dig/dig/xgraph/method/subgraphx.py

Slightly modified the molecule drawing args
"""


class PlotUtils(object):
    def __init__(self, dataset_name, is_show=True):
        self.dataset_name = dataset_name
        self.is_show = is_show

    def plot(self, graph, nodelist, figname, title_sentence=None, **kwargs):
        """plot function for different dataset"""
        if self.dataset_name.lower() in ["ba_2motifs"]:
            self.plot_ba2motifs(
                graph, nodelist, title_sentence=title_sentence, figname=figname
            )
        elif self.dataset_name.lower() in ["mutag", "bbbp", "bace"]:
            x = kwargs.get("x")
            self.plot_molecule(
                graph, nodelist, x, title_sentence=title_sentence, figname=figname
            )
        elif self.dataset_name.lower() in ["graph_sst2", "graph_sst5", "twitter"]:
            words = kwargs.get("words")
            self.plot_sentence(
                graph,
                nodelist,
                words=words,
                title_sentence=title_sentence,
                figname=figname,
            )
        elif self.dataset_name.lower() in ["ba_shapes"]:
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            self.plot_bashapes(
                graph, nodelist, y, node_idx,
                title_sentence=title_sentence, figname=figname
            )
        else:
            raise NotImplementedError

    def plot_subgraph(
        self,
        graph,
        nodelist,
        colors: Union[None, str, List[str]] = "#FFA500",
        labels=None,
        edge_color="gray",
        edgelist=None,
        subgraph_edge_color="black",
        title_sentence=None,
        figname=None,
    ):

        if edgelist is None:
            edgelist = [
                (n_frm, n_to)
                for (n_frm, n_to) in graph.edges()
                if n_frm in nodelist and n_to in nodelist
            ]

        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(
            graph,
            pos_nodelist,
            nodelist=nodelist,
            node_color="black",
            node_shape="o",
            node_size=400,
        )
        nx.draw_networkx_nodes(
            graph, pos, nodelist=list(graph.nodes()), node_color=colors, node_size=200
        )
        nx.draw_networkx_edges(graph, pos, width=2, edge_color=edge_color, arrows=False)
        nx.draw_networkx_edges(
            graph,
            pos=pos_nodelist,
            edgelist=edgelist,
            width=6,
            edge_color="black",
            arrows=False,
        )

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis("off")
        if title_sentence is not None:
            plt.title(
                "\n".join(wrap(title_sentence, width=60)), fontdict={"fontsize": 15}
            )
        if figname is not None:
            plt.savefig(figname, format=figname[-3:])

        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_sentence(
        self, graph, nodelist, words, edgelist=None, title_sentence=None, figname=None
    ):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(
                graph,
                pos_coalition,
                nodelist=nodelist,
                node_color="yellow",
                node_shape="o",
                node_size=500,
            )
            if edgelist is None:
                edgelist = [
                    (n_frm, n_to)
                    for (n_frm, n_to) in graph.edges()
                    if n_frm in nodelist and n_to in nodelist
                ]
                nx.draw_networkx_edges(
                    graph,
                    pos=pos_coalition,
                    edgelist=edgelist,
                    width=5,
                    edge_color="yellow",
                )

        nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color="grey")
        nx.draw_networkx_labels(graph, pos, words_dict)

        plt.axis("off")
        plt.title("\n".join(wrap(" ".join(words), width=50)))
        if title_sentence is not None:
            string = "\n".join(wrap(" ".join(words), width=50)) + "\n"
            string += "\n".join(wrap(title_sentence, width=60))
            plt.title(string)
        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_ba2motifs(
        self, graph, nodelist, edgelist=None, title_sentence=None, figname=None
    ):
        return self.plot_subgraph(
            graph,
            nodelist,
            edgelist=edgelist,
            title_sentence=title_sentence,
            figname=figname,
        )
    
    def plot_bashapes(self, graph, nodelist, y, node_idx, edgelist=None, figname=None, title_sentence=None):
        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#4970C6', '#FE0000', 'green']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph(graph, nodelist, colors, edgelist=edgelist, figname=figname,
                           title_sentence=title_sentence)

    def plot_molecule(
        self, graph, nodelist, x, edgelist=None, title_sentence=None, figname=None
    ):
        # collect the text information and node color
        if self.dataset_name == "mutag":
            node_dict = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
            node_idxs = {
                k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])
            }
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = [
                "#E49D1C",
                "#4970C6",
                "#FF5357",
                "#29A329",
                "brown",
                "darkslategray",
                "#F0EA00",
            ]
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        elif self.dataset_name in ["bbbp", "bace"]:
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_idxs = element_idxs
            node_labels = {
                k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                for k, v in element_idxs.items()
            }
            node_color = [
                "#29A329",
                "lime",
                "#F0EA00",
                "maroon",
                "brown",
                "#E49D1C",
                "#4970C6",
                "#FF5357",
            ]
            colors = [
                node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()
            ]
        else:
            raise NotImplementedError

        self.plot_subgraph(
            graph,
            nodelist,
            colors=colors,
            labels=node_labels,
            edgelist=edgelist,
            edge_color="gray",
            subgraph_edge_color="black",
            title_sentence=title_sentence,
            figname=figname,
        )
