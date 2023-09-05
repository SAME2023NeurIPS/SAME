
import torch
from torch_geometric.utils import subgraph, to_dense_adj
from utils import *


class GStarX(object):
    def __init__(
        self,
        model,
        device,
        max_sample_size=10,
        tau=0.01,
        payoff_type="norm_prob",
        payoff_avg=None,
        subgraph_building_method="remove",
    ):

        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.max_sample_size = max_sample_size
        self.coalitions = get_ordered_coalitions(max_sample_size)
        self.tau = tau
        self.M = get_associated_game_matrix_M(self.coalitions, max_sample_size, tau)
        self.M = self.M.to(device)

        self.payoff_type = payoff_type
        self.payoff_avg = payoff_avg
        self.subgraph_building_func = get_graph_build_func(subgraph_building_method)

    def explain(
        self, data, superadditive_ext=True, sample_method="khop", num_samples=10, k=3, node_idx=None, target_node=None
    ):
        """
        Args:
        sample_method (str): `khop` or `random`. see `sample_subgraph` in utils for details
        num_samples (int): set to -1 then data.num_nodes will be used as num_samples
        """
        data = data.to(self.device)
        adj = (
            to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
            .detach()
            .cpu()
        )
        for_node = None
        if node_idx is None:
            target_class = self.model(data).argmax(-1).item()
        else:
            target_class = node_idx
            for_node = target_node.item()

            
        char_func = get_char_func(
            self.model, target_class, self.payoff_type, self.payoff_avg
        )
        if data.num_nodes < self.max_sample_size:
            scores = self.compute_scores(data, adj, char_func, superadditive_ext, for_node)
        else:
            scores = torch.zeros(data.num_nodes)
            counts = torch.zeros(data.num_nodes)
            if sample_method == "khop" or num_samples == -1:
                num_samples = data.num_nodes

            i = 0
            while not counts.all() or i < num_samples:
                sampled_nodes, sampled_data, sampled_adj = sample_subgraph(
                    data, self.max_sample_size, sample_method, i, k, adj
                )
                if for_node is not None and for_node not in sampled_nodes:
                    continue
                sampled_scores = self.compute_scores(
                    sampled_data, sampled_adj, char_func, superadditive_ext, for_node
                )
                scores[sampled_nodes] += sampled_scores
                counts[sampled_nodes] += 1
                i += 1

            nonzero_mask = counts != 0
            scores[nonzero_mask] = scores[nonzero_mask] / counts[nonzero_mask]
        return scores.tolist()

    def compute_scores(self, data, adj, char_func, superadditive_ext=True, for_node=None):
        n = data.num_nodes
        if n == self.max_sample_size:  # use pre-computed results
            coalitions = self.coalitions
            M = self.M
        else:
            coalitions = get_ordered_coalitions(n)
            M = get_associated_game_matrix_M(coalitions, n, self.tau)
            M = M.to(self.device)

        v = get_coalition_payoffs(
            data, coalitions, char_func, self.subgraph_building_func, 
            for_node=for_node if for_node is not None else None
        )
        if superadditive_ext:
            v = v.tolist()
            v_ext = superadditive_extension(n, v)
            v = torch.tensor(v_ext).to(self.device)

        P = get_associated_game_matrix_P(coalitions, n, adj)
        P = P.to(self.device)
        H = torch.sparse.mm(P, torch.sparse.mm(M, P))
        H_tilde = get_limit_game_matrix(H, is_sparse=True)
        v_tilde = torch.sparse.mm(H_tilde, v.view(-1, 1)).view(-1)

        scores = v_tilde[:n].cpu()
        return scores
