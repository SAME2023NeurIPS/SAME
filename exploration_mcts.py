import math
import torch
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, remove_self_loops
from shapley import mc_shapley, l_shapley, mc_l_shapley, gnn_score, NC_mc_l_shapley
from initialization_mcts_GC import MCTSNode
from functools import partial
from collections import Counter

class exploration_MCTSNode():

    def __init__(self, coalition: list, data: Data, candidates: [list], 
                 ori_graph: nx.Graph, c_puct: float = 10.0,
                 W: float = 0, N: int = 0, P: float = 0):
        self.data = data
        self.candidates = candidates
        self.coalition = coalition  # permutation of any candidates
        self.ori_graph = ori_graph
        self.c_puct = c_puct
        self.children = []
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n): 
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)


class exploration_MCTS():
    def __init__(self, X: torch.Tensor, edge_index: torch.Tensor, candidates, 
                 n_rollout: int, explanation_size: int, c_puct: float, score_func):
        """ graph is a networkX graph """
        self.X = X
        self.candidates = candidates
        self.edge_index = edge_index
        self.data = Data(x=self.X, edge_index=self.edge_index)
        graph_data = Data(x=self.X, edge_index=remove_self_loops(self.edge_index)[0])
        self.graph = to_networkx(graph_data, to_undirected=True)
        self.data = Batch.from_data_list([self.data])
        self.num_nodes = self.graph.number_of_nodes()
        self.num_candidates = len(candidates)
        self.score_func = score_func
        self.n_rollout = n_rollout
        self.explanation_size = explanation_size
        self.c_puct = c_puct

        self.MCTSNodeClass = partial(exploration_MCTSNode, data=self.data, candidates=self.candidates,
                                     ori_graph=self.graph, c_puct=self.c_puct)
        self.root = self.MCTSNodeClass([])
        self.state_map = {str(self.root.coalition): self.root}

    def mcts_rollout(self, tree_node):
        unvisited_candidates = [i for i in range(self.num_candidates) if i not in tree_node.coalition]
        current_explanation = []
        for substructure in tree_node.coalition:
            current_explanation.extend(self.candidates[substructure].coalition)
        current_explanation = list(set(current_explanation))
        if len(current_explanation) >= self.explanation_size:
            return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            for each_node in unvisited_candidates:
                new_graph_coalition = [candidate for candidate in self.candidates 
                                       if candidate in tree_node.coalition or candidate == each_node]

                new_graph_coalition = []
                for candidate in range(len(self.candidates)): 
                    if candidate in tree_node.coalition or candidate == each_node: 
                        new_graph_coalition.append(candidate)
                new_graph_coalition = list(set(new_graph_coalition))

                new_graph_coalition = sorted(new_graph_coalition)
                Find_same = False
                for old_graph_node in self.state_map.values():
                    if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                        new_node = old_graph_node
                        Find_same = True

                if Find_same == False:
                    new_node = self.MCTSNodeClass(new_graph_coalition)
                    self.state_map[str(new_graph_coalition)] = new_node

                Find_same_child = False
                for cur_child in tree_node.children:
                    if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                        Find_same_child = True

                if Counter(new_node.coalition) == Counter(tree_node.coalition):
                    Find_same_child = True

                if Find_same_child == False:
                    tree_node.children.append(new_node)

            scores = compute_scores(self.score_func, tree_node.children)
            for child, score in zip(tree_node.children, scores):
                child.P = score

        sum_count = sum([c.N for c in tree_node.children])
        if len(tree_node.children) == 0:
            return tree_node.P
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v

    def mcts(self, verbose=True):
        if verbose:
            print(f"There are {self.num_candidates} candidates")
        for rollout_idx in range(self.n_rollout):
            self.mcts_rollout(self.root)
            if verbose:
                print(f"[Exploration] Rollout {rollout_idx}: {len(self.state_map)} accumulative permutations.")

        explanations = []
        for _, node in self.state_map.items():
            graph_coalition = []
            for substructure in node.coalition:
                graph_coalition.extend(self.candidates[substructure].coalition)
            graph_coalition = list(set(graph_coalition))
            n = MCTSNode(graph_coalition, data=node.data, ori_graph=node.ori_graph, P=node.P)
            explanations.append(n)
            
        explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
        return explanations


def compute_scores(score_func, children: list): # list[exploration_MCTSNode]
    results = []
    for child in children: 
        if child.P == 0:
            subgraph_coalition = []
            for substructure_id in child.coalition:             
                subgraph_coalition.extend(child.candidates[substructure_id].coalition)
            score = score_func(subgraph_coalition, child.data)
        else:
            score = child.P
        results.append(score)
    return results


def reward_func(reward_args, value_func, node_idx=-1):
    if reward_args.reward_method.lower() == 'gnn_score':
        return partial(gnn_score,
                       value_func=value_func,
                       subgraph_building_method=reward_args.subgraph_building_method)

    elif reward_args.reward_method.lower() == 'mc_shapley':
        return partial(mc_shapley,
                       value_func=value_func,
                       subgraph_building_method=reward_args.subgraph_building_method,
                       sample_num=reward_args.sample_num)

    elif reward_args.reward_method.lower() == 'l_shapley':
        return partial(l_shapley,
                       local_raduis=reward_args.local_raduis,
                       value_func=value_func,
                       subgraph_building_method=reward_args.subgraph_building_method)

    elif reward_args.reward_method.lower() == 'mc_l_shapley':
        return partial(mc_l_shapley,
                       local_raduis=reward_args.local_raduis,
                       value_func=value_func,
                       subgraph_building_method=reward_args.subgraph_building_method,
                       sample_num=reward_args.sample_num)
    elif reward_args.reward_method.lower() == 'nc_mc_l_shapley':
        return partial(NC_mc_l_shapley,
                       node_idx=node_idx,
                       local_raduis=reward_args.local_raduis,
                       value_func=value_func,
                       subgraph_building_method=reward_args.subgraph_building_method,
                       sample_num=reward_args.sample_num)
    else:
        raise NotImplementedError
