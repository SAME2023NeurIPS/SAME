import math
import torch
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, remove_self_loops
from shapley import mc_shapley, l_shapley, mc_l_shapley, gnn_score, NC_mc_l_shapley
from functools import partial
from collections import Counter

class MCTSNode():

    def __init__(self, coalition: list, data: Data,
                 ori_graph: nx.Graph, c_puct: float = 10.0,
                 W: float = 0, N: int = 0, P: float = 0):
        self.data = data
        self.coalition = coalition
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


class MCTS():
    def __init__(self, X: torch.Tensor, edge_index: torch.Tensor, n_rollout: int,
                 min_atoms: int, c_puct: float, expand_atoms: int, score_func, high2low):
        """ graph is a networkX graph """
        self.X = X
        self.edge_index = edge_index
        self.data = Data(x=self.X, edge_index=self.edge_index)
        graph_data = Data(x=self.X, edge_index=remove_self_loops(self.edge_index)[0])
        self.graph = to_networkx(graph_data, to_undirected=True)
        self.data = Batch.from_data_list([self.data])
        self.num_nodes = self.graph.number_of_nodes()
        self.score_func = score_func
        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low

        self.MCTSNodeClass = partial(MCTSNode, data=self.data, ori_graph=self.graph, c_puct=self.c_puct)
        ### Modified
        self.root = self.MCTSNodeClass([])
        self.root_coalition = [i for i in range(self.num_nodes)]
        ###
        self.state_map = {str(self.root.coalition): self.root}

    def mcts_rollout(self, tree_node):
        unvisited_graph_coalition = [i for i in range(self.num_nodes) if i not in tree_node.coalition]
        if len(tree_node.coalition) >= self.min_atoms or len(tree_node.coalition)/self.num_nodes >= 0.8:
            return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            if len(tree_node.coalition) == 0:
                node_degree_list = list(self.graph.subgraph(unvisited_graph_coalition).degree)
                node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=self.high2low)
                all_nodes = [x[0] for x in node_degree_list]

                if len(all_nodes) < self.expand_atoms:
                    expand_nodes = all_nodes
                else:
                    expand_nodes = all_nodes[:self.expand_atoms]
            else:
                expand_nodes = []
                for n in tree_node.coalition:
                    nbrs = self.graph.adj[n]
                    for nbr, _ in nbrs.items():
                        if nbr in unvisited_graph_coalition:
                            expand_nodes.append(nbr)

            for each_node in expand_nodes:
                subgraph_coalition = [node for node in range(self.num_nodes) if node in tree_node.coalition or node == each_node]

                subgraphs = [self.graph.subgraph(c)
                             for c in nx.connected_components(self.graph.subgraph(subgraph_coalition))]
                
                assert len(subgraphs) == 1
                main_sub = subgraphs[0]
                for sub in subgraphs:
                    if sub.number_of_nodes() > main_sub.number_of_nodes():
                        main_sub = sub

                new_graph_coalition = sorted(list(main_sub.nodes()))
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
            print(f"The nodes in graph is {self.graph.number_of_nodes()}")
        for rollout_idx in range(self.n_rollout):
            self.mcts_rollout(self.root)
            if verbose:
                print(f"[Initialization] Rollout {rollout_idx}, {len(self.state_map)} states that have been explored.")

        explanations = []
        for _, node in self.state_map.items():
            node.coalition = [k for k in self.root_coalition if k in node.coalition]
            explanations.append(node)
            pass
        explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
        return explanations


def compute_scores(score_func, children):
    results = []
    for child in children:
        if child.P == 0:
            score = score_func(child.coalition, child.data)
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
