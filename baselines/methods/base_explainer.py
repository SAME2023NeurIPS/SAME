import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
from math import sqrt
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
from rdkit import Chem
from matplotlib.axes import Axes
from typing import List, Tuple


EPS = 1e-15
"""
From DIG xgraph model utils
https://github.com/divelab/DIG/blob/dig/dig/xgraph/models/utils.py
"""


def subgraph(
    node_idx,
    num_hops,
    edge_index,
    relabel_nodes=False,
    num_nodes=None,
    flow="source_to_target",
):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`. when num_hops == -1,
            the whole graph will be returned.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ["source_to_target", "target_to_source"]
    if flow == "target_to_source":
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor(
            [node_idx], device=row.device, dtype=torch.int64
        ).flatten()
    else:
        node_idx = node_idx.to(row.device)

    inv = None

    if num_hops != -1:
        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[: node_idx.numel()]
    else:
        subsets = node_idx
        cur_subsets = node_idx
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


class ExplainerBase(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        epochs: int = 0,
        lr: float = 0,
        explain_graph: bool = False,
        molecule: bool = False,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.explain_graph = explain_graph
        self.molecule = molecule
        self.mp_layers = [
            module
            for module in self.model.modules()
            if isinstance(module, MessagePassing)
        ]
        self.num_layers = len(self.mp_layers)

        self.ori_pred = None
        self.ex_labels = None
        self.edge_mask = None
        self.hard_edge_mask = None

        self.num_edges = None
        self.num_nodes = None
        self.device = None
        self.table = Chem.GetPeriodicTable().GetElementSymbol

    def __set_masks__(self, x: Tensor, edge_index: Tensor, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(
            torch.randn(F, requires_grad=True, device=self.device) * 0.1
        )

        std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(
            torch.randn(E, requires_grad=True, device=self.device) * std
        )
        # self.edge_mask = torch.nn.Parameter(100 * torch.ones(E, requires_grad=True))

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def __num_hops__(self):
        if self.explain_graph:
            return -1
        else:
            return self.num_layers

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return "source_to_target"

    def __subgraph__(self, node_idx: int, x: Tensor, edge_index: Tensor, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = subgraph(
            node_idx,
            self.__num_hops__,
            edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
            flow=self.__flow__(),
        )

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, kwargs

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs):
        self.num_edges = edge_index.shape[1]
        self.num_nodes = x.shape[0]
        self.device = x.device

    def control_sparsity(self, mask: Tensor, sparsity=None, **kwargs):
        r"""

        :param mask: mask that need to transform
        :param sparsity: sparsity we need to control i.e. 0.7, 0.5
        :return: transformed mask where top 1 - sparsity values are set to inf.
        """
        if sparsity is None:
            sparsity = 0.7

        if not self.explain_graph:
            assert self.hard_edge_mask is not None
            mask_indices = torch.where(self.hard_edge_mask)[0]
            sub_mask = mask[self.hard_edge_mask]
            mask_len = sub_mask.shape[0]
            _, sub_indices = torch.sort(sub_mask, descending=True)
            split_point = int((1 - sparsity) * mask_len)
            important_sub_indices = sub_indices[:split_point]
            important_indices = mask_indices[important_sub_indices]
            unimportant_sub_indices = sub_indices[split_point:]
            unimportant_indices = mask_indices[unimportant_sub_indices]
            trans_mask = mask.clone()
            trans_mask[:] = -float("inf")
            trans_mask[important_indices] = float("inf")
        else:
            _, indices = torch.sort(mask, descending=True)
            mask_len = mask.shape[0]
            split_point = int((1 - sparsity) * mask_len)
            important_indices = indices[:split_point]
            unimportant_indices = indices[split_point:]
            trans_mask = mask.clone()
            trans_mask[important_indices] = float("inf")
            trans_mask[unimportant_indices] = -float("inf")

        return trans_mask

    def visualize_graph(
        self,
        node_idx: int,
        edge_index: Tensor,
        edge_mask: Tensor,
        y: Tensor = None,
        threshold: float = None,
        nolabel: bool = True,
        **kwargs,
    ) -> Tuple[Axes, nx.DiGraph]:
        r"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=kwargs.get("num_nodes"))
        assert edge_mask.size(0) == edge_index.size(1)

        if self.molecule:
            atomic_num = torch.clone(y)

        # Only operate on a k-hop subgraph around `node_idx`.
        subset, edge_index, _, hard_edge_mask = subgraph(
            node_idx,
            self.__num_hops__,
            edge_index,
            relabel_nodes=True,
            num_nodes=None,
            flow=self.__flow__(),
        )

        edge_mask = edge_mask[hard_edge_mask]

        # --- temp ---
        edge_mask[edge_mask == float("inf")] = 1
        edge_mask[edge_mask == -float("inf")] = 0
        # ---

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if kwargs.get("dataset_name") == "ba_lrp":
            y = torch.zeros(edge_index.max().item() + 1, device=edge_index.device)
        if y is None:
            y = torch.zeros(edge_index.max().item() + 1, device=edge_index.device)
        else:
            y = y[subset]

        if self.molecule:
            atom_colors = {
                6: "#8c69c5",
                7: "#71bcf0",
                8: "#aef5f1",
                9: "#bdc499",
                15: "#c22f72",
                16: "#f3ea19",
                17: "#bdc499",
                35: "#cc7161",
            }
            node_colors = [None for _ in range(y.shape[0])]
            for y_idx in range(y.shape[0]):
                node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]
        else:
            atom_colors = {0: "#8c69c5", 1: "#c56973", 2: "#a1c569", 3: "#69c5ba"}
            node_colors = [None for _ in range(y.shape[0])]
            for y_idx in range(y.shape[0]):
                node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]

        data = Data(edge_index=edge_index, att=edge_mask, y=y, num_nodes=y.size(0)).to(
            "cpu"
        )
        G = to_networkx(data, node_attrs=["y"], edge_attrs=["att"])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        kwargs["with_labels"] = kwargs.get("with_labels") or True
        kwargs["font_size"] = kwargs.get("font_size") or 10
        kwargs["node_size"] = kwargs.get("node_size") or 250
        kwargs["cmap"] = kwargs.get("cmap") or "cool"

        # calculate Graph positions
        pos = nx.kamada_kawai_layout(G)
        ax = plt.gca()

        for source, target, data in G.edges(data=True):
            ax.annotate(
                "",
                xy=pos[target],
                xycoords="data",
                xytext=pos[source],
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="->",
                    lw=max(data["att"], 0.5) * 2,
                    alpha=max(data["att"], 0.4),  # alpha control transparency
                    color="#e1442a",  # color control color
                    shrinkA=sqrt(kwargs["node_size"]) / 2.0,
                    shrinkB=sqrt(kwargs["node_size"]) / 2.0,
                    connectionstyle="arc3,rad=0.08",  # rad control angle
                ),
            )
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, **kwargs)
        # define node labels
        if self.molecule:
            if nolabel:
                node_labels = {
                    n: f"{self.table(atomic_num[n].int().item())}" for n in G.nodes()
                }
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
            else:
                node_labels = {
                    n: f"{n}:{self.table(atomic_num[n].int().item())}"
                    for n in G.nodes()
                }
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
        else:
            if not nolabel:
                nx.draw_networkx_labels(G, pos, **kwargs)

        return ax, G

    def eval_related_pred(
        self, x: Tensor, edge_index: Tensor, edge_masks: List[Tensor], **kwargs
    ):

        node_idx = kwargs.get("node_idx")
        node_idx = (
            0 if node_idx is None else node_idx
        )  # graph level: 0, node level: node_idx
        related_preds = []

        # change the mask from -inf ~ +inf into 0 ~ 1
        for ex_label, edge_mask in enumerate(edge_masks):
            #             if self.hard_edge_mask is not None:
            #                 sparsity = 1.0 - (edge_mask[self.hard_edge_mask] != 0).sum() / edge_mask[self.hard_edge_mask].size(0)
            #             else:
            #                 sparsity = 1.0 - (edge_mask != 0).sum() / edge_mask.size(0)

            """
            Correct sparsity computation
            """
            if self.hard_edge_mask is not None:
                edge_mask[~self.hard_edge_mask.bool()] = 0
            if edge_mask.view(-1).shape[0] == edge_index.shape[1]:
                selected_edge_index = edge_index[:, edge_mask.bool()]
            else:
                self_loop_edge_index, _ = add_self_loops(
                    edge_index, num_nodes=self.num_nodes
                )
                selected_edge_index = self_loop_edge_index[:, edge_mask.bool()]

            selected_nodes = selected_edge_index.view(-1).unique()
            sparsity = torch.tensor(1.0 - selected_nodes.shape[0] / x.shape[0])
            """
            Correct sparsity computation
            """

            self.edge_mask.data = torch.ones(edge_mask.size(), device=self.device)
            ori_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            self.edge_mask.data = edge_mask
            masked_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # mask out important elements for fidelity calculation
            self.edge_mask.data = 1.0 - edge_mask  # keep Parameter's id
            maskout_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # zero_mask
            self.edge_mask.data = torch.zeros(edge_mask.size(), device=self.device)
            zero_mask_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            related_preds.append(
                {
                    "zero": zero_mask_pred[node_idx],
                    "masked": masked_pred[node_idx],
                    "maskout": maskout_pred[node_idx],
                    "origin": ori_pred[node_idx],
                    "sparsity": sparsity,
                }
            )

            # Adding proper activation function to the models' outputs.
            tmp_result_dict = {}
            for key, pred in related_preds[ex_label].items():
                if key in ["sparsity"]:
                    tmp_result_dict[key] = pred.item()
                else:
                    tmp_result_dict[key] = pred.reshape(-1).softmax(0)[ex_label].item()
            related_preds[ex_label] = tmp_result_dict

        self.__clear_masks__()
        return related_preds
