import os
import torch
from methods.base_explainer import ExplainerBase
from torch import Tensor
from typing import List, Dict, Tuple
from torch_geometric.utils import add_self_loops


class PGExplainer_edges(ExplainerBase):
    def __init__(self, pgexplainer, model, molecule: bool):
        super().__init__(
            model=model, explain_graph=pgexplainer.explain_graph, molecule=molecule
        )
        self.explainer = pgexplainer

    def forward(
        self, x: Tensor, edge_index: Tensor, **kwargs
    ) -> Tuple[List, List, List[Dict]]:
        # set default subgraph with 10 edges

        num_classes = kwargs.get("num_classes")
        self.model.eval()
        self.explainer.__clear_masks__()

        x = x.to(self.device)

        ##------ normalize ---------
        # Need to normalize because of the add_self_loops below
        V = x.shape[0]
        E = edge_index.shape[1]
        sparsity = 1 - (1 - kwargs.get("sparsity")) * V / (V + E)
        ##------ normalize ---------

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = edge_index.to(self.device)

        if self.explain_graph:
            embed = self.model.get_emb(x, edge_index)
            _, edge_mask = self.explainer.explain(
                x, edge_index, embed=embed, tmp=1.0, training=False
            )

        else:
            node_idx = kwargs.get("node_idx")
            assert kwargs.get("node_idx") is not None, "please input the node_idx"
            x, edge_index, _, subset, _ = self.explainer.get_subgraph(
                node_idx, x, edge_index
            )
            self.hard_edge_mask = edge_index.new_empty(
                edge_index.size(1), device=self.device, dtype=torch.bool
            )
            self.hard_edge_mask.fill_(True)

            new_node_idx = torch.where(subset == node_idx)[0]
            embed = self.model.get_emb(x, edge_index)
            _, edge_mask = self.explainer.explain(
                x,
                edge_index,
                embed=embed,
                tmp=1.0,
                training=False,
                node_idx=new_node_idx,
            )

        # edge_masks
        edge_masks = [edge_mask for _ in range(num_classes)]
        # Calculate mask
        """
        Notice: the sparsity is normalized by V/(V+E) to avoid the self_loop issue (see code above)
        """
        hard_edge_masks = [
            self.control_sparsity(edge_mask, sparsity=sparsity).sigmoid()
            for _ in range(num_classes)
        ]

        self.__clear_masks__()
        self.__set_masks__(x, edge_index)
        with torch.no_grad():
            if self.explain_graph:
                related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks)
            else:
                related_preds = self.eval_related_pred(
                    x, edge_index, hard_edge_masks, node_idx=new_node_idx
                )

        self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds
