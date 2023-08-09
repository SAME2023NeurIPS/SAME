"""
Adapted from official OrphicX code
https://github.com/WanyuGroup/CVPR2022-OrphicX/tree/main
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import argparse
import os
from networkx.algorithms.components.connected import connected_components
import sklearn.metrics as metrics
from functools import partial
import sys
import time
import math
import pickle
import shutil
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import torch.nn.functional as F
from torch import nn, optim

import argparse
import torch.nn.modules.loss
from typing import Tuple, List, Dict
from .base_explainer import ExplainerBase
from baselines.methods import causaleffect
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = self.linear(input)
        output = torch.bmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class VGAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, output_dim, dropout):
        super(VGAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, output_dim, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, output_dim, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


# +
class VGAE3(VGAE):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, output_dim, dropout):
        super(VGAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc1_1 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim2, output_dim, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim2, output_dim, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

        # Normalize to stablize training on high dim data
        self.norm1 = nn.LayerNorm(input_feat_dim)
        self.norm1_1 = nn.LayerNorm(hidden_dim1)
        self.norm2 = nn.LayerNorm(hidden_dim2)

    def encode(self, x, adj):
        x = self.norm1(x)
        hidden1 = self.gc1(x, adj)
        hidden1 = self.norm1_1(hidden1)
        hidden2 = self.gc1_1(hidden1, adj)
        hidden2 = self.norm2(hidden2)

        #         hidden1 = self.gc1(x, adj)
        #         hidden2 = self.gc1_1(hidden1, adj)
        return self.gc2(hidden2, adj), self.gc3(hidden2, adj)


# -


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.bmm(z, torch.transpose(z, 1, 2)))
        return adj


class VGAE3MLP(VGAE3):
    def __init__(
        self,
        input_feat_dim,
        hidden_dim1,
        hidden_dim2,
        output_dim,
        decoder_hidden_dim1,
        decoder_hidden_dim2,
        K,
        dropout,
    ):
        super(VGAE3MLP, self).__init__(
            input_feat_dim, hidden_dim1, hidden_dim2, output_dim, dropout
        )
        self.dc = InnerProductDecoderMLP(
            output_dim,
            decoder_hidden_dim1,
            decoder_hidden_dim2,
            dropout,
            act=lambda x: x,
        )


class InnerProductDecoderMLP(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout, act=torch.sigmoid):
        super(InnerProductDecoderMLP, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout = dropout
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = torch.sigmoid(self.fc2(z))
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.bmm(z, torch.transpose(z, 1, 2)))
        return adj


# +
class Orphicx(ExplainerBase):
    """
    Wrapper of VGAE3MLP and classifier (GNN model) in the original Orphicx code.
    The VGAE3MLP and the classifier are stored as self.explainer and self.model to stay consistent with other methods

    """

    def __init__(
        self, vgae, classifier, device, num_causal_factors, explain_graph=True
    ):
        super(Orphicx, self).__init__(model=classifier, explain_graph=True)
        self.explainer = vgae
        self.device = device
        self.K = num_causal_factors

    def explain(self, data):
        mu, logvar = self.explainer.encode(data["sub_feat"], data["sub_adj"])
        alpha_mu = torch.zeros_like(mu)
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        alpha_mu[:, :, : self.K] = eps.mul(std).add_(mu)[:, :, : self.K]
        alpha_adj = torch.sigmoid(self.explainer.dc(alpha_mu))
        alpha_edge_index = alpha_adj[data["sub_adj"].bool()]
        edge_mask = alpha_edge_index

        return edge_mask

    def forward(self, data, sparse_batch, **kwargs) -> Tuple[List, List, List[Dict]]:

        num_classes = kwargs.get("num_classes")
        self.model.eval()
        self.__clear_masks__()

        x = sparse_batch.x.to(self.device)
        edge_index = sparse_batch.edge_index.to(self.device)

        ##------ normalize ---------
        # Normalize the sparsity by V / E
        # because sparsity is defined on nodes, but the mask is in terms of edges.

        V = x.shape[0]
        E = edge_index.shape[1]
        sparsity = 1 - (1 - kwargs.get("sparsity")) * V / E
        ##------ normalize ---------

        if self.explain_graph:
            edge_mask = self.explain(data)

        edge_masks = [edge_mask for _ in range(num_classes)]
        hard_edge_masks = [
            self.control_sparsity(edge_mask, sparsity=sparsity)
            .sigmoid()
            .to(self.device)
            for _ in range(num_classes)
        ]

        self.__clear_masks__()
        self.__set_masks__(x, edge_index)
        with torch.no_grad():
            if self.explain_graph:
                related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks)

        self.__clear_masks__()
        return edge_masks, hard_edge_masks, related_preds


def graph_labeling(G):
    for node in G:
        G.nodes[node]["string"] = 1
    old_strings = tuple([G.nodes[node]["string"] for node in G])
    for iter_num in range(100):
        for node in G:
            string = sorted([G.nodes[neigh]["string"] for neigh in G.neighbors(node)])
            G.nodes[node]["concat_string"] = tuple([G.nodes[node]["string"]] + string)
        d = nx.get_node_attributes(G, "concat_string")
        nodes, strings = zip(*{k: d[k] for k in sorted(d, key=d.get)}.items())
        map_string = dict(
            [[string, i + 1] for i, string in enumerate(sorted(set(strings)))]
        )
        for node in nodes:
            G.nodes[node]["string"] = map_string[G.nodes[node]["concat_string"]]
        new_strings = tuple([G.nodes[node]["string"] for node in G])
        if old_strings == new_strings:
            break
        else:
            old_strings = new_strings
    return G


class GraphSampler(torch.utils.data.Dataset):
    """
    Sample graph and nodes.
    Modified from the GraphSampler function in the official OrphicX implementation.
    https://github.com/WanyuGroup/CVPR2022-OrphicX/blob/98d8d8259439c45661573e575cf956331df16abc/orphicx_graph.py#L176

    Turn PyG dataset to the dense format required by the OrphicX
    """

    def __init__(self, dataset, graph_idxs):
        self.graph_idxs = graph_idxs
        self.dense_data = []

        # Turn sparse to dense
        batch_dataset = Batch.from_data_list(dataset)
        self.dense_x, self.dense_mask = to_dense_batch(
            batch_dataset.x, batch_dataset.batch
        )
        self.dense_adj = to_dense_adj(batch_dataset.edge_index, batch_dataset.batch)
        self.labels = batch_dataset.y

        cum_size = dataset.slices["x"]
        sizes = cum_size - cum_size.roll(1, 0)
        label_onehot = torch.eye(sizes.max().item(), dtype=torch.float)

        for graph_idx in tqdm(graph_idxs):
            adj = self.dense_adj[graph_idx].float()
            feat = self.dense_x[graph_idx, :].float()
            feat_mask = self.dense_mask[graph_idx, :]
            label = self.labels[graph_idx].long()

            G = graph_labeling(nx.from_numpy_array(self.dense_adj[graph_idx].numpy()))
            graph_label = np.array([G.nodes[node]["string"] for node in G])
            graph_label_onehot = label_onehot[graph_label]
            sub_feat = torch.cat((feat, graph_label_onehot), dim=1)
            adj_label = adj + np.eye(adj.shape[0])
            n_nodes = adj.shape[0]
            graph_size = torch.count_nonzero(adj.sum(-1))
            pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / max(
                float(adj.sum()), 1e-3
            )
            pos_weight = torch.from_numpy(np.array(pos_weight))
            norm = torch.tensor(
                adj.shape[0]
                * adj.shape[0]
                / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
            )
            self.dense_data += [
                {
                    "graph_idx": graph_idx,
                    "graph_size": graph_size,
                    "sub_adj": adj,
                    "feat": feat.float(),
                    "feat_mask": feat_mask,
                    "sub_feat": sub_feat.float(),
                    "sub_label": label.float(),
                    "adj_label": adj_label.float(),
                    "n_nodes": torch.Tensor([n_nodes])[0],
                    "pos_weight": pos_weight,
                    "norm": norm,
                }
            ]

    def __len__(self):
        return len(self.graph_idxs)

    def __getitem__(self, idx):
        return self.dense_data[idx]


def dense_to_sparse_batch(x, adj, mask):
    data_list = [
        Data(
            x=x[i][mask[i]], edge_index=dense_to_sparse(adj[i][mask[i]][:, mask[i]])[0]
        )
        for i in range(x.shape[0])
    ]
    sparse_batch = Batch.from_data_list(data_list)
    return sparse_batch


# +
def gaeloss(x, mu, logvar, data):
    return gae_loss(
        preds=x,
        labels=data["adj_label"],
        mu=mu,
        logvar=logvar,
        n_nodes=data["n_nodes"],
        norm=data["norm"],
        pos_weight=data["pos_weight"],
    )


def gae_loss(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    bce = F.binary_cross_entropy_with_logits(
        preds.flatten(1).T, labels.flatten(1).T, pos_weight=pos_weight, reduction="none"
    ).mean(0)
    cost = norm * bce

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = (
        -0.5
        / n_nodes
        * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), -1))
    )

    return cost + KLD


# -


def get_orphicx_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="Mutagenicity", help="Name of dataset."
    )
    parser.add_argument("--output", type=str, default=None, help="output path.")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument(
        "-e", "--epoch", type=int, default=300, help="Number of training epochs."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=128,
        help="Number of samples in a minibatch.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Number of training epochs."
    )
    parser.add_argument("--max_grad_norm", type=float, default=1, help="max_grad_norm.")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate (1 - keep probability).",
    )
    parser.add_argument(
        "--encoder_hidden1",
        type=int,
        default=32,
        help="Number of units in hidden layer 1.",
    )
    parser.add_argument(
        "--encoder_hidden2",
        type=int,
        default=16,
        help="Number of units in hidden layer 2.",
    )
    parser.add_argument(
        "--encoder_output", type=int, default=16, help="Dim of output of VGAE encoder."
    )
    parser.add_argument(
        "--decoder_hidden1",
        type=int,
        default=16,
        help="Number of units in decoder hidden layer 1.",
    )
    parser.add_argument(
        "--decoder_hidden2",
        type=int,
        default=16,
        help="Number of units in decoder  hidden layer 2.",
    )
    parser.add_argument("--K", type=int, default=8, help="Number of casual factors.")
    parser.add_argument(
        "--coef_lambda", type=float, default=0.01, help="Coefficient of gae loss."
    )
    parser.add_argument(
        "--coef_kl", type=float, default=0.01, help="Coefficient of gae loss."
    )
    parser.add_argument(
        "--coef_causal", type=float, default=1.0, help="Coefficient of causal loss."
    )
    parser.add_argument(
        "--coef_size", type=float, default=0.0, help="Coefficient of size loss."
    )
    parser.add_argument(
        "--NX",
        type=int,
        default=1,
        help="Number of monte-carlo samples per causal factor.",
    )
    parser.add_argument(
        "--NA",
        type=int,
        default=1,
        help="Number of monte-carlo samples per causal factor.",
    )
    #     parser.add_argument('--Nalpha', type=int, default=25, help='Number of monte-carlo samples per causal factor.')
    #     parser.add_argument('--Nbeta', type=int, default=100, help='Number of monte-carlo samples per noncausal factor.')
    ## Too large for a single GPU
    parser.add_argument(
        "--Nalpha",
        type=int,
        default=15,
        help="Number of monte-carlo samples per causal factor.",
    )
    parser.add_argument(
        "--Nbeta",
        type=int,
        default=50,
        help="Number of monte-carlo samples per noncausal factor.",
    )

    parser.add_argument(
        "--node_perm",
        action="store_true",
        help="Use node permutation as data augmentation for causal training.",
    )
    parser.add_argument(
        "--load_ckpt", default=None, help="Load parameters from checkpoint."
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument(
        "--patient", type=int, default=100, help="Patient for early stopping."
    )
    parser.add_argument("--plot_info_flow", action="store_true")

    # New args
    #     parser.add_argument('--one_hot_label_dim', type=int, default=100, help='dimensions used for one_hot_label.')
    parser.add_argument(
        "--eval_epoch", type=int, default=1, help="How many epochs for one eval."
    )
    parser.add_argument("--load_processed_dataset", action="store_true")
    return parser.parse_args("")


def eval_model(args, orphicx, dataset, device, criterion, ceparams):
    orphicx.eval()
    with torch.no_grad():
        for data in dataset:
            sparse_batch = dense_to_sparse_batch(
                data["feat"], data["sub_adj"], data["feat_mask"]
            ).to(device)
            for k in data:
                data[k] = data[k].to(device)
            labels = data["sub_label"].long()

            recovered, mu, logvar = orphicx.explainer(data["sub_feat"], data["sub_adj"])
            recovered_adj = torch.sigmoid(recovered)

            nll_loss = criterion(recovered, mu, logvar, data).mean()
            org_adjs = data["sub_adj"]
            org_logits = orphicx.model(sparse_batch.x, sparse_batch.edge_index)
            org_probs = F.softmax(org_logits, dim=1)
            org_log_probs = F.log_softmax(org_logits, dim=1)
            masked_recovered_adj = recovered_adj * data["sub_adj"]
            masked_recovered_adj_hard = (masked_recovered_adj > 0.5).float()
            masked_recovered_sparse_batch = dense_to_sparse_batch(
                data["feat"], masked_recovered_adj_hard, data["feat_mask"]
            )
            recovered_logits = orphicx.model(
                sparse_batch.x, masked_recovered_sparse_batch.edge_index
            )
            recovered_probs = F.softmax(recovered_logits, dim=1)
            recovered_log_probs = F.log_softmax(recovered_logits, dim=1)

            alpha_mu = torch.zeros_like(mu)
            alpha_mu[:, :, : args.K] = mu[:, :, : args.K]
            alpha_adj = torch.sigmoid(orphicx.explainer.dc(alpha_mu))
            masked_alpha_adj = alpha_adj * data["sub_adj"]
            masked_alpha_adj_hard = (masked_alpha_adj > 0.5).float()
            masked_alpha_sparse_batch = dense_to_sparse_batch(
                data["feat"], masked_alpha_adj_hard, data["feat_mask"]
            )
            alpha_logits = orphicx.model(
                sparse_batch.x, masked_alpha_sparse_batch.edge_index
            )

            beta_mu = torch.zeros_like(mu)
            beta_mu[:, :, args.K :] = mu[:, :, args.K :]
            beta_adj = torch.sigmoid(orphicx.explainer.dc(beta_mu))
            masked_beta_adj = beta_adj * data["sub_adj"]
            masked_beta_adj_hard = (masked_beta_adj > 0.5).float()
            masked_beta_sparse_batch = dense_to_sparse_batch(
                data["feat"], masked_beta_adj_hard, data["feat_mask"]
            )
            beta_logits = orphicx.model(
                sparse_batch.x, masked_beta_sparse_batch.edge_index
            )

            causal_loss = []
            beta_info = []
            for idx in random.sample(range(0, data["feat"].shape[0]), args.NX):
                _causal_loss, _ = causaleffect.joint_uncond(
                    ceparams,
                    orphicx.explainer.dc,
                    orphicx.model,
                    data["sub_adj"][idx],
                    data["feat"][idx],
                    act=torch.sigmoid,
                    device=device,
                )
                _beta_info, _ = causaleffect.beta_info_flow(
                    ceparams,
                    orphicx.explainer.dc,
                    orphicx.model,
                    data["sub_adj"][idx],
                    data["feat"][idx],
                    act=torch.sigmoid,
                    device=device,
                )
                causal_loss += [_causal_loss]
                beta_info += [_beta_info]
                for A_idx in random.sample(
                    range(0, data["feat"].shape[0]), args.NA - 1
                ):
                    if args.node_perm:
                        perm = torch.randperm(data["graph_size"][idx])
                        perm_adj = data["sub_adj"][idx].clone().detach()
                        perm_adj[: data["graph_size"][idx]] = perm_adj[perm]
                    else:
                        perm_adj = data["sub_adj"][A_idx]
                    _causal_loss, _ = causaleffect.joint_uncond(
                        ceparams,
                        orphicx.explainer.dc,
                        orphicx.model,
                        perm_adj,
                        data["feat"][idx],
                        act=torch.sigmoid,
                        device=device,
                    )
                    _beta_info, _ = causaleffect.beta_info_flow(
                        ceparams,
                        orphicx.explainer.dc,
                        orphicx.model,
                        perm_adj,
                        data["feat"][idx],
                        act=torch.sigmoid,
                        device=device,
                    )
                    causal_loss += [_causal_loss]
                    beta_info += [_beta_info]

            causal_loss = torch.stack(causal_loss).mean()
            alpha_info = causal_loss
            beta_info = torch.stack(beta_info).mean()
            klloss = F.kl_div(
                F.log_softmax(alpha_logits, dim=1), org_probs, reduction="mean"
            )

        pred_labels = torch.argmax(org_probs, axis=1)
        org_acc = (
            (torch.argmax(org_probs, axis=1) == torch.argmax(recovered_probs, axis=1))
            .float()
            .mean()
        )
        pred_acc = (torch.argmax(recovered_probs, axis=1) == labels).float().mean()
        kl_pred_org = F.kl_div(recovered_log_probs, org_probs, reduction="mean")
        alpha_probs = F.softmax(alpha_logits, dim=1)
        alpha_log_probs = F.log_softmax(alpha_logits, dim=1)
        beta_probs = F.softmax(beta_logits, dim=1)
        beta_log_probs = F.log_softmax(beta_logits, dim=1)
        alpha_gt_acc = (torch.argmax(alpha_probs, axis=1) == labels).float().mean()
        alpha_pred_acc = (
            (torch.argmax(alpha_probs, axis=1) == pred_labels).float().mean()
        )
        alpha_kld = F.kl_div(alpha_log_probs, org_probs, reduction="mean")
        beta_gt_acc = (torch.argmax(beta_probs, axis=1) == labels).float().mean()
        beta_pred_acc = (torch.argmax(beta_probs, axis=1) == pred_labels).float().mean()
        beta_kld = F.kl_div(beta_log_probs, org_probs, reduction="mean")
        alpha_sparsity = masked_alpha_adj.mean((1, 2)) / org_adjs.mean((1, 2)).clamp(
            min=1e-3
        )
        loss = (
            args.coef_lambda * nll_loss
            + args.coef_causal * causal_loss
            + args.coef_kl * klloss
            + args.coef_size * alpha_sparsity.mean()
        )

        eval_losses = [nll_loss, causal_loss, klloss, alpha_sparsity.mean()]
    return loss.item()
