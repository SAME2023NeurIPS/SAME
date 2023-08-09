import torch
import torch.nn as nn
from functools import partial
from typing import Union, List, Callable, Tuple
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import GCNConv, GATConv, GINConv
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
from torch import Tensor
from torch_sparse import SparseTensor, fill_diag, matmul
from torch_geometric.typing import Adj, OptTensor, Size, OptPairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from dig.xgraph.models import GNNPool
from collections import OrderedDict


def get_gnnNets(input_dim, output_dim, model_config):
    if model_config.gnn_name.lower() == 'gcn':
        gcn_model_param_names = GCNNet.__init__.__code__.co_varnames
        gcn_model_params = {param_name: getattr(model_config.param, param_name)
                            for param_name in gcn_model_param_names
                            if param_name in model_config.param.keys()}
        return GCNNet(input_dim=input_dim,
                      output_dim=output_dim,
                      **gcn_model_params)
    elif model_config.gnn_name.lower() == "gin":
        gin_model_param_names = GINNet.__init__.__code__.co_varnames
        gin_model_params = {
            param_name: getattr(model_config.param, param_name)
            for param_name in gin_model_param_names
            if param_name in model_config.param.keys()
        }
        return GINNet(input_dim=input_dim, output_dim=output_dim, **gin_model_params)

    elif model_config.gnn_name.lower() == "gat":
        gat_model_param_names = GATNet.__init__.__code__.co_varnames
        gat_model_params = {
            param_name: getattr(model_config.param, param_name)
            for param_name in gat_model_param_names
            if param_name in model_config.param.keys()
        }
        return GATNet(input_dim=input_dim, output_dim=output_dim, **gat_model_params)
    else:
        raise ValueError(
            f"GNN name should be gcn " f"and {model_config.gnn_name} is not defined."
        )


def identity(x: torch.Tensor, batch: torch.Tensor):
    return x


def cat_max_sum(x, batch):
    node_dim = x.shape[-1]
    num_node = 25
    x = x.reshape(-1, num_node, node_dim)
    return torch.cat([x.max(dim=1)[0], x.sum(dim=1)], dim=-1)


def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool,
        'identity': identity,
        "cat_max_sum": cat_max_sum,
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    return readout_func_dict[readout.lower()]


# GNN_LRP takes GNNPool class as pooling layer
class GNNPool(GNNPool):
    def __init__(self, readout):
        super().__init__()
        self.readout = get_readout_layers(readout)

    def forward(self, x, batch):
        return self.readout(x, batch)


def get_nonlinear(nonlinear):
    nonlinear_func_dict = {
        "relu": F.relu,
        "leakyrelu": partial(F.leaky_relu, negative_slope=0.2),
        "sigmoid": F.sigmoid,
        "elu": F.elu
    }
    return nonlinear_func_dict[nonlinear]


class GNNBase(nn.Module):
    def __init__(self):
        super(GNNBase, self).__init__()

    def _argsparse(self, *args, **kwargs):
        r""" Parse the possible input types.
        If the x and edge_index are in args, follow the args.
        In other case, find them in kwargs.
        """
        if args:
            if len(args) == 1:
                data = args[0]
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, 'batch'):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)

            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)

            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]

            else:
                raise ValueError(f"forward's args should take 1, 2 or 3 arguments but got {len(args)}")
        else:
            data: Batch = kwargs.get('data')
            if not data:
                x = kwargs.get('x')
                edge_index = kwargs.get('edge_index')
                assert x is not None, "forward's args is empty and required node features x is not in kwargs"
                assert edge_index is not None, "forward's args is empty and required edge_index is not in kwargs"
                batch = kwargs.get('batch')
                if not batch:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
            else:
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, 'batch'):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        return x, edge_index, batch

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            if key in self.state_dict().keys():
                new_state_dict[key] = state_dict[key]

        super(GNNBase, self).load_state_dict(new_state_dict)


# GCNConv
class GCNConv(GCNConv):
    def __init__(self, *args, **kwargs):
        super(GCNConv, self).__init__(*args, **kwargs)
        self.edge_weight = None
        self.weight = nn.Parameter(self.lin.weight.data.T.clone().detach())

    # add edge_weight for normalize=False
    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize and edge_weight is None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # new
        elif not self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    if edge_weight is None:
                        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
                    if self.add_self_loops:
                        edge_index, edge_weight = add_self_loops(
                            edge_index, edge_weight, num_nodes=x.size(self.node_dim))
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    adj_t = edge_index
                    if not adj_t.has_value():
                        adj_t = adj_t.fill_value(1.)
                    if self.add_self_loops:
                        adj_t = fill_diag(adj_t, 1.)
                    edge_index = adj_t
                    if self.cached:
                        self._cached_adj_t = edge_index

        # --- add require_grad ---
        edge_weight.requires_grad_(True)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out += self.bias

        # --- My: record edge_weight ---
        self.edge_weight = edge_weight

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self._explain):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self._explain:
                edge_mask = self.__edge_mask__
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)


class GCNNet(GNNBase):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 gnn_latent_dim: Union[List[int]],
                 gnn_dropout: float = 0.0,
                 gnn_emb_normalization: bool = False,
                 gcn_adj_normalization: bool = True,
                 add_self_loop: bool = True,
                 gnn_nonlinear: str = 'relu',
                 readout: str = 'mean',
                 concate: bool = False,
                 fc_latent_dim: Union[List[int]] = [],
                 fc_dropout: float = 0.0,
                 fc_nonlinear: str = 'relu',
                 ):
        super(GCNNet, self).__init__()
        # first and last layer - dim_features and classes
        self.input_dim = input_dim
        self.output_dim = output_dim
        # GNN part
        self.gnn_latent_dim = gnn_latent_dim
        self.gnn_dropout = gnn_dropout
        self.num_gnn_layers = len(self.gnn_latent_dim)
        self.add_self_loop = add_self_loop
        self.gnn_emb_normalization = gnn_emb_normalization
        self.gcn_adj_normalization = gcn_adj_normalization
        self.gnn_nonlinear = get_nonlinear(gnn_nonlinear)
        self.concate = concate
        # readout
        self.readout_layer = GNNPool(readout)
        # FC part
        self.fc_latent_dim = fc_latent_dim
        self.fc_dropout = fc_dropout
        self.num_mlp_layers = len(self.fc_latent_dim) + 1
        self.fc_nonlinear = get_nonlinear(fc_nonlinear)

        if self.concate:
            self.emb_dim = sum(self.gnn_latent_dim)
        else:
            self.emb_dim = self.gnn_latent_dim[-1]

        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, self.gnn_latent_dim[0],
                                  add_self_loops=self.add_self_loop,
                                  normalize=self.gcn_adj_normalization))
        for i in range(1, self.num_gnn_layers):
            self.convs.append(GCNConv(self.gnn_latent_dim[i - 1], self.gnn_latent_dim[i],
                                      add_self_loops=self.add_self_loop,
                                      normalize=self.gcn_adj_normalization))
        # FC layers
        self.mlps = nn.ModuleList()
        if self.num_mlp_layers > 1:
            self.mlps.append(nn.Linear(self.emb_dim, self.fc_latent_dim[0]))

            for i in range(1, self.num_mlp_layers - 1):
                self.mlps.append(nn.Linear(self.fc_latent_dim[i - 1], self.fc_latent_dim[1]))
            self.mlps.append(nn.Linear(self.fc_latent_dim[-1], self.output_dim))
        else:
            self.mlps.append(nn.Linear(self.emb_dim, self.output_dim))

    def device(self):
        return self.convs[0].weight.device

    def get_emb(self, *args, **kwargs):
        #  node embedding for GNN
        x, edge_index, _ = self._argsparse(*args, **kwargs)
        xs = []
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index)
            if self.gnn_emb_normalization:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_nonlinear(x)
            x = F.dropout(x, self.gnn_dropout)
            xs.append(x)

        if self.concate:
            return torch.cat(xs, dim=1)
        else:
            return x

    def forward(self, *args, **kwargs):
        _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb = self.get_emb(*args, **kwargs)
        # pooling process
        x = self.readout_layer(emb, batch)

        for i in range(self.num_mlp_layers - 1):
            x = self.mlps[i](x)
            x = self.fc_nonlinear(x)
            x = F.dropout(x, p=self.fc_dropout)

        logits = self.mlps[-1](x)
        return logits

class GINConv(GINConv):
    def __init__(
        self, nn: Callable, eps: float = 0.0, train_eps: bool = False, **kwargs
    ):
        super().__init__(nn, eps, train_eps, **kwargs)
        self.edge_weight = None
        self.fc_steps = None
        self.reweight = None

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
        task="explain",
        **kwargs,
    ) -> Tensor:
        """"""
        self.num_nodes = x.shape[0]
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        if edge_weight is not None:
            self.edge_weight = edge_weight
            assert edge_weight.shape[0] == edge_index.shape[1]
            self.reweight = False
        else:
            edge_index, _ = remove_self_loops(edge_index)
            self_loop_edge_index, _ = add_self_loops(
                edge_index, num_nodes=self.num_nodes
            )
            if self_loop_edge_index.shape[1] != edge_index.shape[1]:
                edge_index = self_loop_edge_index
            self.reweight = True
        out = self.propagate(edge_index, x=x[0], size=None)

        if task == "explain":
            layer_extractor = []
            hooks = []

            def register_hook(module: nn.Module):
                if not list(module.children()):
                    hooks.append(module.register_forward_hook(forward_hook))

            def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
                # input contains x and edge_index
                layer_extractor.append((module, input[0], output))

            # --- register hooks ---
            self.nn.apply(register_hook)

            nn_out = self.nn(out)

            for hook in hooks:
                hook.remove()

            fc_steps = []
            step = {"input": None, "module": [], "output": None}
            for layer in layer_extractor:
                if isinstance(layer[0], nn.Linear):
                    if step["module"]:
                        fc_steps.append(step)
                    # step = {'input': layer[1], 'module': [], 'output': None}
                    step = {"input": None, "module": [], "output": None}
                step["module"].append(layer[0])
                if kwargs.get("probe"):
                    step["output"] = layer[2]
                else:
                    step["output"] = None

            if step["module"]:
                fc_steps.append(step)
            self.fc_steps = fc_steps
        else:
            nn_out = self.nn(out)

        return nn_out

    def message(self, x_j: Tensor) -> Tensor:
        if self.reweight:
            edge_weight = torch.ones(x_j.shape[0], device=x_j.device)
            edge_weight.data[-self.num_nodes :] += self.eps
            edge_weight = edge_weight.detach().clone()
            edge_weight.requires_grad_(True)
            self.edge_weight = edge_weight
        return x_j * self.edge_weight.view(-1, 1)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if isinstance(edge_index, SparseTensor) and self.fuse and not self.__explain__:
            coll_dict = self.__collect__(
                self.__fused_user_args__, edge_index, size, kwargs
            )

            msg_aggr_kwargs = self.inspector.distribute(
                "message_and_aggregate", coll_dict
            )
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute("update", coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)

            msg_kwargs = self.inspector.distribute("message", coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self._explain:
                edge_mask = self.__edge_mask__
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute("update", coll_dict)
            return self.update(out, **update_kwargs)


class GINNet(GNNBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        gnn_latent_dim: Union[List[int]],
        gnn_dropout: float = 0.0,
        gnn_emb_normalization: bool = False,
        readout: str = "mean",
        fc_latent_dim: Union[List[int]] = [],
        fc_dropout: float = 0.0,
        fc_nonlinear: str = "relu",
    ):
        super(GINNet, self).__init__()
        # first and last layer - dim_features and classes
        self.input_dim = input_dim
        self.output_dim = output_dim
        # GNN part
        self.gnn_latent_dim = gnn_latent_dim
        self.gnn_dropout = gnn_dropout
        self.num_gnn_layers = len(self.gnn_latent_dim)
        self.gnn_emb_normalization = gnn_emb_normalization
        # readout
        self.readout_layer = GNNPool(readout)
        # FC part
        self.fc_latent_dim = fc_latent_dim
        self.fc_dropout = fc_dropout
        self.num_mlp_layers = len(self.fc_latent_dim) + 1
        self.fc_nonlinear = get_nonlinear(fc_nonlinear)

        self.emb_dim = self.gnn_latent_dim[-1]
        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(input_dim, self.gnn_latent_dim[0]),
                    nn.BatchNorm1d(self.gnn_latent_dim[0]),
                    nn.ReLU(),
                    nn.Linear(self.gnn_latent_dim[0], self.gnn_latent_dim[0]),
                ),
                train_eps=True,
            )
        )

        for i in range(1, self.num_gnn_layers):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(self.gnn_latent_dim[i - 1], self.gnn_latent_dim[i]),
                        nn.BatchNorm1d(self.gnn_latent_dim[i]),
                        nn.ReLU(),
                        nn.Linear(self.gnn_latent_dim[i], self.gnn_latent_dim[i]),
                    ),
                    train_eps=True,
                )
            )

        # FC layers
        self.mlps = nn.ModuleList()
        if self.num_mlp_layers > 1:
            self.mlps.append(nn.Linear(self.emb_dim, self.fc_latent_dim[0]))

            for i in range(1, self.num_mlp_layers - 1):
                self.mlps.append(
                    nn.Linear(self.fc_latent_dim[i - 1], self.fc_latent_dim[1])
                )
            self.mlps.append(nn.Linear(self.fc_latent_dim[-1], self.output_dim))
        else:
            self.mlps.append(nn.Linear(self.emb_dim, self.output_dim))

    def device(self):
        return self.convs[0].weight.device

    def get_emb(self, *args, **kwargs):
        #  node embedding for GNN
        x, edge_index, _ = self._argsparse(*args, **kwargs)
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index)
            if self.gnn_emb_normalization:
                x = F.normalize(x, p=2, dim=-1)
            x = F.dropout(x, self.gnn_dropout)
        return x

    def forward(self, *args, **kwargs):
        _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb = self.get_emb(*args, **kwargs)
        # pooling process
        x = self.readout_layer(emb, batch)

        for i in range(self.num_mlp_layers - 1):
            x = self.mlps[i](x)
            x = self.fc_nonlinear(x)
            x = F.dropout(x, p=self.fc_dropout)

        logits = self.mlps[-1](x)

        return logits


# +
"""
Adapted from the original SubgraphX implementation of GAT 
https://github.com/divelab/DIG/blob/main/dig/xgraph/SubgraphX/models/GAT.py
"""


class GATNet(GNNBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        gnn_dropout: float = 0.0,
        gnn_emb_normalization: bool = False,
        gat_dropout: float = 0.6,  # dropout in gat layer
        gat_heads: int = 10,  # multi-head
        gat_hidden: int = 10,  # the hidden units for each head
        gat_concate: bool = True,  # the concatenation of the multi-head feature
        num_gat_layer: int = 3,
        readout: str = "mean",
        fc_latent_dim: Union[List[int]] = [],
        fc_dropout: float = 0.0,
    ):

        super(GATNet, self).__init__()
        self.mlp_hidden = fc_latent_dim
        self.emb_normlize = gnn_emb_normalization
        self.num_gnn_layers = num_gat_layer
        self.num_mlp_layers = len(fc_latent_dim) + 1
        self.dense_dim = gat_hidden * gat_heads
        self.readout_layer = GNNPool(readout)

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(
            GATConv(
                input_dim,
                gat_hidden,
                heads=gat_heads,
                dropout=gat_dropout,
                concat=gat_concate,
            )
        )
        for i in range(1, self.num_gnn_layers):
            self.gnn_layers.append(
                GATConv(
                    self.dense_dim,
                    gat_hidden,
                    heads=gat_heads,
                    dropout=gat_dropout,
                    concat=gat_concate,
                )
            )
        self.gnn_non_linear = nn.ReLU()

        self.mlps = nn.ModuleList()
        if self.num_mlp_layers > 1:
            self.mlps.append(nn.Linear(self.dense_dim, mlp_hidden[0]))
            for i in range(1, self.num_mlp_layers - 1):
                self.mlps.append(nn.Linear(self.mlp_hidden[i - 1], self.mlp_hidden[1]))
            self.mlps.append(nn.Linear(self.mlp_hidden[-1], output_dim))
        else:
            self.mlps.append(nn.Linear(self.dense_dim, output_dim))
        self.dropout = nn.Dropout(fc_dropout)
        self.Softmax = nn.Softmax(dim=-1)
        self.mlp_non_linear = nn.ELU()

    def device(self):
        return self.convs[0].weight.device

    def get_emb(self, *args, **kwargs):
        #  node embedding for GNN
        x, edge_index, _ = self._argsparse(*args, **kwargs)
        for i in range(self.num_gnn_layers):
            x = self.gnn_layers[i](x, edge_index)
            if self.emb_normlize:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_non_linear(x)
        return x

    def forward(self, *args, **kwargs):
        _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb = self.get_emb(*args, **kwargs)
        # pooling process
        x = self.readout_layer(emb, batch)

        for i in range(self.num_mlp_layers - 1):
            x = self.mlps[i](x)
            x = self.mlp_non_linear(x)
            x = self.dropout(x)

        logits = self.mlps[-1](x)
        return logits
