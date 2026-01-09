''''
# Staying consistent with Karhadkar et al. (2022)
# https://github.com/kedar2/FoSR/blob/1a7360c2c77c42624bdc7ffef1490a2eb0a8afd0/models/graph_model.py#L77
'''

from argparse import Namespace
from typing import Optional

from torch import Tensor
from torch.nn import Module, Sequential, Linear, BatchNorm1d
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.conv import GINConv

from model.dropout.base import BaseDropout


class GINLayer(GINConv):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_strategy: BaseDropout,
        activation: Optional[Module] = None,
        add_self_loops: bool = False,   # ignored
        normalize: bool = False,        # ignored
        bias: bool = False,
        others: Optional[Namespace] = None,
    ):

        # https://github.com/kedar2/FoSR/blob/1a7360c2c77c42624bdc7ffef1490a2eb0a8afd0/models/graph_model.py#L34
        nn = Sequential(
            Linear(in_channels, out_channels, bias=bias),
            BatchNorm1d(out_channels),
            activation,
            Linear(out_channels, out_channels, bias=bias),
        )
        super(GINLayer, self).__init__(nn=nn)

        self.activation = activation
        self.drop_strategy = drop_strategy

    def treat_adj_mat(self, edge_index):

        # Don't want self-loops because we add (1+eps)*x to the message-aggregation output
        edge_index, _ = remove_self_loops(edge_index)
        # Want to compute edge weights first because in-degrees will change after dropping,
        #   so can't get accurate dropping probability for the edges
        try:
            dropping_probs = self.drop_strategy.compute_edge_dropping_probs(edge_index)
            edge_weight = 1 / (1-dropping_probs.to(edge_index.device))
        except NotImplementedError:
            edge_weight = None
        edge_index, edge_weight = self.drop_strategy.apply_adj_mat(edge_index, edge_weight)

        return edge_index, edge_weight

    def message_passing(self, edge_index, x, edge_weight):

        # https://github.com/pyg-team/pytorch_geometric/issues/9772
        # propagate_type: (x: PairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        return out

    def feature_transformation(self, out):

        out = self.nn(out)
        if self.activation is not None:
            out = self.activation(out)

        return out

    def forward(self, x: Tensor, edge_index: Adj):

        # DROPOUT
        x = self.drop_strategy.apply_feature_mat(x)
        # TREAT ADJACENCY MATRIX
        edge_index, edge_weight = self.treat_adj_mat(edge_index)
        # MESSAGE PASSING
        out = self.message_passing(edge_index, x=x, edge_weight=edge_weight) + \
            (1+self.eps) * self.drop_strategy.apply_message_mat(x)
        # APPLY TRANSFORMATION
        out = self.feature_transformation(out)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor):

        # GINConv does not use edge_weight, but we need it because 
        #   edge_index is dropped from but it is not normalized here
        if edge_weight is not None:
            x_j = x_j * edge_weight.view(-1, 1)

        # drop from message matrix -- drop message
        x_j = self.drop_strategy.apply_message_mat(x_j)

        return x_j