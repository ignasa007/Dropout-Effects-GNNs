''''
# Staying consistent with Karhadkar et al. (2022)
# https://github.com/kedar2/FoSR/blob/1a7360c2c77c42624bdc7ffef1490a2eb0a8afd0/models/graph_model.py#L77
'''

from argparse import Namespace
from typing import Optional

from torch import Tensor
from torch.nn import Module, Sequential, Linear
from torch_geometric.typing import Adj
from torch_geometric.utils import remove_self_loops 
from torch_geometric.nn.conv import GINConv

from model.dropout.base import BaseDropout


class GINLayer(GINConv):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_strategy: BaseDropout,
        activation: Module,
        add_self_loops: bool = False,   # ignored
        normalize: bool = False,        # ignored
        bias: bool = False,
        others: Optional[Namespace] = None,
    ):
        
        nn = Sequential(
            Linear(in_channels, out_channels, bias=bias),
            activation,
            Linear(out_channels, out_channels, bias=bias),
        )
        super(GINLayer, self).__init__(nn=nn)

        self.activation = activation
        self.drop_strategy = drop_strategy

    def treat_adj_mat(self, edge_index):
        
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = self.drop_strategy.apply_adj_mat(edge_index, None, self.training)

        return edge_index

    def message_passing(self, edge_index, x):

        out = self.propagate(edge_index, x=(x, x))
        
        return out
    
    def feature_transformation(self, out):

        out = self.nn(out)
        out = self.activation(out)

        return out
    
    def forward(self, x: Tensor, edge_index: Adj):

        # DROPOUT
        x = self.drop_strategy.apply_feature_mat(x, self.training)
        # TREAT ADJACENCY MATRIX
        edge_index = self.treat_adj_mat(edge_index)
        # MESSAGE PASSING
        out = self.message_passing(edge_index, x=x) + (1+self.eps) * x
        # APPLY TRANSFORMATION
        out = self.feature_transformation(out)

        return out

    def message(self, x_j: Tensor):

        # drop from message matrix -- drop message
        x_j = self.drop_strategy.apply_message_mat(x_j, self.training)

        return x_j