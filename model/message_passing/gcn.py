from argparse import Namespace
from typing import Optional

from torch import Tensor
from torch.nn import Module
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import GCNConv

from model.dropout.base import BaseDropout
from model.message_passing.pretreatment import ModelPretreatment


# TODO: Graph Drop Connect needs an implementation of the aggregation function (after the message step)

class GCNLayer(GCNConv):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_strategy: BaseDropout,
        activation: Module,
        add_self_loops: bool = True,
        normalize: bool = True,
        others: Optional[Namespace] = None,
    ):

        super(GCNLayer, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            add_self_loops=add_self_loops,
            normalize=normalize,
        )
        
        self.pt = ModelPretreatment(add_self_loops, normalize)
        self.activation = activation
        self.drop_strategy = drop_strategy

    def feature_transformation(self, x):

        x = self.drop_strategy.apply_feature_mat(x, self.training)
        x = self.lin(x)

        return x
    
    def treat_adj_mat(self, edge_index, num_nodes, dtype):

        edge_index, _ = self.drop_strategy.apply_adj_mat(edge_index, None, self.training)
        edge_index, edge_weight = self.pt.pretreatment(num_nodes, edge_index, dtype)

        return edge_index, edge_weight
    
    def message_passing(self, edge_index, x, edge_weight):

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
        return out
    
    def nonlinearity(self, out):

        if self.bias is not None:
            out = out + self.bias

        out = self.activation(out)

        return out
    
    def forward(self, x: Tensor, edge_index: Adj):

        # FEATURE TRANSFORMATION
        x = self.feature_transformation(x)
        # TREAT ADJACENCY MATRIX
        edge_index, edge_weight = self.treat_adj_mat(edge_index, num_nodes=x.size(0), dtype=x.dtype)
        # MESSAGE PASSING
        out = self.message_passing(edge_index, x, edge_weight)
        # ADD BIAS
        out = self.nonlinearity(out)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor):

        if edge_weight is not None:
            x_j = x_j * edge_weight.view(-1, 1)

        # drop from message matrix -- drop message
        x_j = self.drop_strategy.apply_message_mat(x_j, self.training)

        return x_j