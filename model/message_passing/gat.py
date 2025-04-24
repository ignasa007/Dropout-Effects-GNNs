from typing import Optional
from argparse import Namespace

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import leaky_relu
from torch_geometric.utils import softmax, remove_self_loops
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import GATConv

from model.dropout.base import BaseDropout
from model.message_passing.pretreatment import ModelPretreatment


class GATLayer(GATConv):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_strategy: BaseDropout,
        activation: Module,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        others: Optional[Namespace] = None,
    ):

        assert out_channels % others.attention_heads == 0
        
        super(GATLayer, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels//others.attention_heads,
            heads=others.attention_heads,
            concat=True,
            add_self_loops=add_self_loops,
            bias=bias,
        )

        self.pt = ModelPretreatment(add_self_loops, normalize)
        self.activation = activation
        self.drop_strategy = drop_strategy

    def feature_transformation(self, x):

        x = self.drop_strategy.apply_feature_mat(x)
        x = self.lin(x)

        return x

    def treat_adj_mat(self, edge_index, num_nodes, dtype):
        
        if self.add_self_loops: # going to add self loops in pretreatment
            edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = self.drop_strategy.apply_adj_mat(edge_index, None)
        edge_index, edge_weight = self.pt.pretreatment(num_nodes, edge_index, dtype)

        return edge_index, edge_weight

    def message_passing(self, edge_index, x, alpha):

        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out.flatten(start_dim=1)  # concatenate heads
        
        return out
    
    def nonlinearity(self, out):

        if self.bias is not None:
            out = out + self.bias

        out = self.activation(out)

        return out
    
    def forward(self, x: Tensor, edge_index: Adj):

        # FEATURE TRANSFORMATION
        x = self.feature_transformation(x)

        # SOURCE AND TARGET FEATURES
        x_src = x_dst = x.view(-1, self.heads, self.out_channels)   # split heads
        x = (x_src, x_dst)
        # SOURCE AND TARGET ATTENTION WEIGHTS
        alpha_src = (x_src*self.att_src).sum(dim=-1)
        alpha_dst = (x_dst*self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        # TREAT ADJACENCY MATRIX
        edge_index, _ = self.treat_adj_mat(edge_index, num_nodes=x_src.size(0), dtype=x_src.dtype)
        # MESSAGE PASSING
        out = self.message_passing(edge_index, x=x, alpha=alpha)
        # APPLY ACTIVATION
        out = self.nonlinearity(out)

        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        
        alpha = alpha_j + alpha_i
        alpha = leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        
        x_j = alpha.unsqueeze(-1) * x_j
        x_j = self.drop_strategy.apply_message_mat(x_j)

        return x_j