from argparse import Namespace
from typing import Optional

from torch import Tensor, empty
from torch.nn import Parameter, Module
from torch_geometric.typing import Adj
from torch_geometric.utils import remove_self_loops 

from model.dropout.base import BaseDropout
from model.message_passing.gcn import GCNLayer


class GINLayer(GCNLayer):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_strategy: BaseDropout,
        activation: Module,
        add_self_loops: bool = False,   # ignored
        normalize: bool = False,        # ignored
        bias: bool = True,
        others: Optional[Namespace] = None,
    ):

        super(GINLayer, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            drop_strategy=drop_strategy,
            activation=activation,
            add_self_loops=False,
            normalize=False,
            bias=bias,
            others=others,
        )

        if others.eps is None:
            self.initial_eps = 0.
            self.eps = Parameter(empty(1))
        else:
            self.initial_eps = others.eps
            self.register_buffer('eps', empty(1))
        self.reset_eps()

    def reset_parameters(self):
        
        # Messy logic: need to intialize module before setting parameter `eps`
        # module is initialized in `GCNConv.__init__()`, which also resets parameters
        # but eps is not initialized before that, so cannot reset `eps` till then
        
        super(GINLayer, self).reset_parameters()
        if hasattr(self, 'eps'): self.reset_eps()
        
    def reset_eps(self):

        self.eps.data.fill_(self.initial_eps)
    
    def forward(self, x: Tensor, edge_index: Adj):

        # FEATURE TRANSFORMATION
        x = self.feature_transformation(x)
        # TREAT ADJACENCY MATRIX -- no self loops, no weight normalization => edge_weight = None
        edge_index, edge_weight = self.treat_adj_mat(
            remove_self_loops(edge_index)[0], num_nodes=x.size(0), dtype=x.dtype
        )
        # MESSAGE PASSING
        out = self.message_passing(edge_index, x, edge_weight)
        out = out + (1+self.eps) * x
        # APPLY ACTIVATION
        out = self.nonlinearity(out)

        return out