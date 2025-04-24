from argparse import Namespace
from typing import Optional

from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import GCNConv

from model.dropout.base import BaseDropout


# TODO: Graph Drop Connect needs an implementation of the aggregation function (after the message step)

class GCNLayer(GCNConv):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_strategy: BaseDropout,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        others: Optional[Namespace] = None,
    ):

        super(GCNLayer, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            add_self_loops=add_self_loops,
            normalize=normalize,
            bias=bias,
        )
        
        self.drop_strategy = drop_strategy

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:

        x = self.drop_strategy.apply_feature_mat(x)
        # Don't need to drop self loops because GCNConv.forward() adds only remaining self loops
        edge_index, edge_weight = self.drop_strategy.apply_adj_mat(edge_index, edge_weight)

        return super(GCNLayer, self).forward(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:

        # Drop from message matrix -- drop message
        messages = super(GCNLayer, self).message(
            x_j=x_j,
            edge_weight=edge_weight,
        )
        out = self.drop_strategy.apply_message_mat(messages)

        return out