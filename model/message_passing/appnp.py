from argparse import Namespace
from typing import Optional

from torch.nn import Module

from model.dropout.base import BaseDropout
from model.message_passing.gcn import GCNLayer


class APPNPLayer(GCNLayer):

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

        super(APPNPLayer, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            drop_strategy=drop_strategy,
            activation=activation,
            add_self_loops=add_self_loops,
            normalize=normalize,
            others=others,
        )

        self.power_iter = others.power_iter
        self.teleport_p = others.teleport_p

    def message_passing(self, edge_index, x, edge_weight):

        h = x
        for _ in range(self.power_iter):
            x = (1-self.teleport_p) * self.propagate(edge_index, x=x, edge_weight=edge_weight) \
                +  self.teleport_p  * h
            
        return x