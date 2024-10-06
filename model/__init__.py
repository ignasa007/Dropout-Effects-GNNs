from argparse import Namespace
from typing import Union, Optional

from torch import Tensor, BoolTensor
from torch.nn import Module, ModuleList
from torch_geometric.typing import Adj

from model.message_passing import get_layer
from model.readout import get_head
from model.activation import get_activation
from model.dropout import get_dropout


class Model(Module):

    def __init__(self, config: Namespace, others: Optional[Namespace] = None):
        
        super(Model, self).__init__()

        if others is None:
            others = config
        
        drop_strategy = get_dropout(config.dropout)(config.drop_p)
        activation = get_activation(config.gnn_activation)()
        gnn_layer = get_layer(config.gnn)
        gnn_layer_sizes = [others.input_dim] + config.gnn_layer_sizes
        self.message_passing = ModuleList()
        for in_channels, out_channels in zip(gnn_layer_sizes[:-1], gnn_layer_sizes[1:]):
            self.message_passing.append(gnn_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                drop_strategy=drop_strategy,
                activation=activation,
                add_self_loops=config.add_self_loops,
                normalize=config.normalize,
                others=others,
            ))

        ffn_head = get_head(others.task_name)
        ffn_layer_sizes = config.gnn_layer_sizes[-1:] + config.ffn_layer_sizes + [others.output_dim]
        self.readout = ffn_head(
            layer_sizes=ffn_layer_sizes,
            activation=get_activation(config.ffn_activation)(),
            others=others,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        mask: Optional[Union[Tensor, BoolTensor]] = None,
    ):

        for mp_layer in self.message_passing:
            x = mp_layer(x, edge_index)
        
        out = self.readout(x, mask)

        return out