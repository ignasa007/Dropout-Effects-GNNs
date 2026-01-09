from argparse import Namespace
from typing import Union, Optional

from torch import Tensor, BoolTensor
from torch.nn import Identity, Module, Sequential
from torch_geometric.data import Data

from model.message_passing import get_layer
from model.readout import get_readout
from model.activation import get_activation
from model.dropout import get_dropout


class Model(Module):

    def __init__(self, config: Namespace, others: Optional[Namespace] = None):

        super(Model, self).__init__()

        if others is None:
            others = config

        layer_sizes = [others.input_dim] + config.gnn_layer_sizes + config.ffn_layer_sizes + [others.output_dim]
        # 1 for input_dim, 1 for output_dim of *message-passing step*.
        gnn_layer_sizes = layer_sizes[:1+len(config.gnn_layer_sizes)+1]
        # if `config.ffn_layer_sizes` is empty (no readout transformation), then
        #   `ffn_layer_sizes` will be [config.output_dim], but that's okay because
        #   `zip(ffn_layer_sizes[:-1], ffn_layer_sizes[1:])` will be empty.
        ffn_layer_sizes = layer_sizes[1+len(config.gnn_layer_sizes):]
        
        drop_strategy = get_dropout(config.dropout)(config.drop_p, others=others)
        activation = get_activation(config.gnn_activation)()
        gnn_layer = get_layer(config.gnn)
        
        module_list = []
        for i, (in_channels, out_channels) in enumerate(zip(gnn_layer_sizes[:-1], gnn_layer_sizes[1:]), 1):
            # With input having index 0, index of the output of layer `i` is `i`.
            module_list.append(gnn_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                drop_strategy=drop_strategy,
                # If using an FFN, use activation in every layer, else don't use in the last layer
                activation=activation if len(ffn_layer_sizes) >= 2 or i != len(gnn_layer_sizes)-1 else Identity(),
                add_self_loops=config.add_self_loops,
                normalize=config.normalize,
                bias=config.bias,
                others=others,
            ))
        self.message_passing = Sequential(*module_list)

        ffn_head = get_readout(others.task_name)
        activation = get_activation(config.ffn_activation)()
        self.readout = ffn_head(
            layer_sizes=ffn_layer_sizes,
            activation=activation,
            others=others,
        )

    def reset_parameters(self):

        for mp_layer in self.message_passing:
            mp_layer.reset_parameters()

        self.readout.reset_parameters()

    def forward(
        self,
        graph: Data,
        mask: Optional[Union[Tensor, BoolTensor]] = None,
    ):

        x, edge_index = graph.x, graph.edge_index
        
        for mp_layer in self.message_passing:
            x = mp_layer(x, edge_index)

        out = self.readout(x, mask)

        return out