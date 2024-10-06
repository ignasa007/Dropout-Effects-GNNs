from argparse import Namespace
from typing import Union, Optional

from torch import Tensor, BoolTensor
from torch.nn import Module, Linear, Sequential


class BaseHead(Module):

    def __init__(self, layer_sizes: list, activation: Module, others: Optional[Namespace]):

        super(BaseHead, self).__init__()

        if others.task_name.startswith('graph'):
            from model.readout.utils import get_pooler
            self.pooler = get_pooler(others.pooler)
        
        module_list = []
        for in_channels, out_channels in zip(layer_sizes[:-1], layer_sizes[1:]):
            module_list.append(Linear(
                in_features=in_channels,
                out_features=out_channels,
                bias=True
            ))
            module_list.append(activation)

        # the output layer does not use any activation
        self.ffn = Sequential(*module_list[:-1])

    def preprocess(self, node_repr: Tensor, mask: Optional[Union[Tensor, BoolTensor]] = None):

        '''
        Preprocess the input:
            - for node-level tasks, filter out the embeddings using $mask.
            - for graph-level tasks, compute the mean of the node embeddings from each graph.
        '''

        raise NotImplementedError

    def forward(self, node_repr: Tensor, mask: Optional[Union[Tensor, BoolTensor]] = None):

        '''
        Process the node embeddings and compute the loss plus any other metrics.
        
        Args:
            node_repr: node representations as returned by the model.
            mask: 
                - for node-level tasks, specify indices to compute the metrics over.
                - for graph-level tasks, specify node sizes for the batch of graphs. 
        '''

        node_repr = self.preprocess(node_repr, mask)
        out = self.ffn(node_repr)

        return out