from argparse import Namespace
from typing import Union, Optional

from torch import Tensor, BoolTensor
from torch.nn import Module, Linear, Sequential


def get_pooler(pooler_name: str):

    from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
    
    pooler_map = {
        'mean': global_mean_pool,
        'add': global_add_pool,
        'max': global_max_pool,
    }

    formatted_name = pooler_name.lower()
    if formatted_name not in pooler_map:
        raise ValueError(f'Parameter `pooler_name` not recognised (got `{pooler_name}`).')
    
    pooler = pooler_map.get(formatted_name)
    
    return pooler


class BaseHead(Module):

    def __init__(self, layer_sizes: list, activation: Module, others: Optional[Namespace]):

        super(BaseHead, self).__init__()

        if others.task_name.lower().startswith('graph'):
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

    def reset_parameters(self):
        
        for linear_layer in self.ffn:
            linear_layer.reset_parameters()

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