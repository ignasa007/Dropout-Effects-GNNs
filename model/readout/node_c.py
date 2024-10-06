from typing import Optional
from torch import Tensor, BoolTensor
from model.readout.base import BaseHead


class NodeClassification(BaseHead):

    def preprocess(self, node_repr: Tensor, mask: Optional[BoolTensor] = None):

        '''
        Preprocess the input -- filter out the masked nodes' embeddings.

        Args:
            node_repr: tensor of shape (N, H), where $N is the number of nodes in the graph, and 
                $H is the dimension of messages.
            mask: boolean tensor of shape (N,) indicating which nodes to compute metrics over.
        '''

        if mask is not None:
            node_repr = node_repr[mask, ...]

        return node_repr