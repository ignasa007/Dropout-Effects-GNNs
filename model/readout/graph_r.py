from typing import Optional
from torch import Tensor
from model.readout.base import BaseHead


class GraphRegression(BaseHead):

    def preprocess(self, node_repr: Tensor, mask: Optional[Tensor] = None):

        '''
        Preprocess the input -- compute the mean of the node embeddings from each graph.

        Args:
            node_repr: tensor of shape (N_1+...+N_B, H), where $N_i is the number of nodes in graph $i,
                $B is the batch size, and $H is the dimension of messages.
            mask: tensor {0, ..., B-1}^N of shape (N,) assigning each node to a graph
        '''

        return self.pooler(x=node_repr, batch=mask)