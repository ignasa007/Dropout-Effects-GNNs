from typing import Optional
from argparse import Namespace
import torch.nn as nn


class BaseDropout(nn.Module):

    def __init__(self, dropout_prob: float = 0.5, others: Optional[Namespace] = None):

        super(BaseDropout, self).__init__()

        if not isinstance(dropout_prob, (float, int)):
            raise TypeError(f'Parameter `dropout_prob` must be of type `float` or `int` (got {type(dropout_prob)}).')
        
        if not (0.0 < dropout_prob < 1.0):
            raise ValueError(f'Parameter `dropout_prob` must be between 0 and 1 (got {dropout_prob}).')
        
        self.dropout_prob = dropout_prob

    def apply_feature_mat(self, x, training=True):

        '''
        Dropout methods applied to the feature matrix, eg.
            1. Dropout
            2. DropNode

        Args:
            x (Tensor): feature matrix, eg. shape (|V|, H_{i})
        '''

        return x
    
    def apply_adj_mat(self, edge_index, edge_attr=None, training=True):

        '''
        Dropout methods applied to the adjacency matrix, eg.
            1. DropEdge
            2. DropAgg
            3. DropGNN

        Args:
            edge_index (Adj): adjacency matrix, eg. shape (2, |E|)
        '''

        return edge_index, edge_attr
    
    def apply_message_mat(self, messages, training=True):

        '''
        Dropout methods applied to the message matrix, eg.
            1. DropMessage

        Args:
            messages (Tensor): message matrix, eg. shape (|E|, H_{i+1})
        '''

        return messages