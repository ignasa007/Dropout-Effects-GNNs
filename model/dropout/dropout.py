from typing import Optional
from argparse import Namespace
from torch.nn.functional import dropout
from model.dropout.base import BaseDropout


class Dropout(BaseDropout):

    def __init__(self, dropout_prob: float = 0.5, others: Optional[Namespace] = None):

        super(Dropout, self).__init__(dropout_prob)

    def apply_feature_mat(self, x):
        
        if not self.training or self.dropout_prob == 0.0:
            return x
        
        return dropout(x, self.dropout_prob)
    
    def apply_adj_mat(self, edge_index, edge_attr=None):

        return super(Dropout, self).apply_adj_mat(edge_index, edge_attr)
    
    def apply_message_mat(self, messages):

        return super(Dropout, self).apply_message_mat(messages)