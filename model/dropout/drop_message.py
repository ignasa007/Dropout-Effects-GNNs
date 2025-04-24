from typing import Optional
from argparse import Namespace
from torch.nn.functional import dropout
from model.dropout.base import BaseDropout


class DropMessage(BaseDropout):

    def __init__(self, dropout_prob: float = 0.5, others: Optional[Namespace] = None):

        super(DropMessage, self).__init__(dropout_prob)
    
    def apply_feature_mat(self, x):

        return super(DropMessage, self).apply_feature_mat(x)
    
    def apply_adj_mat(self, edge_index, edge_attr=None):

        return super(DropMessage, self).apply_adj_mat(edge_index, edge_attr)
    
    def apply_message_mat(self, messages):

        if not self.training or self.dropout_prob == 0.0:
            return messages
        
        return dropout(messages, self.dropout_prob)