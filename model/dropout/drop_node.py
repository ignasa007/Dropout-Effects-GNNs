from typing import Optional
from argparse import Namespace
import torch
from model.dropout.base import BaseDropout


class DropNode(BaseDropout):

    def __init__(self, dropout_prob: float = 0.5, others: Optional[Namespace] = None):

        super(DropNode, self).__init__(dropout_prob)
    
    def apply_feature_mat(self, x):

        if not self.training or self.dropout_prob == 0.0:
            return x
        
        unif_samples = torch.rand(x.size(0), 1, device=x.device)
        node_mask = unif_samples > self.dropout_prob

        x = (x*node_mask) / (1-self.dropout_prob)

        return x

    def apply_adj_mat(self, edge_index, edge_attr=None):
        
        return super(DropNode, self).apply_adj_mat(edge_index, edge_attr)
    
    def apply_message_mat(self, messages):

        return super(DropNode, self).apply_message_mat(messages)