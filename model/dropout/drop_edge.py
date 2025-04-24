from typing import Optional
from argparse import Namespace
from torch_geometric.utils import dropout_edge
from model.dropout.base import BaseDropout


class DropEdge(BaseDropout):

    def __init__(self, dropout_prob: float = 0.5, others: Optional[Namespace] = None):

        super(DropEdge, self).__init__(dropout_prob)
    
    def apply_feature_mat(self, x):

        return super(DropEdge, self).apply_feature_mat(x)
    
    def apply_adj_mat(self, edge_index, edge_attr=None):

        if not self.training or self.dropout_prob == 0.0:
            return edge_index, edge_attr
            
        edge_index, edge_mask = dropout_edge(edge_index, p=self.dropout_prob)
        edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

        return edge_index, edge_attr
    
    def apply_message_mat(self, messages):

        return super(DropEdge, self).apply_message_mat(messages)