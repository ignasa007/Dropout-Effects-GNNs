from torch_geometric.utils import remove_self_loops, dropout_edge
from model.dropout.base import BaseDropout


class DropEdge(BaseDropout):

    def __init__(self, dropout_prob=0.5):

        super(DropEdge, self).__init__(dropout_prob)
    
    def apply_feature_mat(self, x, training=True):

        return super(DropEdge, self).apply_feature_mat(x, training)
    
    def apply_adj_mat(self, edge_index, edge_attr=None, training=True):

        edge_index, edge_mask = dropout_edge(edge_index, p=self.dropout_prob, training=training)
        edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

        return edge_index, edge_attr
    
    def apply_message_mat(self, messages, training=True):

        return super(DropEdge, self).apply_message_mat(messages, training)