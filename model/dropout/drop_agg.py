import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from model.dropout.base import BaseDropout


class DropAgg(BaseDropout):

    def __init__(self, dropout_prob=0.5):

        super(DropAgg, self).__init__(dropout_prob)
    
    def apply_feature_mat(self, x, training=True):

        return super(DropAgg, self).apply_feature_mat(x, training)
    
    def apply_adj_mat(self, edge_index, edge_attr=None, training=True):

        if not training or self.dropout_prob == 0.0:
            return edge_index, edge_attr

        num_nodes = maybe_num_nodes(edge_index)
        unif_samples = torch.rand(num_nodes, device=edge_index.device)
        node_mask = unif_samples > self.dropout_prob

        # the edges (i, j) imply a directed edge i -> j
        edge_mask = node_mask[edge_index[1]]
        edge_index = edge_index[:, edge_mask]
        if edge_attr is not None:
            edge_attr = edge_attr[edge_mask]

        return edge_index, edge_attr
    
    def apply_message_mat(self, messages, training=True):

        return super(DropAgg, self).apply_message_mat(messages, training)