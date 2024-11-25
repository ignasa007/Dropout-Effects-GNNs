import sympy
from sympy.abc import x
import torch
from torch_geometric.utils import degree, remove_self_loops
from model.dropout.base import BaseDropout


class DropSens(BaseDropout):

    def __init__(self, dropout_prob=0.5):

        super(DropSens, self).__init__(dropout_prob)

    def compute_q(self, edge_index, c=0.9, max_d=10):

        mapper = torch.zeros(max_d+1)
        for d in range(1, max_d+1):
            mapper[d] = float(sympy.N(sympy.real_roots(d*(1-c)*(1-x)-x+x**(d+1))[-2]))
        
        degrees = degree(edge_index[1]).int()
        self.q = torch.Tensor(list(map(lambda d: mapper[d.item()] if d.item()<=max_d else mapper[max_d], degrees)))

    def apply_feature_mat(self, x, training=True):

        return super(DropSens, self).apply_feature_mat(x, training)
    
    def apply_adj_mat(self, edge_index, edge_attr=None, training=True):

        if not training or self.dropout_prob == 0.0:
            return edge_index, edge_attr

        if not hasattr(self, 'q'):
            self.compute_q(edge_index, c=0.9, max_d=10)
        
        edge_index, _ = remove_self_loops(edge_index)
        edge_mask = torch.rand(edge_index.size(1)) >= self.q[edge_index[1]]
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

        return edge_index, edge_attr
    
    def apply_message_mat(self, messages, training=True):

        return super(DropSens, self).apply_message_mat(messages, training)