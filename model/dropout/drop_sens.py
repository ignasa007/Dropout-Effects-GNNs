from typing import Optional
from argparse import Namespace

import numpy as np
import sympy
from sympy.abc import x
import torch
from torch_geometric.utils import degree
from model.dropout.base import BaseDropout


class DropSens(BaseDropout):

    def __init__(self, dropout_prob: float = 0.5, others: Optional[Namespace] = None):

        # Only calling BaseDropout.__init__() because it calls nn.Module.__init__()
        # Cannot directly inherit nn.Module because need default implementations of
        #     apply_feature_mat() and apply_message_mat()

        super(DropSens, self).__init__(dropout_prob)    # Maximum value q_i can take
        self.c = others.info_loss_ratio

    def compute_q(self, edge_index):

        degrees = degree(edge_index[1]).int().tolist()
        unique_degrees = np.unique(degrees) # Sorted array

        mapper = dict()
        for d_i in enumerate(unique_degrees):
            q_i = float(sympy.N(sympy.real_roots(d_i*(1-self.c)*(1-x)-x+x**(d_i+1))[-2]))
            if q_i > self.dropout_prob: break
            mapper[d_i] = q_i
        
        self.q = torch.Tensor(list(map(lambda d_i: mapper.get(d_i, self.dropout_prob), degrees)))

    def apply_feature_mat(self, x, training=True):

        return super(DropSens, self).apply_feature_mat(x, training)
    
    def apply_adj_mat(self, edge_index, edge_attr=None, training=True):

        if not training or self.dropout_prob == 0.0:
            return edge_index, edge_attr

        if not hasattr(self, 'q'):
            self.compute_q(edge_index)
        
        edge_mask = torch.rand(edge_index.size(1)) >= self.q[edge_index[1]]
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

        return edge_index, edge_attr
    
    def apply_message_mat(self, messages, training=True):

        return super(DropSens, self).apply_message_mat(messages, training)