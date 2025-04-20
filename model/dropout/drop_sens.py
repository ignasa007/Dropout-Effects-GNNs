from typing import Optional
from argparse import Namespace
import warnings

import sympy
from sympy.abc import x
import torch
from torch_geometric.utils import degree, contains_self_loops
from model.dropout.base import BaseDropout


class DropSens(BaseDropout):

    def __init__(self, dropout_prob: float = 0.5, others: Optional[Namespace] = None):

        # Only calling BaseDropout.__init__() because it calls nn.Module.__init__()
        # Cannot directly inherit nn.Module because need default implementations of
        #     apply_feature_mat() and apply_message_mat()

        super(DropSens, self).__init__(dropout_prob)    # Maximum value q_i can take
        self.c = others.info_loss_ratio
        self.node_level_task = others.task_name.lower().startswith('node')

    def init_mapper(self, edge_index):

        # Assuming edge index does not have self loops
        if contains_self_loops(edge_index):
            warnings.warn('Degree computation in DropSens assumes absence of self-loops, \
                but the edge_index received has them.')
        degrees = degree(edge_index[1]).int()       # Node index -> node degree

        if self.node_level_task:
            # If node level task, compute mapper *once*, only for the valid degrees, 
            # since they won't change in each run
            ds = torch.unique(degrees).tolist()             # Sorted array
            self.mapper = torch.nan * torch.ones(ds[-1]+1)  # torch.nan where index is not in ds
            self.mapper[ds] = self.dropout_prob             # Reset some of the dropping probabilities below
        else:
            # If graph level task, compute mapper for all d upto max(unique_degrees), 
            # because degrees will change in each run
            ds = range(0, degrees.max().item()+1)
            self.mapper = self.dropout_prob * torch.ones(ds[-1]+1)
        
        for d in ds:
            q = float(sympy.N(sympy.real_roots(d*(1-self.c)*(1-x)-x+x**(d+1))[-2])) if d>0 else 0.
            if q > self.dropout_prob:
                break   # Because q monontonic wrt d, and unique_degrees is sorted
            self.mapper[d] = q

    def update_mapper(self, degrees):

        # If q has been computed for all degrees, simply return 
        if degrees.max().item() < self.mapper.size(0):
            return

        # Set max dropping probability for d for which q was not computed before
        ds = range(len(self.mapper), degrees.max().item()+1)
        self.mapper = torch.cat((self.mapper, self.dropout_prob*torch.ones(len(ds))))
        
        # If q had already been maxed out, no need to compute even once
        if self.mapper[ds[0]-1] == self.dropout_prob:
            return
        
        for d in ds:
            q = float(sympy.N(sympy.real_roots(d*(1-self.c)*(1-x)-x+x**(d+1))[-2])) if d>0 else 0.
            # Because q monontonic wrt d, and unique_degrees is sorted
            if q > self.dropout_prob: break
            self.mapper[d] = q

    def apply_feature_mat(self, x, training=True):

        return super(DropSens, self).apply_feature_mat(x, training)
    
    def apply_adj_mat(self, edge_index, edge_attr=None, training=True):

        if not training or self.dropout_prob == 0.0:
            return edge_index, edge_attr

        degrees = degree(edge_index[1]).int()
        if not hasattr(self, 'mapper'):
            self.init_mapper(edge_index)
        elif not self.node_level_task:
            self.update_mapper(degrees)

        # degrees[edge_index[1]]: i -> d_i
        # self.mapper[degrees[edge_index[1]]]: (i -> d_i) -> q_i
        qs = self.mapper[degrees[edge_index[1]].to('cpu')]
        edge_mask = qs <= torch.rand(edge_index.size(1))
        edge_mask = edge_mask.to(edge_index.device)

        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

        return edge_index, edge_attr
    
    def apply_message_mat(self, messages, training=True):

        return super(DropSens, self).apply_message_mat(messages, training)