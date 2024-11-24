from torch import Tensor
from torch_geometric.typing import Adj

from model.message_passing.gcn import GCNLayer


class ResGCNLayer(GCNLayer):

    def forward(self, x: Tensor, edge_index: Adj):

        out = super(ResGCNLayer, self).forward(x, edge_index)
        
        return out+x if self.in_channels == self.out_channels else out