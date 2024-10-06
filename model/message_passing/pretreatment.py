from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import remove_self_loops, add_self_loops, degree


class ModelPretreatment:
    
    def __init__(self, add_self_loops: bool = True, normalize: bool = True):
        
        self.add_self_loops = add_self_loops
        self.normalize = normalize

    def pretreatment(self, num_nodes: int, edge_index: Adj, dtype):

        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        edge_weight = None
        if self.normalize:
            row, col = edge_index
            deg = degree(col, num_nodes, dtype=dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return edge_index, edge_weight