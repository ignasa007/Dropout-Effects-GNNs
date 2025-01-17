from scipy.sparse.csgraph import connected_components, shortest_path
import torch
from torch.func import jacrev
from torch_geometric.utils import to_undirected, remove_self_loops, to_scipy_sparse_matrix

from model import Model as Base


# TODO: I think functionality broke because A is a sparse matrix now, but we need the dense matrix in some places.
def to_adj_mat(edge_index, num_nodes=None, undirected=True, assert_connected=True):

    if num_nodes is None:
        num_nodes = (edge_index.max()+1).item()
    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    A = to_scipy_sparse_matrix(remove_self_loops(edge_index)[0])    # TODO: should we be removing self loops?
    if assert_connected:
        assert connected_components(A, directed=False, return_labels=False) == 1

    return A

def compute_shortest_distances(edge_index, num_nodes=None, undirected=True, assert_connected=True):

    A = to_adj_mat(edge_index, num_nodes, undirected, assert_connected)
    shortest_distances = torch.from_numpy(shortest_path(A))
    
    return shortest_distances


class Model(Base):
    
    def forward(self, mask, edge_index, x):
    
        for mp_layer in self.message_passing:
            x = mp_layer(x, edge_index)
    
        return x if mask is None else x[mask, ...]    # self.readout(x, mask=mask)

def get_jacobian_norms(x, edge_index, mask, model, n_samples, config, others):

    model.train()

    if mask is None:
        dim0 = x.size(0) 
    elif hasattr(mask, '__len__'):
        dim0 = len(mask)
    else:
        mask = [mask]
        dim0 = 1

    jacobians = torch.zeros((dim0, config.gnn_layer_sizes[-1], x.size(0), others.input_dim))
    n_samples = n_samples if config.drop_p > 0. else 1
    for _ in range(1):
        jacobians += jacrev(model, argnums=2)(mask, edge_index, x)
    jacobians /= n_samples
    jacobian_norms = jacobians.transpose(1, 2).flatten(start_dim=2).norm(dim=2, p=1)

    return jacobian_norms.detach().cpu().flatten()

def bin_jac_norms(jac_norms, bin_assignments, bins, agg='mean'):

    if jac_norms.ndim > 1:
        jac_norms = jac_norms.flatten()
    
    assert jac_norms.size() == bin_assignments.size()

    if agg == 'mean':
        aggregator = torch.mean
    elif agg == 'mean_nz':
        aggregator = lambda members: torch.mean(members[members!=0.])
    elif agg == 'sum':
        aggregator = torch.sum
    else:
        raise ValueError(f"Expected `agg` to be one of 'mean', 'mean_nz' or 'sum'. Instead received '{agg}'.")
    
    aggregated_jac_norms = list()
    for bin in bins:
        bin_members = jac_norms[torch.where(bin_assignments == bin)]
        aggregated_jac_norms.append(aggregator(bin_members))

    return torch.Tensor(aggregated_jac_norms)