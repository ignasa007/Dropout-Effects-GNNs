from scipy.sparse.csgraph import connected_components, shortest_path
import torch
from torch.func import jacrev
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import degree, is_undirected, to_undirected, \
    remove_self_loops, to_scipy_sparse_matrix


def is_connected(edge_index):
    
    return connected_components(to_scipy_sparse_matrix(edge_index), directed=False, return_labels=False) == 1

def to_adj_mat(edge_index, num_nodes=None, undirected=False):
    
    num_nodes = num_nodes if isinstance(num_nodes, int) else maybe_num_nodes(edge_index)
    A = torch.full((num_nodes, num_nodes), 0.)

    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = edge_index.type(torch.int64)
    A[edge_index[0], edge_index[1]] = 1.
    
    return A

def compute_shortest_distances(edge_index, num_nodes=None, undirected=True):
    # TODO: check that all function calls expect tensor output and not array
    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    
    return torch.from_numpy(shortest_path(to_scipy_sparse_matrix(edge_index)))

def compute_commute_times(edge_index, P=0.):
    
    assert is_undirected(edge_index)
    assert is_connected(edge_index)
    
    # Can alternatively add remaining self loops, since D-A remains unchanged
    edge_index = remove_self_loops(edge_index)[0]
    A = to_adj_mat(edge_index, undirected=True)

    degrees = degree(edge_index[1])
    L = torch.diag(degrees) - A
    
    # Can also use torch.linalg.pinv(L+1/A.size(0))) -- I didn't see any diff for simple test cases
    L_pinv = torch.linalg.pinv(L)
    L_pinv_diag = torch.diag(L_pinv)
    beta = torch.sum(degrees / (1 - P**degrees))
    C = beta * (L_pinv_diag.unsqueeze(0) + L_pinv_diag.unsqueeze(1) - 2*L_pinv)

    return C

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