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
    A = torch.full((num_nodes, num_nodes), 0., device=edge_index.device)

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
    beta = torch.sum(degrees / (1-P**degrees))
    C = beta * (L_pinv_diag.unsqueeze(0) + L_pinv_diag.unsqueeze(1) - 2*L_pinv)

    return C

def get_jacobian_norms(x, edge_index, model, config, mask=None, n_samples=1):

    if mask is None:
        # compute full Jacobian, wrt all nodes' representations -- can be impractical 
        dim0 = x.size(0)
    elif hasattr(mask, '__len__'):
        dim0 = len(mask)
    else:
        # convert to a list so that the dimensions are preserved when taking a subset of node representations
        # otherwise, out[mask, :] will be of size (output_dim,) instead of the desired size (1, output_dim)
        mask = [mask]
        dim0 = 1

    # If all dropout_prob == 0., then don't need to average over samples
    #   I don't think this should be handled inside the method, but rather before calling it
    # n_samples = n_samples if any((mp_layer.drop_strategy.dropout_prob>0. for mp_layer in model.message_passing)) else 1

    model.train()
    jacobians = torch.zeros((dim0, model.message_passing[-1].out_channels, x.size(0), x.size(1)), device=x.device)
    for _ in range(n_samples):
        # forward mode auto-diff is more efficient for computing derivates wrt more outputs than inputs
        #   - not our use case, since the set of source nodes will usually be much larger than the set of targets
        # reverse mode is more efficient when we have more inputs than outputs
        jacobians += jacrev(model, argnums=2)(mask, edge_index, x)
    jacobians /= n_samples
    # take 1-norm over the input and output feature dimensions
    jacobian_norms = jacobians.transpose(1, 2).flatten(start_dim=2).norm(dim=2, p=1)

    return jacobian_norms.detach().cpu()

def aggregate(values, bin_assignments, bins, agg='mean'):

    if values.ndim > 1:
        values = values.flatten()
    
    assert values.size() == bin_assignments.size()

    if agg == 'mean':
        aggregator = torch.mean
    elif agg == 'mean_nz':
        aggregator = lambda members: torch.mean(members[members!=0.])
    elif agg == 'sum':
        aggregator = torch.sum
    else:
        raise ValueError(f"Expected `agg` to be one of 'mean', 'mean_nz' or 'sum'. Instead received '{agg}'.")
    
    '''
    TODO: can be parallelized using broadcasting
    Below is a memory inefficient way:
        inclusion = bin_assignments.reshape(1, -1) == bins.reshape(-1, 1)   # shape (n_bins, n_sources)
        members = values[inclusion]                                         # shape (n_bins, n_sources)
        aggregated_members = aggregator(members, dim=1)
    '''
    aggregated_values = list()
    for bin in bins:
        bin_members = values[torch.where(bin_assignments == bin)]
        aggregated_values.append(aggregator(bin_members))

    return torch.Tensor(aggregated_values)