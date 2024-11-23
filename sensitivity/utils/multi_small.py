import torch
from torch.func import jacrev


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
    for _ in range(n_samples):
        jacobians += jacrev(model, argnums=2)(mask, edge_index, x)
    jacobians /= n_samples
    jacobian_norms = jacobians.transpose(1, 2).flatten(start_dim=2).norm(dim=2, p=1)

    return jacobian_norms