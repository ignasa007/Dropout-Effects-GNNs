import os
import pickle

import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
import torch
from torch_geometric.utils import subgraph

from dataset import get_dataset
from utils.config import parse_arguments
from sensitivity.utils import to_adj_mat, Model, get_jacobian_norms


NODE_SAMPLES = 100
MASK_SAMPLES = 10
INIT_SAMPLES = 5

config, others = parse_arguments(return_others=True)
DEVICE = torch.device(f'cuda:{config.device_index}' if torch.cuda.is_available() and config.device_index is not None else 'cpu')

dataset = get_dataset(config.dataset, config=config, others=others, device=DEVICE)
others.input_dim = dataset.num_features
others.output_dim = dataset.output_dim
others.task_name = dataset.task_name

model = Model(config, others).to(device=DEVICE)

jac_norms_dir = f'./jac-norms/{config.dropout}/{config.dataset}/{config.gnn}/L={len(config.gnn_layer_sizes)}/P={round(config.drop_p, 6)}'
os.makedirs(jac_norms_dir, exist_ok=True)

num_nodes = dataset.x.size(0)
A = to_adj_mat(dataset.edge_index, num_nodes, undirected=True, assert_connected=False)

# sample nodes from the largest component
assignments = connected_components(A, return_labels=True)[1]
cc_labels, sizes = np.unique(assignments, return_counts=True)
# all indices assigned to the largest component
all_indices = np.where(assignments == cc_labels[np.argmax(sizes)])[0]
# indices for which a directory is already created (in case we have incomplete logging for some nodes)
logged_indices = set((int(x.split('=')[1]) for x in os.listdir(jac_norms_dir)))
# set the sample of nodes to be the set of nodes that have been partially logged and the remaining being random nodes
node_samples = np.random.choice(all_indices[~np.isin(all_indices, logged_indices)], NODE_SAMPLES-len(logged_indices), replace=False)
node_samples = logged_indices.union(node_samples)

for i in node_samples:

    shortest_distances = torch.from_numpy(shortest_path(A, method='D', indices=i))
    subset = torch.where(shortest_distances <= len(config.gnn_layer_sizes))[0]

    print(i, subset.size(0))
    i_dir = f'{jac_norms_dir}/i={i}'
    os.makedirs(i_dir, exist_ok=True)
    torch.save(shortest_distances[subset], f'{i_dir}/shortest_distances.pkl')

    edge_index, _ = subgraph(subset, dataset.edge_index, relabel_nodes=True, num_nodes=dataset.x.size(0))
    # checked implementation and relabelling is such that subset[i] is relabelled as i
    x = dataset.x[subset, :]
    new_i = torch.where(subset == i)[0].item()

    for init_sample in range(INIT_SAMPLES):

        save_fn = f'{i_dir}/sample-{init_sample+1}.pkl'
        if os.path.exists(save_fn):
            continue
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        jac_norms = get_jacobian_norms(x, edge_index, new_i, model, MASK_SAMPLES, config, others)
        torch.save(jac_norms, save_fn)