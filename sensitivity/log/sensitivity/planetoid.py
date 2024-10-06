import os
import pickle
from tqdm import tqdm
import argparse

import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph

from utils.format import format_dataset_name
from sensitivity.utils import *
from sensitivity.utils.planetoid import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--L', type=int, default=6)
args = parser.parse_args()

NODE_SAMPLES = 100
MODEL_SAMPLES = 10
models_dir = f'./results/sensitivity/model-store/{format_dataset_name[args.dataset.lower()]}'
jac_norms_dir = f'./results/sensitivity/jac-norms-store/{format_dataset_name[args.dataset.lower()]}'
os.makedirs(jac_norms_dir, exist_ok=True)

dataset = Planetoid(root='./data/Planetoid', name=format_dataset_name[args.dataset.lower()], split='full')
num_nodes = dataset.x.size(0)
A = to_scipy_sparse_matrix(dataset.edge_index)
commute_times = compute_commute_times(dataset.edge_index, assert_connected=False)

# sample nodes from the largest component
assignments = connected_components(A, return_labels=True)[1]
cc_labels, sizes = np.unique(assignments, return_counts=True)
# all indices assigned to the largest component
all_indices = np.where(assignments == cc_labels[np.argmax(sizes)])[0]
# indices for which a directory is already created (in case we have incomplete logging for some nodes)
logged_indices = set((int(x.split('=')[1]) for x in os.listdir(jac_norms_dir)))
# set the sample of nodes to be the set of nodes that have been partially logged and the remaining being random nodes
sample = np.random.choice(all_indices[~np.isin(all_indices, logged_indices)], NODE_SAMPLES-len(logged_indices), replace=False)
sample = logged_indices.union(sample)

for i in sample:

    shortest_distances = torch.from_numpy(shortest_path(A, method='D', indices=i))
    subset = torch.where(shortest_distances <= args.L)[0]

    print(i, subset.size(0))
    i_dir = f'{jac_norms_dir}/i={i}'
    os.makedirs(i_dir, exist_ok=True)
    with open(f'{i_dir}/shortest_distances.pkl', 'wb') as f:
        pickle.dump(shortest_distances[subset], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{i_dir}/commute_times.pkl', 'wb') as f:
        pickle.dump(commute_times[i, subset], f, protocol=pickle.HIGHEST_PROTOCOL)

    edge_index, _ = subgraph(subset, dataset.edge_index, relabel_nodes=True, num_nodes=dataset.x.size(0))
    # checked implementation and relabelling is such that subset[i] is relabelled as i
    x = dataset.x[subset, :]
    new_i = torch.where(subset == i)[0].item()

    for P_dir in tqdm(os.listdir(models_dir)):
        P = float(P_dir.split('=')[1])
        P_dir = f'{models_dir}/{P_dir}'
        for timestamp in os.listdir(P_dir):
            model_dir = f'{P_dir}/{timestamp}'
            for trained_fn in ('trained', 'untrained'):
                save_fn = f'{i_dir}/P={P}/{timestamp}/{trained_fn}.pkl'
                if os.path.exists(save_fn):
                    continue
                os.makedirs(os.path.dirname(save_fn), exist_ok=True)
                jac_norms = get_jacobian_norms(x, edge_index, new_i, model_dir, MODEL_SAMPLES, trained_fn=='trained')
                with open(save_fn, 'wb') as f:
                    pickle.dump(jac_norms.flatten(), f, protocol=pickle.HIGHEST_PROTOCOL)