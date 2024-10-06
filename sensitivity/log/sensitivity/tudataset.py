import os
import pickle
from tqdm import tqdm
import argparse

import numpy as np
from torch_geometric.datasets import TUDataset

from dataset.utils import normalize_features
from utils.format import format_dataset_name
from sensitivity.utils import *
from sensitivity.utils.tudataset import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--L', type=int, default=6)
args = parser.parse_args()

NODE_SAMPLES = 100
MODEL_SAMPLES = 10
models_dir = f'./results/sensitivity/model-store/{format_dataset_name[args.dataset.lower()]}'
jac_norms_dir = f'./results/sensitivity/jac-norms-store/{format_dataset_name[args.dataset.lower()]}'
os.makedirs(jac_norms_dir, exist_ok=True)

dataset = TUDataset(root='./data/TUDataset', name=args.dataset, use_node_attr=True)
dataset, = normalize_features(dataset)
num_nodes = np.array([molecule.num_nodes for molecule in dataset])

# indices for which a directory is already created (in case we have incomplete logging for some nodes)
logged_indices = set((int(x.split('=')[1]) for x in os.listdir(jac_norms_dir)))
# index choices for new samples
options = np.where(num_nodes <= 50)[0]
options = options[~np.isin(options, logged_indices)].tolist()

for _ in range(NODE_SAMPLES):

    if logged_indices:
        i = logged_indices.pop()
        molecule = dataset[i]
        shortest_distances = compute_shortest_distances(molecule.edge_index).flatten()
        commute_times = compute_commute_times(molecule.edge_index).flatten()
    else:
        while True:
            i = np.random.choice(options)
            if os.path.isdir(f'{jac_norms_dir}/i={i}'):
                continue
            molecule = dataset[i]
            try:
                shortest_distances = compute_shortest_distances(molecule.edge_index).flatten()
                commute_times = compute_commute_times(molecule.edge_index).flatten()
            except AssertionError:
                continue
            options.remove(i)
            break

    print(i, molecule.num_nodes)
    i_dir = f'{jac_norms_dir}/i={i}'
    os.makedirs(i_dir, exist_ok=True)
    with open(f'{i_dir}/shortest_distances.pkl', 'wb') as f:
        pickle.dump(shortest_distances, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{i_dir}/commute_times.pkl', 'wb') as f:
        pickle.dump(commute_times, f, protocol=pickle.HIGHEST_PROTOCOL)

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
                jac_norms = get_jacobian_norms(molecule, model_dir, MODEL_SAMPLES, trained_fn=='trained')
                with open(save_fn, 'wb') as f:
                    pickle.dump(jac_norms.flatten(), f, protocol=pickle.HIGHEST_PROTOCOL)