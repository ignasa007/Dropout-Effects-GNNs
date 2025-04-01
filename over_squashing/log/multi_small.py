import os
import pickle

import numpy as np
import torch

from dataset import get_dataset
from utils.config import parse_arguments
from over_squashing.utils import compute_shortest_distances, Model, get_jacobian_norms


GRAPH_SAMPLES = 100
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

# TODO: dataset is custom object, not iterable, causing trouble
num_nodes = np.array([datum.num_nodes for datum in dataset])

# indices for which a directory is already created (in case we have incomplete logging for some nodes)
logged_indices = set((int(x.split('=')[1]) for x in os.listdir(jac_norms_dir)))
# index choices for new samples
options = np.where(num_nodes <= 50)[0]
options = options[~np.isin(options, logged_indices)].tolist()

for _ in range(GRAPH_SAMPLES):

    if logged_indices:
        i = logged_indices.pop()
        datum = dataset[i]
        shortest_distances = compute_shortest_distances(datum.edge_index).flatten()
    else:
        while True:
            i = np.random.choice(options)
            if os.path.isdir(f'{jac_norms_dir}/i={i}'):
                continue
            datum = dataset[i]
            try:
                shortest_distances = compute_shortest_distances(datum.edge_index).flatten()
            except AssertionError:
                continue
            options.remove(i)
            break

    print(i, datum.num_nodes)
    i_dir = f'{jac_norms_dir}/i={i}'
    os.makedirs(i_dir, exist_ok=True)
    torch.save(shortest_distances, f'{i_dir}/shortest_distances.pkl')

    for init_sample in range(INIT_SAMPLES):

        save_fn = f'{i_dir}/sample-{init_sample+1}.pkl'
        if os.path.exists(save_fn):
            continue
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        jac_norms = get_jacobian_norms(datum.x, datum.edge_index, None, model, MASK_SAMPLES, config, others)
        torch.save(jac_norms, save_fn)