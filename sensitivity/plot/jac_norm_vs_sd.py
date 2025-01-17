import os
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt

from sensitivity.utils import bin_jac_norms


jac_norms_dir = './jac-norms'
fig, ax = plt.subplots(1, 1, figsize=(6, 4)); ncol = 4
N_SAMPLES = 25

models = (
    ('NoDrop', 'Cora', 'GCN', 6, 0.0),
    ('NoDrop', 'Cora', 'ResGCN', 6, 0.0),
    # ('DropEdge', 'Cora', 'GCN', 6, 0.5),
    # ('DropEdge', 'Cora', 'GCN', 6, 0.1),
    # ('DropNode', 'Cora', 'GCN', 6, 0.5),
    # ('DropAgg', 'Cora', 'GCN', 6, 0.5),
    # ('DropGNN', 'Cora', 'GCN', 6, 0.5),
    # ('Dropout', 'Cora', 'GCN', 6, 0.5),
    # ('DropMessage', 'Cora', 'GCN', 6, 0.5),
    # ('SkipNode', 'Cora', 'GCN', 6, 0.5),
    # ('DropSens', 'Cora', 'GCN', 6, 0.9),
)

for dropout, dataset, gnn, L, drop_p in tqdm(models):

    count_jac_norms = torch.zeros(L+1)
    sum_jac_norms = torch.zeros(N_SAMPLES, L+1)

    for i_dir in os.listdir(jac_norms_dir):
        
        i_dir = f'{jac_norms_dir}/{i_dir}'
        model_dir = f'{i_dir}/{dropout}-{gnn}'
        if not os.path.isdir(model_dir):
            continue

        shortest_distances = torch.load(f'{i_dir}/shortest_distances.pkl').int()
        x_sd, counts = torch.unique(shortest_distances, return_counts=True)
        count_jac_norms[x_sd] += counts     # total number of node pairs at different distances

        for sample in (fn for fn in os.listdir(model_dir) if fn.startswith('sample-')):
            
            jac_norms = torch.load(f'{model_dir}/{sample}')
            y_sd = bin_jac_norms(jac_norms, shortest_distances, x_sd, agg='sum')
            sum_jac_norms[int(sample.removeprefix('sample-').removesuffix('.pkl'))-1, x_sd] += y_sd

    # average over source nodes in a single large network, or multiple small graphs
    mean_jac_norms = sum_jac_norms / count_jac_norms
    # convert to ratio (cf. influence distribution)
    mean_jac_norms = mean_jac_norms / mean_jac_norms.sum(dim=1, keepdim=True)
    # average over initialization and/or mask samples
    std, mean = torch.std_mean(mean_jac_norms, dim=0)
    
    x = torch.arange(L+1)
    ax.plot(x, mean, label=f'{dropout}-{gnn}')
    ax.fill_between(x, mean-std, mean+std, alpha=0.2)

ax.set_xlabel('Shortest Distances', fontsize=18)
ax.set_ylabel('Influence Distribution', fontsize=18)
ax.set_yscale('log')
ax.grid()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=15, ncol=ncol, bbox_to_anchor = (0, -0.15, 1, 1))
fig.tight_layout()

fn = f'./assets/sensitivity.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')