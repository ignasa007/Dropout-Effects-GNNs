import os
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt

from sensitivity.utils import bin_jac_norms


fig, ax = plt.subplots(1, 1, figsize=(6, 4))
N_SAMPLES = 25

models = (
    ('NoDrop', 'Cora', 'GCN', 6, 0.0),
    # ('NoDrop', 'Cora', 'ResGCN', 6, 0.0),
    # ('DropEdge', 'Cora', 'GCN', 6, 0.1),
    # ('DropNode', 'Cora', 'GCN', 6, 0.5),
    # ('DropAgg', 'Cora', 'GCN', 6, 0.5),
    # ('DropGNN', 'Cora', 'GCN', 6, 0.5),
    # ('Dropout', 'Cora', 'GCN', 6, 0.5),
    # ('DropMessage', 'Cora', 'GCN', 6, 0.5),
    # ('SkipNode', 'Cora', 'GCN', 6, 0.5),
    ('DropSens', 'Cora', 'GCN', 6, 0.5),
)

for dropout, dataset, gnn, L, drop_p in tqdm(models):

    model_dir = f'./jac-norms-copy/{dropout}/{dataset}/{gnn}/L={L}/P={round(drop_p, 6)}'
    count_jac_norms = torch.zeros(1, L+1)
    sum_jac_norms = torch.zeros(N_SAMPLES, L+1)

    for i_dir in os.listdir(model_dir):

        i_dir = f'{model_dir}/{i_dir}'
        shortest_distances = torch.load(f'{i_dir}/shortest_distances.pkl').int()
        x_sd, counts = torch.unique(shortest_distances, return_counts=True)
        count_jac_norms[0, x_sd] += counts     # total number of node pairs at different distances

        for sample in (fn for fn in os.listdir(i_dir) if fn.startswith('sample-')):
            
            jac_norms = torch.load(f'{i_dir}/{sample}')
            y_sd = bin_jac_norms(jac_norms, shortest_distances, x_sd, agg='sum')
            sum_jac_norms[int(sample.lstrip('sample-').rstrip('.pkl'))-1, x_sd] += y_sd

    # average over source nodes in a single large network, or multiple small graphs
    mean_jac_norms = sum_jac_norms / count_jac_norms
    # convert to ratio (cf. influence distribution)
    mean_jac_norms = mean_jac_norms / mean_jac_norms.sum(dim=1, keepdim=True)

    # average over initialization and/or mask samples
    std, mean = torch.std_mean(mean_jac_norms, dim=0)
    x = torch.arange(L+1)

    ax.plot(x, mean, label=f'{gnn}') # , {dropout}, P={drop_p}')
    ax.fill_between(x, mean-2*std, mean+2*std, alpha=0.2)

ax.set_xlabel('Shortest Distances', fontsize=12)
# ax.set_ylabel('Mean Sensitivity', fontsize=12)
ax.set_ylabel('Influence Distribution', fontsize=12)
ax.set_yscale('log')
ax.grid()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor = (0.05, -0.98, 1, 1))
fig.tight_layout()

fn = f'./assets/sensitivity-copy.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')