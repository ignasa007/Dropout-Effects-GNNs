import warnings; warnings.filterwarnings('ignore')
import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
import matplotlib.pyplot as plt

from sensitivity.utils import aggregate

parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, default=6)
parser.add_argument('--drop_p', type=float, default=0.5)
args = parser.parse_args()

dataset = 'Cora'
models = (
    ('NoDrop', 'GCN'),
    ('', ''),
    # ('NoDrop', 'ResGCN'),
    ('DropEdge', 'GCN'),
    ('DropNode', 'GCN'),
    ('Dropout', 'GCN'),
    ('DropAgg', 'GCN'),
    ('DropMessage', 'GCN'),
    ('DropGNN', 'GCN'),
    # ('SkipNode', 'GCN'),
    # ('DropSens', 'GCN'),
)
'''Mean aggregation (instead of sum) because each source-target pair is not equal.
Rather, sensitivity to a source depends on the topolgy from the perspective of the target.'''
agg = 'mean'

jac_norms_dir = './jac-norms'
fig, ax = plt.subplots(1, 1, figsize=(6, 4)); ncol = 4
MODEL_SAMPLES = 25

for dropout, gnn in tqdm(models):

    if not dropout or not dataset or not gnn:
        ax.plot(np.nan, np.nan, '-', color='none', label=' ')
        continue

    dataset_dir = f'{jac_norms_dir}/{dataset}'
    P = 0.0 if dropout == 'NoDrop' else args.drop_p

    count_pairs = torch.zeros(args.L+1)
    sum_norms = torch.zeros(MODEL_SAMPLES, args.L+1)

    for i_dir in os.listdir(dataset_dir):
        
        i_dir = f'{dataset_dir}/{i_dir}/L={args.L}'
        model_dir = f'{i_dir}/{gnn}/{dropout}/P={P}'
        if not os.path.isdir(model_dir) or not os.listdir(model_dir):
            continue

        shortest_distances = torch.load(f'{i_dir}/shortest_distances.pkl').int()
        x_sd, count = torch.unique(shortest_distances, return_counts=True)
        '''Total number of node pairs at different distances'''
        count_pairs[x_sd] += (count if agg == 'sum' else 1)
        
        for sample in range(1, MODEL_SAMPLES+1):
            jac_norms = torch.load(f'{model_dir}/sample={sample}.pkl')
            y_sd = aggregate(jac_norms, shortest_distances, x_sd, agg=agg)
            sum_norms[sample-1, x_sd] += y_sd

    '''Average over source nodes in a single large network, or multiple small graphs'''
    mean_norms = sum_norms/count_pairs

    '''Average over initialization and/or mask samples'''
    std, mean = torch.std_mean(mean_norms, dim=0)
    x = torch.arange(args.L+1)
    ax.plot(x, mean, label=f'{gnn}, {dropout}({P})')
    # ax.fill_between(x, mean-std, mean+std, alpha=0.2)

ax.set_xlabel('Shortest Distances', fontsize=18)
ax.set_ylabel('Mean Sensitivity', fontsize=18)
ax.set_yscale('log')
ax.grid()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=15, ncol=ncol, bbox_to_anchor = (0, -0.15, 1, 1))
fig.tight_layout()

fn = f'./assets/sensitivity.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')