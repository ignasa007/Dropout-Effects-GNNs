import warnings; warnings.filterwarnings('ignore')
import os
from argparse import ArgumentParser

import torch
import matplotlib.pyplot as plt

from over_squashing.utils import aggregate


parser = ArgumentParser()
parser.add_argument('--baseline', type=str, choices=['NoDrop', 'DropEdge'])
args = parser.parse_args()

hyperparams_mapper = {  # For DropSens
    'Cora': 'P=0.8/C=0.8',
    'CiteSeer': 'P=0.8/C=0.8',
    'PubMed': 'P=0.8/C=0.9',
    'Chameleon': 'P=0.5/C=0.8',
    'Squirrel': 'P=0.5/C=0.8',
    'TwitchDE': 'P=0.5/C=0.8',
    'Actor': 'P=0.8/C=0.9',
}

model = 'GCN'
L = 6
agg = 'mean'

jac_norms_dir = './jac-norms'
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8)); ncol = 2
MODEL_SAMPLES = 25


def compute_mean_influence(dataset_dir, dropout, hyperparams):

    count_pairs = torch.zeros(L+1)
    sum_influence = torch.zeros(MODEL_SAMPLES, count_pairs.size(0))

    for i_dir in os.listdir(dataset_dir):

        i_dir = f'{dataset_dir}/{i_dir}/L={L}'
        model_dir = f'{i_dir}/{model}/{dropout}/{hyperparams}'
        if not os.path.isdir(model_dir) or not os.listdir(model_dir):
            continue

        shortest_distances = torch.load(f'{i_dir}/shortest_distances.pkl').int()
        x_sd, count = torch.unique(shortest_distances, return_counts=True)
        count_pairs[x_sd] += 1  # (count if agg == 'sum' else 1)

        for sample in range(1, MODEL_SAMPLES+1):
            jac_norms = torch.load(f'{model_dir}/sample={sample}.pkl')
            if jac_norms.sum().item() > 0.:
                influence_distribution = jac_norms / jac_norms.sum()
                y_sd = aggregate(influence_distribution, shortest_distances, x_sd, agg=agg)
            else:
                y_sd = torch.zeros_like(x_sd)
            sum_influence[sample-1, x_sd] += y_sd

    # Mean of influence of source nodes at different distances from the target
    mean_influence = sum_influence / count_pairs
    # Average over initialization and/or mask samples
    std, mean = torch.std_mean(mean_influence, dim=0)

    return std, mean

mean_base = list()
for dataset, hyperparams in hyperparams_mapper.items():
    dataset_dir = f'{jac_norms_dir}/{dataset}'
    _, mean_base = compute_mean_influence(dataset_dir, dropout=args.baseline, hyperparams=f'P={0.5*(args.baseline != "NoDrop"):.1f}')
    _, mean_drop = compute_mean_influence(dataset_dir, dropout='DropSens', hyperparams=hyperparams)
    ax.plot(torch.arange(L+1), mean_drop/mean_base, label=dataset)
    print(f"| {dataset} | {''.join(map(lambda x: f'{x:.3e} | ', (mean_drop/mean_base).tolist()))}")

ax.hlines(y=1.0, xmin=0., xmax=L, color='black')
ax.set_xticks([], minor=True)
ax.set_xlabel('Shortest Distances', fontsize=18)
ax.set_ylabel(f'Influence, DropSens/{args.baseline}', fontsize=18)
ax.grid()

handles, labels = ax.get_legend_handles_labels()
fig.tight_layout()

fn = f'./assets/influence/DropSens-vs-{args.baseline}.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')