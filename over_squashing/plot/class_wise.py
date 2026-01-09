'''
Average over sampled node-pairs in a single large network, or multiple small graphs
PROBLEM: using sensitivity instead of influence distribution can give a distorted view of over-squashing
    Say you have two nodes, `i` and `j`, with `m` and `n` neighbors, respectively.
    z_i = 1/(m+1) Σ_{j∈N(i)} Wx_j
    => |dz_i/dx_j| = c/(m+1), i.e. equal sensitivity to self and neighbors
    => arguably, no over-squashing
    sum_jac_norms ∝ [1/(m+1) + 1/(n+1), m/(m+1) + n/(n+1)]
    count_jac_norms = [1 + 1, m + n]
    => mean_jac_norms ∝ [(1/(m+1)+1/(n+1)) / 2, (m/(m+1)+n/(n+1)) / (m+n)]
    => mean_jac_norms[0] > mean_jac_norms[1], which is not a quantitative representation
    Can similarly extend this argument to more number of nodes, just need the inequality m/n+n/m > 2
SOLUTION: using influence distribution removes the dependence on normalization
    The rest is kept the same: compute a weighted average of influence at each distance,
    weights implicitly being the number of neighbors, eg. `m` and `n` above.
'''

import warnings; warnings.filterwarnings('ignore')
import os
import argparse
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt

from over_squashing.utils import aggregate

parser = argparse.ArgumentParser()
parser.add_argument('--dropout', type=str, default='DropEdge')
args = parser.parse_args()

model = 'GCN'
L = 6
hyperparams_mapper = {  # For DropSens
    'Cora': 'P=0.8/C=0.8',
    'CiteSeer': 'P=0.8/C=0.8',
    'PubMed': 'P=0.8/C=0.9',
    'Chameleon': 'P=0.5/C=0.8',
    'Squirrel': 'P=0.5/C=0.8',
    # 'TwitchDE': 'P=0.5/C=0.8',
    'Actor': 'P=0.8/C=0.9',
}
if args.dropout != 'DropSens':
    hyperparams_mapper = {k: 'P=0.5' for k in hyperparams_mapper}

agg = 'mean'
jac_norms_dir = './jac-norms'
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8)); ncol = 2
MODEL_SAMPLES = 25


def compute_mean_influence(dataset_dir, dropout, hyperparams):

    count_pairs = torch.zeros(L+1)
    sum_influence = {
        kind: torch.zeros(MODEL_SAMPLES, count_pairs.size(0))
        for kind in ('same', 'diff')
    }
    # overall_sum_influence = torch.zeros(MODEL_SAMPLES, count_pairs.size(0))
    
    for i_dir in os.listdir(dataset_dir):

        i_dir = f'{dataset_dir}/{i_dir}/L={L}'
        model_dir = f'{i_dir}/{model}/{dropout}/{hyperparams}'
        if not os.path.isdir(model_dir) or not os.listdir(model_dir):
            continue
        
        shortest_distances = torch.load(f'{i_dir}/shortest_distances.pkl').int()
        i = torch.where(shortest_distances == 0.)[0].item()
        labels = torch.load(f'{i_dir}/labels.pkl')
        same_label_indices = torch.where(labels == labels[i])
        diff_label_indices = torch.where(labels != labels[i])

        x_sd, count = torch.unique(shortest_distances, return_counts=True)
        count_pairs[x_sd] += 1

        for sample in range(1, MODEL_SAMPLES+1):
            jac_norms = torch.load(f'{model_dir}/sample={sample}.pkl')
            if jac_norms.sum().item() > 0.:
                influence_distribution = jac_norms / jac_norms.sum()
                y_sd = aggregate(influence_distribution, shortest_distances, x_sd, agg='sum')
                # overall_sum_influence[sample-1, x_sd] += y_sd
                for kind, indices in zip(('same', 'diff'), (same_label_indices, diff_label_indices)):
                    y_sd = aggregate(influence_distribution[indices], shortest_distances[indices], x_sd, agg=agg)
                    y_sd = torch.where(y_sd.isnan(), 0., y_sd)
                    sum_influence[kind][sample-1, x_sd] += y_sd

    mean_influence = {
        kind: sum_influence[kind] / count_pairs
        for kind in sum_influence
    }

    # overall_mean_influnce = torch.mean(overall_sum_influence/count_pairs, dim=0)
    # print(overall_mean_influnce, overall_mean_influnce.sum())

    return {
        kind: torch.std_mean(mean_influence[kind], dim=0)
        for kind in mean_influence
    }

for dataset, hyperparams in tqdm(hyperparams_mapper.items()):

    dataset_dir = f'{jac_norms_dir}/{dataset}'
    mean_influences = compute_mean_influence(dataset_dir, dropout=args.dropout, hyperparams=hyperparams_mapper[dataset])
    ax.plot(torch.arange(1, L+1), mean_influences['same'][1][1:]/mean_influences['diff'][1][1:], label=dataset)
    # p = ax.plot(torch.arange(L+1), mean_influences['same'][1], label=dataset)
    # ax.plot(torch.arange(1, L+1), mean_influences['diff'][1][1:], color=p[-1].get_color(), linestyle='--')
    # print(f"| {dataset} | {''.join(map(lambda x: f'{x:.3e} | ', (mean_drop/mean_base).tolist()))}")


ax.set_xticks([], minor=True)
ax.set_xlabel('Shortest Distances', fontsize=18)
ax.set_ylabel(f'Same vs Diff Label Nodes', fontsize=18)
# ax.set_yscale('log')
ax.grid()

ax.legend()
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower left', fontsize=15, ncol=ncol, bbox_to_anchor = (0.132, 0.135))
fig.tight_layout()

fn = f'./assets/influence/class_wise_{args.dropout}_{agg}.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')