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

import numpy as np
import torch
import matplotlib.pyplot as plt
# http://stackoverflow.com/questions/26223937/matplotlib-pandas-zoom-part-of-a-plot-with-time-series/26229669#26229669
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset

from over_squashing.utils import aggregate

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--L', type=int, default=6)
parser.add_argument('--drop_p', type=float, default=0.5)
args = parser.parse_args()

models = (
    ('NoDrop', 'GCN', f'P=0.0'),
    ('DropEdge', 'GCN', f'P={args.drop_p}'),
    ('DropSens', 'GCN', f'P=0.8/C=0.8'),
    # ('Dropout', 'GCN', f'P={args.drop_p}'),
    # ('DropMessage', 'GCN', f'P={args.drop_p}'),
    ('DropNode', 'GCN', f'P={args.drop_p}'),
    ('DropAgg', 'GCN', f'P={args.drop_p}'),
    ('DropGNN', 'GCN', f'P={args.drop_p}'),
)

# Difference between agg='sum' and agg='mean':
# agg='sum' followed by normalization using count_pairs treats each source node equally,
#   computing a weighted mean (over the target nodes, weighted by their in-degrees) of
#   average sensitivity to source nodes at different distances.
# agg='mean' treats each target node equally, computing an unweighted mean of the
#   average sensitivity to source nodes at different distances.
agg = 'mean'

jac_norms_dir = './jac-norms'
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8)); ncol = 2
# axins = zoomed_inset_axes(ax, 0.7, loc=1)
axins = inset_axes(ax, width='32%', height='32%', bbox_to_anchor=(0.065, 0.34, 1, 1), bbox_transform=ax.transAxes, loc='lower left')
MODEL_SAMPLES = 25

for dropout, gnn, hyperparams in tqdm(models):

    if not dropout or not args.dataset or not gnn:
        ax.plot(np.nan, np.nan, '-', color='none', label=' ')
        continue

    dataset_dir = f'{jac_norms_dir}/{args.dataset}'

    # Number of node-pairs at different distances
    #   computed using all target nodes and their corresponding source nodes
    count_pairs = torch.zeros(args.L+1)
    # Sum of influence of source nodes at different distances from the target
    sum_influence = torch.zeros(MODEL_SAMPLES, args.L+1)

    for i_dir in os.listdir(dataset_dir):

        i_dir = f'{dataset_dir}/{i_dir}/L={args.L}'
        model_dir = f'{i_dir}/{gnn}/{dropout}/{hyperparams}'
        # Continue if no model samples exist
        if not os.path.isdir(model_dir) or not os.listdir(model_dir):
            continue

        shortest_distances = torch.load(f'{i_dir}/shortest_distances.pkl').int()
        x_sd, count = torch.unique(shortest_distances, return_counts=True)
        count_pairs[x_sd] += (count if agg == 'sum' else 1)

        for sample in range(1, MODEL_SAMPLES+1):
            jac_norms = torch.load(f'{model_dir}/sample={sample}.pkl')
            # Reasons for computing the influence right away,
            #   instead of eg. summing sensitivities over target nodes first:
            # 1. Graph topology can affect the scale of sensitivity, and we would not
            #   want to treat a well positioned node equal to a poorly positioned node.
            # 2. Magnitude of model parameters can also affect the scale of sensitivities.
            # Essentially, we want scale invariance in order to quantify over-squashing as
            #   sensitivity to distant nodes *after* controlling total sensitivity.
            if jac_norms.sum().item() > 0.:
                influence_distribution = jac_norms / jac_norms.sum()
            else:
                # jac_norms.sum() == 0. in some cases with DropNode
                influence_distribution = torch.zeros_like(jac_norms)
            y_sd = aggregate(influence_distribution, shortest_distances, x_sd, agg=agg)
            # OBSERVATION: For non-edge-dropping methods, influence of neighbors can be less or even
            #   *more* than the self-influence because each neighbor contributes to updating the
            #   target node's representations in each step, just like the target node itself.
            # Their contribution should be related to the L-step random walk transition probability,
            #   P(s_L=i|s_0=j), which would obviously be low for source nodes beyond the neighbors.
            # With edge-dropping methods, the retention of the self-loop makes this probability higher
            #   for self-transitions than for any cross-transitions.
            sum_influence[sample-1, x_sd] += y_sd

    # Mean of influence of source nodes at different distances from the target
    mean_influence = sum_influence / count_pairs

    # Average over initialization and/or mask samples
    std, mean = torch.std_mean(mean_influence, dim=0)
    x = torch.arange(args.L+1)
    ax.plot(x, mean, label=dropout)

    if dropout == 'DropEdge':
        mean_de = mean
    elif dropout == 'DropSens' and 'mean_de' in locals():
        axins.plot(x, mean/mean_de, color='black')
        

ax.set_xticks([], minor=True)
ax.set_xlabel('Shortest Distances', fontsize=18)
ax.set_ylabel('Influence Distribution', fontsize=18)
ax.set_yscale('log')
ax.grid()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', fontsize=15, ncol=ncol, bbox_to_anchor = (0.132, 0.135))
fig.tight_layout()

axins.set_xlim(0, 6)
axins.set_xticks(range(0, 7))
# axins.set_ylim(1.0)
# axins.set_yticks([1e-2, 1e-3, 1e-4, 1e-5, 1e-6], minor=False)
# axins.set_yticks([], minor=True)
axins.set_title('DropSens/DropEdge', fontsize=11.5)
axins.grid()
# plt.xticks(visible=False)
# # plt.yticks(visible=False)
# axins.set_yscale('log', base=10)
# mark_inset(ax, axins, loc1=2, loc2=4)

fn = f'./assets/influence/{args.dataset}_zoom.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')