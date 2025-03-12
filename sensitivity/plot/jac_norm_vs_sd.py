import warnings; warnings.filterwarnings('ignore')
import os
import argparse
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sensitivity.utils import aggregate


parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, default=6)
parser.add_argument('--drop_p', type=float, default=0.5)
args = parser.parse_args()

models = (
    ('NoDrop', 'Cora', 'GCN'),
    # ('NoDrop', 'Cora', 'ResGCN'),
    # ('DropEdge', 'Cora', 'GCN'),
    # ('DropNode', 'Cora', 'GCN'),
    # ('DropAgg', 'Cora', 'GCN'),
    # ('DropGNN', 'Cora', 'GCN'),
    ('Dropout', 'Cora', 'GCN'),
    ('DropMessage', 'Cora', 'GCN'),
    # ('SkipNode', 'Cora', 'GCN'),
    # ('DropSens', 'Cora', 'GCN'),
)

jac_norms_dir = './tmp_jac-norms'
fig, ax = plt.subplots(1, 1, figsize=(6, 4)); ncol = 4
MODEL_SAMPLES = 25

for dropout, dataset, gnn in tqdm(models):

    dataset_dir = f'{jac_norms_dir}/{dataset}'
    P = 0.0 if dropout == 'NoDrop' else args.drop_p
    
    # Number of node-pairs at different distances
    #   computed using all target nodes and their corresponding source nodes
    count_jac_norms = torch.zeros(args.L+1)
    # Sum of sensitivities between node-pairs at different distances
    sum_jac_norms = torch.zeros(MODEL_SAMPLES, args.L+1)

    for i_dir in os.listdir(dataset_dir):
        
        i_dir = f'{dataset_dir}/{i_dir}/L={args.L}'
        model_dir = f'{i_dir}/{gnn}/{dropout}/P={P}'
        if not os.path.isdir(model_dir):
            continue

        shortest_distances = torch.load(f'{i_dir}/shortest_distances.pkl').int()
        x_sd, counts = torch.unique(shortest_distances, return_counts=True)
        count_jac_norms[x_sd] += counts     # total number of node pairs at different distances

        for sample in range(1, MODEL_SAMPLES+1):
            jac_norms = torch.load(f'{model_dir}/sample={sample}.pkl')
            y_sd = aggregate(jac_norms, shortest_distances, x_sd, agg='sum')
            sum_jac_norms[sample-1, x_sd] += y_sd

    # Average over sampled node-pairs in a single large network, or multiple small graphs
    mean_jac_norms = sum_jac_norms / count_jac_norms
    # Average over initialization and/or mask samples
    std, mean = torch.std_mean(mean_jac_norms, dim=0)
    
    x = torch.arange(args.L+1)
    ax.plot(x, mean, label=f'{gnn}, {dropout}({P})')
    # ax.fill_between(x, mean-std, mean+std, alpha=0.2)

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