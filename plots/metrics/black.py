import argparse
import os
import shutil
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

from utils.parse_logs import parse_metrics


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, nargs='+', choices=['Proteins', 'Mutag', 'Enzymes', 'Reddit', 'IMDb', 'Collab'])
parser.add_argument('--gnns', type=str, nargs='+', choices=['GCN', 'GAT'])
parser.add_argument('--dropouts', type=str, nargs='+', choices=['Dropout', 'DropMessage', 'DropEdge'])
args = parser.parse_args()

'''
for name in ('PROTEINS', 'MUTAG', 'ENZYMES', 'REDDIT-BINARY', 'IMDB-BINARY', 'COLLAB'): 
    dataset = TUDataset(root='./data/TUDataset', name=name, use_node_attr=True)
    print(dataset.y.unique(return_counts=True)[1].max().item() / dataset.y.size(0))
'''
cutoffs = {
    'Proteins': 0.5957,
    'Mutag': 0.6649,
    'Enzymes': 0.1667,
    'Reddit': 0.5000,
    'IMDb': 0.5000,
    'Collab': 0.5200,
}

metric = 'Accuracy'
drop_ps = np.round(np.arange(0.1, 1, 0.1), decimals=1)
exp_dir = 'results/{dropout}/{dataset}/{gnn}/L=4/P={drop_p}'


def get_samples(dataset, gnn, dropout, drop_p):

    exp_dir_format = exp_dir.format(dropout=dropout, dataset=dataset, gnn=gnn, drop_p=drop_p)
    samples = list()
    for timestamp in os.listdir(exp_dir_format):
        train, val, test = parse_metrics(f'{exp_dir_format}/{timestamp}/logs')
        if len(test.get(metric, [])) < 300:
            print(f'Incomplete training run: {exp_dir_format}/{timestamp}')
            shutil.rmtree(f'{exp_dir_format}/{timestamp}')
            continue
        # if np.max(train[metric]) < cutoffs[dataset]:
        #     print(f'Failed to learn: {exp_dir_format}/{timestamp}')
        #     shutil.rmtree(f'{exp_dir_format}/{timestamp}')
        #     continue
        sample = test[metric][np.argmax(val[metric])]
        # sample = np.max(train[metric])
        samples.append(sample)

    return samples

def plot(ax, dataset, gnn, dropout):

    means, stds = list(), list()
    
    for drop_p in drop_ps:
        samples = get_samples(dataset, gnn, dropout, drop_p)
        # means.append(np.max(samples))
        means.append(np.mean(samples))
        stds.append(np.std(samples))
    
    means, stds = np.array(means), np.array(stds)
    ax.plot(drop_ps, means, label=dropout)
    # ax.fill_between(drop_ps, means-stds, means+stds, alpha=0.2)


fig, axs = plt.subplots(len(args.datasets), len(args.gnns), figsize=(6.4*len(args.gnns), 4.8*len(args.datasets)))
axs = axs.flatten() if isinstance(axs, np.ndarray) else (axs,)

for i, dataset in enumerate(args.datasets):
    for j, gnn in enumerate(args.gnns):
        ax = axs[i*len(args.gnns)+j]
        nodrop_ylevel = np.mean(get_samples(dataset, gnn, 'NoDrop', 0.0))
        ax.hlines(nodrop_ylevel, drop_ps[0], drop_ps[-1], colors='red', linestyles='--')
        for dropout in args.dropouts:
            plot(ax, dataset, gnn, dropout)
        ax.grid()
        ax.legend()

fig.tight_layout()
fn = f'./assets/black.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')