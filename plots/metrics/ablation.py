import argparse
import os
import warnings; warnings.filterwarnings('ignore')
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from utils.parse_logs import parse_metrics, parse_configs


parser = argparse.ArgumentParser()
parser.add_argument('--gnn', type=str, default='GCN', choices=['GCN', 'GAT'])
parser.add_argument('--dropout', type=str, default='DropEdge')
args = parser.parse_args()

hetero_data = ('Chameleon', 'Squirrel', 'TwitchDE')
hetero_cutoffs = (0.2288, 0.1203, 0.6045)

metric = 'Accuracy'
depths = range(2, 9, 2)
ps = np.round(np.arange(0.0, 1, 0.1), decimals=1)

ncol = np.ceil(len(depths)/1)

fig, axs = plt.subplots(1, len(hetero_data), figsize=(6.4*len(hetero_data), 4.8))
if not hasattr(axs, '__len__'): axs = (axs,)

axs[0].set_ylabel(f'Train {metric}', fontsize=20)

for dataset, cutoff, ax in zip(hetero_data, hetero_cutoffs, axs):

    results_dir = f'./results/{args.dropout}/{dataset}'
    exp_dir = results_dir + '/{args.gnn}/L={depth}/P={p}'

    ### RETRIEVE METRICS ###

    train_metrics = defaultdict(list)

    for depth in depths:
        for p in ps:
            exp_dir_format = exp_dir.format(gnn=args.gnn, depth=depth, p=p)
            if not os.path.exists(exp_dir_format):
                continue
            for sample_dir in os.listdir(exp_dir_format):
                config = parse_configs(f'{exp_dir_format}/{sample_dir}/logs')
                if not all([x == 64 for x in eval(config['gnn_layer_sizes'])]):
                    print(f'{exp_dir_format}/{sample_dir}')
                train, val, test = parse_metrics(f'{exp_dir_format}/{sample_dir}/logs')
                if len(train[metric]) != 300:
                    continue
                value = np.max(train[metric])
                if value > cutoff:
                    train_metrics[(depth, p)].append(value)
                
    for depth in depths:
        for p in ps:
            if len(train_metrics[(depth, p)]) != 5:
                print(dataset, args.gnn, depth, p, len(train_metrics[(depth, p)]))

    ### PLOT FOR TEST SET ###

    train_metrics = {exp: (np.mean(samples[:5]), np.std(samples[:5])) for exp, samples in train_metrics.items()}

    for depth in depths:
        drop_ps, means, lower, upper = list(), list(), list(), list()
        for drop_p in ps:
            if (depth, drop_p) in train_metrics:
                mean, std = train_metrics[(depth, drop_p)]
                drop_ps.append(drop_p); means.append(mean); lower.append(mean-std); upper.append(mean+std)
        ax.plot(drop_ps, means, label=f'L = {depth}')
    ax.set_xlabel(fr'{args.dropout} Probability, $q$', fontsize=20)
    ax.set_title(dataset, fontsize=20)
    ax.grid()

if len(hetero_data) == 3:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=18, ncol=ncol, bbox_to_anchor = (0.5, -0.02))
fig.tight_layout()
fn = f'./assets/{args.dropout}/ablation.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')