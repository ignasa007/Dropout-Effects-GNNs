import argparse
import os
import warnings; warnings.filterwarnings('ignore')
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from utils.parse_logs import parse_metrics


parser = argparse.ArgumentParser()
parser.add_argument('--gnn', type=str, default='GCN', choices=['GCN', 'GAT'])
parser.add_argument('--dropout', type=str, default='DropEdge')
parser.add_argument('--train', action='store_true')
args = parser.parse_args()

metric = 'Accuracy'
if args.train:
    get_value = lambda train, val, test: np.max(train[metric])
    ylabel = 'Train'
else:
    get_value = lambda train, val, test: np.max(test[metric][np.argmax(val[metric])])
    ylabel = 'Test'

homo_data = ('Cora', 'CiteSeer', 'PubMed')
hetero_data = ('Chameleon', 'Squirrel', 'TwitchDE')
depths = range(2, 9, 2)
ps = np.round(np.arange(0.1, 1, 0.1), decimals=1)
ncol = np.ceil(len(depths)/1)

for datasets, fn in zip((homo_data, hetero_data), ('homophilic', 'heterophilic')):

    fig, axs = plt.subplots(1, len(datasets), figsize=(6.4*len(datasets), 4.8))
    if not hasattr(axs, '__len__'): axs = (axs,)

    axs[0].set_ylabel(f'{ylabel} {metric}', fontsize=20)

    for dataset, ax in tqdm(zip(datasets, axs)):

        assets_dir = f"./assets/philia/{args.gnn}/{args.dropout}/{ylabel}"
        exp_dir = f'./results/{dataset}/{args.gnn}/' + 'L={depth}/' + f'{args.dropout}/' + 'P={p}'

        ### RETRIEVE METRICS ###

        metrics = defaultdict(list)

        for depth in depths:
            for drop_p in ps:
                exp_dir_format = exp_dir.format(depth=depth, p=drop_p)
                if not os.path.exists(exp_dir_format):
                    continue
                for sample_dir in os.listdir(exp_dir_format):
                    train, val, test = parse_metrics(f'{exp_dir_format}/{sample_dir}/logs')
                    if len(train[metric]) != 300:
                        continue
                    value = get_value(train, val, test)
                    metrics[(depth, drop_p)].append(value)

        ### PLOT FOR TEST SET ###

        metrics = {exp: (np.mean(samples[:20]), np.std(samples[:20])) for exp, samples in metrics.items()}

        for depth in depths:
            drop_ps, means, stds = list(), list(), list()
            for drop_p in ps:
                if (depth, drop_p) in metrics:
                    mean, std = metrics[(depth, drop_p)]
                    drop_ps.append(drop_p); means.append(mean); stds.append(std)
            # ax.plot(drop_ps, means, label=f'L = {depth}')
            ax.errorbar(drop_ps, means, yerr=stds, capsize=3, fmt='o--', ecolor='black', label=f'L = {depth}')
        ax.set_xlabel(fr'{args.dropout} Probability, $q$', fontsize=20)
        ax.set_title(dataset, fontsize=20)
        ax.grid()

    if fn == 'heterophilic':
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', fontsize=18, ncol=ncol, bbox_to_anchor = (0.5, -0.02))
    fig.tight_layout()
    fn = f'./{assets_dir}/{fn}.png'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn, bbox_inches='tight')