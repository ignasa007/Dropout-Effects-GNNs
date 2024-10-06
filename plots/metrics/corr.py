import os
from collections import defaultdict
import argparse
from tqdm import tqdm

import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from utils.parse_logs import parse_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['Cora', 'CiteSeer', 'Proteins', 'MUTAG', 'PTC'])
parser.add_argument('--metric', type=str, required=True, choices=['Cross Entropy Loss', 'Accuracy', 'F1 Score'])
parser.add_argument('--gnns', nargs='+', default=('GCN',))
parser.add_argument('--which', type=str, default=['Best'], choices=['Best', 'Final'])
parser.add_argument('--min_depth', type=int, default=2)
parser.add_argument('--max_depth', type=int, default=8)
args = parser.parse_args()

depths = range(args.min_depth, args.max_depth+1)
ncol = np.ceil(len(depths)/1)
ps = np.round(np.arange(0.1, 1, 0.1), decimals=1)

results_dir = f'./results/drop-edge'
exp_dir = f'{results_dir}/{args.dataset}' + '/{gnn}/L={depth}/P={p}'
# results_dir = f'./results/sensitivity/model-store/{args.dataset}'
# exp_dir = results_dir + '/P={p}'
assets_dir = results_dir.replace('results', 'assets') + f'/corr/{args.dataset}'

### RETRIEVE METRICS ###

train_metrics = defaultdict(list)
test_metrics = defaultdict(list)
gap_metrics = defaultdict(list)

for gnn in args.gnns:
    for depth in tqdm(depths):
        for p in ps:
            exp_dir_format = exp_dir.format(gnn=gnn, depth=depth, p=p)
            for sample_dir in os.listdir(exp_dir_format):
                train, val, test = parse_metrics(f'{exp_dir_format}/{sample_dir}/logs')
                if max(train[args.metric]) < 0.5:
                    continue
                if args.which == 'Best':
                    train_metrics[(gnn, depth, p)].append(max(train[args.metric]))
                    test_metrics[(gnn, depth, p)].append(test[args.metric][np.argmax(val[args.metric])])
                elif args.which == 'Final':
                    train_metrics[(gnn, depth, p)].append(train[args.metric][-1])
                    test_metrics[(gnn, depth, p)].append(test[args.metric][-1])
                gap_metrics[(gnn, depth, p)].append(train_metrics[(gnn, depth, p)][-1]-test_metrics[(gnn, depth, p)][-1])

### PLOT FOR TRAIN SET ###

fig, axs = plt.subplots(1, len(args.gnns), figsize=(6.4*len(args.gnns), 4.8))
if not hasattr(axs, '__len__'): axs = (axs,)

for gnn, ax in zip(args.gnns, axs):
    corr = list()
    for depth in depths:
        xs, ys = list(), list()
        for drop_p in ps:
            metrics = train_metrics.get((gnn, depth, drop_p), list())
            xs.extend(metrics); ys.extend([drop_p]*len(metrics))
        corr.append(spearmanr(xs, ys).statistic)
    ax.plot(depths, corr)
    ax.set_xlabel('Model Depth', fontsize=14)
    ax.set_ylabel(f'Corr with Training {args.metric}', fontsize=14)
    ax.set_title(gnn, fontsize=14)
    ax.grid()

fig.suptitle(args.dataset, fontsize=16)
fig.tight_layout()
fn = f'./{assets_dir}/{args.which}/{args.metric}/train.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')

### PLOT FOR TEST SET ###

fig, axs = plt.subplots(1, len(args.gnns), figsize=(6.4*len(args.gnns), 4.8))
if not hasattr(axs, '__len__'): axs = (axs,)

for gnn, ax in zip(args.gnns, axs):
    corr = list()
    for depth in depths:
        xs, ys = list(), list()
        for drop_p in ps:
            metrics = test_metrics.get((gnn, depth, drop_p), list())
            xs.extend(metrics); ys.extend([drop_p]*len(metrics))
        corr.append(spearmanr(xs, ys).statistic)
    ax.plot(depths, corr)
    ax.set_xlabel('Model Depth', fontsize=14)
    ax.set_ylabel(f'Corr with Test {args.metric}', fontsize=14)
    ax.set_title(gnn, fontsize=14)
    ax.grid()

fig.suptitle(args.dataset, fontsize=16)
fig.tight_layout()
fn = f'./{assets_dir}/{args.which}/{args.metric}/test.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')

### PLOT FOR GEN GAP ###

fig, axs = plt.subplots(1, len(args.gnns), figsize=(6.4*len(args.gnns), 4.8))
if not hasattr(axs, '__len__'): axs = (axs,)

for gnn, ax in zip(args.gnns, axs):
    corr = list()
    for depth in depths:
        xs, ys = list(), list()
        for drop_p in ps:
            metrics = gap_metrics.get((gnn, depth, drop_p), list())
            xs.extend(metrics); ys.extend([drop_p]*len(metrics))
        corr.append(spearmanr(xs, ys).statistic)
    ax.plot(depths, corr)
    ax.set_xlabel('Model Depth', fontsize=14)
    ax.set_ylabel(f'Corr with Gen. Gap in {args.metric}', fontsize=14)
    ax.set_title(gnn, fontsize=14)
    ax.grid()

fig.suptitle(args.dataset, fontsize=16)
fig.tight_layout()
fn = f'./{assets_dir}/{args.which}/{args.metric}/gen-gap.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')