import os
from collections import defaultdict
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.parse_logs import parse_metrics


drop_ps = (0.2, 0.5)
distances = list(map(lambda x: round(x, 2), np.arange(0.1, 1.01, 0.10)))

parser = argparse.ArgumentParser()
parser.add_argument('--gnn', type=str, choices=['mse', 'mae', 'mape'], default='GCN')
parser.add_argument('--metric', type=str, choices=['mse', 'mae', 'mape'], default='mae')
parser.add_argument('--which', type=str, nargs='+', choices=['best', 'final'], default=['best'])
args = parser.parse_args()

dropouts = ('NoDrop', 'DropEdge', 'Dropout', 'DropMessage')
colors = ('black', 'blue', 'green', 'maroon')
fn_to_matric = {'mse': 'Squared', 'mae': 'Absolute', 'mape': 'Absolute Percentage'}
metric_fn, args.metric = args.metric, f'Mean {fn_to_matric[args.metric]} Error'
assets_dir = './assets'
exp_dir = './results/{dropout}/{gnn}/P={P}/distance={distance}'

store = {'train': defaultdict(list), 'test': defaultdict(list)}

for dropout in dropouts:
    for p in (drop_ps if dropout != 'NoDrop' else (0.0,)):
        for distance in distances:
            exp_dir_format = exp_dir.format(dropout=dropout, gnn=args.gnn, P=p, distance=f'{distance:.2f}')
            if not os.path.exists(exp_dir_format):
                continue
            for sample_dir in os.listdir(exp_dir_format):
                train, val, test = parse_metrics(f'{exp_dir_format}/{sample_dir}/logs')
                if len(train[args.metric]) < 200:
                    continue
                if args.which == 'best':
                    store['train'][(dropout, p, distance)].append(min(train[args.metric]))
                    store['test'][(dropout, p, distance)].append(test[args.metric][np.argmin(val[args.metric])])
                elif args.which == 'final':
                    store['train'][(dropout, p, distance)].append(train[args.metric][-1])
                    store['test'][(dropout, p, distance)].append(test[args.metric][-1])
                
def plot(dropout, label, color):

    for p, style in zip(ps, styles):
        means = list()
        for distance in distances:
            metrics = store[label].get((dropout, p, distance), (np.nan, np.nan, np.nan))
            means.append(np.mean(metrics))
        ax.plot(distances, means, style, color=color, label=f'{dropout}({p:.1f})')

fig, axs = plt.subplots(1, len(store), figsize=(6.4*len(store), 4))
styles = ('-', '--')
for label, ax in zip(store, axs):
    for dropout, color in zip(dropouts, colors):
        ps = drop_ps if dropout != 'NoDrop' else (0.0,)
        plot(dropout, label, color)
        if dropout == 'NoDrop':
            ax.plot(np.NaN, np.NaN, '-', color='none', label=' ')
    ax.set_xlabel('Commute Time Percentile', fontsize=16)
    ax.set_ylabel(f"{label.capitalize()} {''.join(map(lambda x: x[0], args.metric.split()))}", fontsize=16)
    ax.grid()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=16, ncol=4, bbox_to_anchor = (0.5, -0.2))
fig.tight_layout()
fn = f'./{assets_dir}/SyntheticZINC/{args.gnn}/{args.which}-{args.metric}.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')