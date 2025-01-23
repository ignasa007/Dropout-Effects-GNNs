import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils.parse_logs import parse_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--sd', action='store_true')
parser.add_argument('--ct', action='store_true')
args = parser.parse_args()

assert args.sd ^ args.ct    # only one of sd or ct is turned on

if args.sd:
    distances = (4, 5, 6, 7, 8)
    exp_dir = 'results/{dropout}/SyntheticZINC_SD/{gnn}/L={distance}/P={drop_p}'
    save_fn = 'assets/synthetics/zinc_sd.png'
elif args.ct:
    distances = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, \
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    exp_dir = 'results/{dropout}/SyntheticZINC_CT/{gnn}/alpha={distance}/P={drop_p}'
    save_fn = 'assets/synthetics/zinc_ct.png'

gnns = (
    'GCN',
    'GAT',
    'GIN',
)
dropouts = (
    'Dropout',
    'DropMessage',
    'DropEdge',
    # 'DropNode',
    # 'DropAgg',
    # 'DropGNN'
)
drop_ps = (
    0.1,
    0.2,
    0.5
)

def plot(ax, gnn, dropout, drop_p):
    means, stds = list(), list()
    for distance in distances:
        exp_dir_format = exp_dir.format(dropout=dropout, gnn=gnn, distance=distance, drop_p=drop_p)
        samples = list()
        for timestamp in os.listdir(exp_dir_format):
            train, val, test = parse_metrics(f'{exp_dir_format}/{timestamp}/logs')
            if len(test.get('Mean Absolute Error', [])) < 250:
                continue
            sample = test['Mean Absolute Error'][np.argmin(val['Mean Absolute Error'])]
            # sample = np.min(train['Mean Absolute Error'])
            samples.append(sample)
            # if gnn == 'GCN' and sample > 0.3:
            #     print(f'{exp_dir_format}/{timestamp}/logs')
        # means.append(np.min(samples))
        means.append(np.mean(samples))
        stds.append(np.std(samples))
    means, stds = np.array(means), np.array(stds)
    # ax.plot(distances, means, label=gnn)
    ax.plot(distances, means, label=dropout)
    # ax.fill_between(distances, means-stds, means+stds, alpha=0.2)
        # ax.scatter([distance]*len(samples), samples)

fig, axs = plt.subplots(len(drop_ps), len(gnns), figsize=(6.4*len(gnns), 4.8*len(drop_ps)))
# fig, axs = plt.subplots(len(drop_ps), 1, figsize=(6.4, 4.8*len(drop_ps)))
if not hasattr(axs, '__len__'): axs = np.array((axs,))
for i, ax in enumerate(axs.flatten()):
    drop_p, gnn = drop_ps[i//len(gnns)], gnns[i%len(gnns)]
    # drop_p = drop_ps[i%len(drop_ps)]
    # for gnn in gnns:
        # plot(ax, gnn, 'NoDrop', 0.0)
    for dropout in dropouts:
        plot(ax, gnn, dropout, drop_p)
    # ax.set_yscale('log')
    ax.grid()
    ax.legend()

fig.tight_layout()
if not os.path.isdir(os.path.dirname(save_fn)):
    os.makedirs(os.path.dirname(save_fn))
plt.savefig(save_fn, bbox_inches='tight')