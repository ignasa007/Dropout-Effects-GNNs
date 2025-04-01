import os
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

from utils.parse_logs import parse_metrics


datasets = ('Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel', 'TwitchDE') + \
    ('Proteins', 'Mutag', 'Enzymes', 'Reddit', 'IMDb', 'Collab')
gnns = ('GCN',)
dropout = 'DropSens'

metric = 'Accuracy'
drop_ps = np.round(np.arange(0.1, 1, 0.1), decimals=1)
info_loss_ratios = (0.5, 0.8, 0.9, 0.95)
exp_dir = './results/{dropout}/{dataset}/{gnn}/L=4/P={drop_p}/C={info_loss_ratio}'


def get_samples(dataset, gnn, dropout, drop_p, info_loss_ratio=None):

    exp_dir_format = exp_dir.format(dropout=dropout, dataset=dataset, gnn=gnn, drop_p=drop_p, info_loss_ratio=info_loss_ratio)
    if info_loss_ratio is None:
        exp_dir_format = os.path.dirname(exp_dir_format)

    samples = list()
    if not os.path.isdir(exp_dir_format):
        return samples
    
    for timestamp in os.listdir(exp_dir_format):
        train, val, test = parse_metrics(f'{exp_dir_format}/{timestamp}/logs')
        if len(test.get(metric, [])) < 300:
            # print(f'Incomplete training run: {exp_dir_format}/{timestamp}')
            continue
        # if np.max(train[metric]) < cutoffs[dataset]:
            # print(f'Failed to learn: {exp_dir_format}/{timestamp}, {np.max(train[metric])} < {cutoffs[dataset]}')
            # pass
        sample = test[metric][np.argmax(val[metric])]
        samples.append(sample)

    if len(samples) < 20:
        print(dataset, gnn, drop_p, info_loss_ratio)

    return samples

def get_best(dataset, gnn, dropout):

    best_mean, best_samples = float('-inf'), None

    for drop_p in drop_ps:
        for info_loss_ratio in (info_loss_ratios if dropout == 'DropSens' else (None,)):
            samples = get_samples(dataset, gnn, dropout, drop_p, info_loss_ratio)
            # Use at least 10 samples and at most 20 samples for computing the best config
            mean = np.mean(samples[:20]) if len(samples) >= 10 else np.nan
            if mean > best_mean:
                best_mean, best_samples = mean, samples
    
    # Return all samples (more than 20 for the best config)
    return best_samples

# best_de_means, best_ds_means = list(), list()
# diff_means, diff_stds = list(), list()
delta_error_rates = list()

for dataset in tqdm(datasets):
    for gnn in gnns:
        best_de_samples = get_best(dataset, gnn, 'DropEdge')
        if not best_de_samples:
            continue
        best_de_mean, best_de_std = np.mean(best_de_samples), np.std(best_de_samples, ddof=1)
        best_ds_samples = get_best(dataset, gnn, 'DropSens')
        if not best_ds_samples:
            continue
        best_ds_mean, best_ds_std = np.mean(best_ds_samples), np.std(best_ds_samples, ddof=1)
        print(dataset, gnn, best_ds_mean)
        # best_de_means.append(100*best_de_mean); best_ds_means.append(100*best_ds_mean)
        # s_pool = np.sqrt((
        #     (len(best_ds_samples)-1) * best_ds_std**2 + 
        #     (len(best_de_samples)-1) * best_de_std**2
        # ) / (len(best_ds_samples) + len(best_de_samples) - 2))
        # diff_means.append(100*(best_ds_mean-best_de_mean))
        # diff_stds.append(100*(s_pool))
        delta_error_rates.append(100*((1-best_de_mean)-(1-best_ds_mean))/(1-best_de_mean))

fig, ax = plt.subplots(1, 1, figsize=(11.2, 4.8))
xs_de = np.arange(len(datasets))
# w = 0.4
# ax.bar(x=xs_de, height=best_de_means, width=w, label='DropEdge')
# ax.bar(x=xs_de+w, height=best_ds_means, width=w, label='DropSens')
# ax.bar(x=xs_de, height=diff_means)
# ax.errorbar(xs_de, diff_means, diff_stds, fmt='.', ecolor='black', capsize=10)
ax.bar(x=xs_de, height=delta_error_rates)

# ax.set_xticks(xs_de+w/2, datasets)
ax.set_xticks(xs_de, datasets, rotation=45, fontsize=15)
yticks = np.arange(-2.5, 22.5, 2.5)
ax.set_yticks(yticks, yticks, fontsize=15)
ax.set_ylabel('Relative Error Change (%)', fontsize=18)
ax.grid()
# ax.legend()

fig.tight_layout()
plt.savefig('errors-diff.png')