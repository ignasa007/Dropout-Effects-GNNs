import os
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

from utils.parse_logs import parse_metrics


datasets = ('Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel', 'TwitchDE', 'Actor') + \
    ('Mutag', 'Proteins', 'Enzymes', 'Reddit', 'IMDb', 'Collab')
gnn = 'GCN'
baseline = 'DropEdge'
dropouts = ('DropSens',)

metric = 'Accuracy'
drop_ps = np.round(np.arange(0.0, 1, 0.1), decimals=1)
info_loss_ratios = (0.5, 0.8, 0.9, 0.95)
exp_dir = './results/{dataset}/{gnn}/L=4/{dropout}/P={drop_p}/C={info_loss_ratio}'


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
        sample = test[metric][np.argmax(val[metric])]
        samples.append(sample)

    return samples

def get_best(dataset, gnn, dropout):

    best_mean, best_samples = float('-inf'), None

    for drop_p in drop_ps:
        for info_loss_ratio in (info_loss_ratios if dropout == 'DropSens' else (None,)):
            if (drop_p, info_loss_ratio) in ((0.2, 0.5), (0.3, 0.5), (0.5, 0.5), (0.2, 0.8)):
                continue
            samples = get_samples(dataset, gnn, dropout, drop_p, info_loss_ratio)
            # Use at least 10 samples and at most 20 samples for computing the best config
            mean = np.mean(samples[:20]) if len(samples) >= 10 else np.nan
            if mean > best_mean:
                best_mean, best_samples = mean, samples
    
    # Return all samples (more than 20 for the best config)
    return best_samples


fig, ax = plt.subplots(1, 1, figsize=(11.2, 4.8))
xs_de = np.arange(len(datasets))
width = 0.8/len(dropouts)
displacements = np.arange(-(len(dropouts)-1), len(dropouts), 2) * (width/2)

baseline_samples = dict()
for dataset in tqdm(datasets):
    best_samples = get_best(dataset, gnn, baseline)
    baseline_samples[dataset] = best_samples

for dropout, displacement in zip(dropouts, displacements):
    heights = list()
    for dataset in tqdm(datasets):
        best_samples = get_best(dataset, gnn, dropout)
        if not best_samples:
            continue
        best_mean, best_std = np.mean(best_samples[:50]), np.std(best_samples[:50], ddof=1)
        baseline_mean = np.mean(baseline_samples[dataset][:50])
        heights.append(100*(best_mean-baseline_mean)/(1-baseline_mean))
    ax.bar(x=xs_de+displacement, height=heights, width=width, label=dropout)

ax.set_xticks(xs_de, datasets, rotation=30, fontsize=15)
# yticks = np.arange(-2.5, 22.5, 2.5)
# ax.set_yticks(yticks, yticks, fontsize=15)
ax.set_ylabel('Relative Error Change (%)', fontsize=18)
ax.grid()

fig.tight_layout()
fn = './assets/DropSens/tmp.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)