import os

import numpy as np
import matplotlib.pyplot as plt

from tables.main import get_best


datasets = ('Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel', 'TwitchDE', 'Actor',) + \
    ('Mutag', 'Proteins', 'Enzymes', 'Reddit', 'IMDb', 'Collab',)
gnn = 'GCN'
baseline_method = 'DropEdge'
dropouts = ('DropSens',)

metric = 'Accuracy'

fig, ax = plt.subplots(1, 1, figsize=(11.2, 4.8))
xs_de = np.arange(len(datasets))
width = 0.8/len(dropouts)
displacements = np.arange(-(len(dropouts)-1), len(dropouts), 2) * (width/2)

baseline_samples = dict()
for dataset in datasets:
    _, best_samples = get_best(dataset, gnn, baseline_method)
    baseline_samples[dataset] = best_samples

for dropout, displacement in zip(dropouts, displacements):
    
    pcnt_diffs, mean_diffs, std_errors = list(), list(), list()
    
    for dataset in datasets:
        
        baseline_mean, baseline_std = np.mean(baseline_samples[dataset]), np.std(baseline_samples[dataset], ddof=1)
        _, best_samples = get_best(dataset, gnn, dropout)
        best_samples = best_samples[-50:]
        best_mean, best_std = np.mean(best_samples), np.std(best_samples, ddof=1)
    
        pcnt_diffs.append(100*(best_mean-baseline_mean)/(1-baseline_mean))
        mean_diffs.append(100*(best_mean-baseline_mean))
        std_errors.append(100*np.sqrt(baseline_std**2/len(baseline_samples[dataset]) + best_std**2/len(best_samples)))
    
    ax.bar(x=xs_de+displacement, height=pcnt_diffs, width=width, label=dropout)
    for x, y, mean_diff, std_err in zip(xs_de+displacement, pcnt_diffs, mean_diffs, std_errors):
        ax.text(x, max(0,y)+0.5, f'{mean_diff:+.2f}\n± {std_err:.2f}', ha='center', va='bottom', fontsize=12)

ax.set_xticks(xs_de, datasets, rotation=30, fontsize=15)
yticks = np.arange(-2.5, 22.6, 2.5)
ax.set_yticks(yticks, yticks, fontsize=15)
ax.set_ylabel('Relative Error Change (%)', fontsize=18)
ax.grid()
if len(dropouts) > 1:
    ax.legend()

fig.tight_layout()
fn = f'./assets/DropSens/errors-diff.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)