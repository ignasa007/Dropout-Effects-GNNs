import os
import warnings; warnings.filterwarnings('ignore')
from collections import defaultdict
from itertools import product

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from utils.parse_logs import parse_metrics, parse_configs


homo_data = ('Cora', 'CiteSeer')
hetero_data = ('Chameleon', 'Squirrel', 'TwitchDE')
homo_cutoffs = (0.3021, 0.2107)
hetero_cutoffs = (0.2288, 0.1203, 0.6045)
gnn = 'GCN'
dropout = 'DropNode'

metric = 'Accuracy'
depths = range(2, 9, 2)
ps = np.round(np.arange(0.0, 1, 0.1), decimals=1)

ncol = np.ceil(len(depths)/1)

for fn, datasets, cutoffs in zip(('homophilic', 'heterophilic'), (homo_data, hetero_data), (homo_cutoffs, hetero_cutoffs)):

    fig, axs = plt.subplots(1, len(datasets), figsize=(6.4*len(datasets), 4.8))
    if not hasattr(axs, '__len__'): axs = (axs,)

    axs[0].set_ylabel(f'Test {metric}', fontsize=20)

    for dataset, cutoff, ax in zip(datasets, cutoffs, axs):

        results_dir = f'./results/{dropout}/{dataset}'
        exp_dir = results_dir + '/{gnn}/L={depth}/P={p}'

        ### RETRIEVE METRICS ###

        test_metrics = defaultdict(list)

        for depth in depths:
            for p in ps:
                exp_dir_format = exp_dir.format(gnn=gnn, depth=depth, p=p)
                if not os.path.exists(exp_dir_format):
                    continue
                for sample_dir in os.listdir(exp_dir_format):
                    config = parse_configs(f'{exp_dir_format}/{sample_dir}/logs')
                    if not all([x == 64 for x in eval(config['gnn_layer_sizes'])]):
                        print(f'{exp_dir_format}/{sample_dir}')
                    train, val, test = parse_metrics(f'{exp_dir_format}/{sample_dir}/logs')
                    if len(train[metric]) != 300:
                        continue
                    value = np.max(test[metric][np.argmax(val[metric])])
                    if value > cutoff:
                        test_metrics[(depth, p)].append(value)
                    
        for depth in depths:
            for p in ps:
                if len(test_metrics[(depth, p)]) != 5:
                    print(dataset, gnn, depth, p, len(test_metrics[(depth, p)]))

        ### PLOT FOR TEST SET ###

        test_metrics = {exp: (np.mean(samples[:5]), np.std(samples[:5])) for exp, samples in test_metrics.items()}

        for depth in depths:
            drop_ps, means, lower, upper = list(), list(), list(), list()
            for drop_p in ps:
                if (depth, drop_p) in test_metrics:
                    mean, std = test_metrics[(depth, drop_p)]
                    drop_ps.append(drop_p); means.append(mean); lower.append(mean-std); upper.append(mean+std)
            ax.plot(drop_ps, means, label=f'L = {depth}')
        ax.set_xlabel(fr'{dropout} Probability, $q$', fontsize=20)
        ax.set_title(dataset, fontsize=20)
        ax.grid()

    if len(datasets) == 3:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', fontsize=18, ncol=ncol, bbox_to_anchor = (0.5, -0.02))
    fig.tight_layout()
    fn = f'./assets/philia/{dropout}/{fn}.png'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn, bbox_inches='tight')