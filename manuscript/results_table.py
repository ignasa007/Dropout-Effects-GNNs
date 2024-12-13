import os
import warnings; warnings.filterwarnings('ignore')
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from utils.parse_logs import parse_metrics, parse_configs

datasets = ('Cora', 'CiteSeer', 'Chameleon', 'Squirrel', 'TwitchDE')
cutoffs = (0.3021, 0.2107, 0.2288, 0.1203, 0.6045)
gnns = ('GCN', 'GAT')
dropouts = ('DropEdge', 'DropNode', 'DropAgg', 'DropGNN', 'Dropout', 'DropMessage', 'SkipNode')

metric = 'Accuracy'
depths = range(2, 9, 2)
ps = np.round(np.arange(0.0, 1, 0.1), decimals=1)

for dropout in dropouts:

    print(f'\\multirow{{2}}{{*}}{{{dropout}}}', end='')
    
    for gnn in gnns:

        print(f' & {gnn} & ', end='')
        to_print = list()

        for dataset, cutoff in zip(datasets, cutoffs):

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
                        train, val, test = parse_metrics(f'{exp_dir_format}/{sample_dir}/logs')
                        value = np.max(test[metric][np.argmax(val[metric])])
                        if value > cutoff:
                            test_metrics[(depth, p)].append(value)

            ### CORRELATION STATS ###

            status = 'complete'
            corrs = list()
            for depth in depths:
                drop_ps, accs = list(), list()
                for drop_p in ps:
                    exp = (depth, drop_p)
                    if exp in test_metrics:
                        samples = test_metrics[exp]
                        drop_ps.extend([drop_p]*len(samples)); accs.extend(samples)
                    if exp not in test_metrics or len(samples) < 5:
                        status = 'incomplete'
                if len(accs) < 2:
                    continue
                spearman = spearmanr(drop_ps, accs).statistic
                if not np.isnan(spearman):
                    corrs.append(spearman)
            if corrs:
                mean = np.mean(corrs)
                if mean > 0:
                    to_print.append(f"\\cellcolor{{\\positive!{100*mean:.3f}}} ${'+' if mean > 0 else ''}{mean:.3f}$")
                else:
                    to_print.append(f"\\cellcolor{{\\negative!{-100*mean:.3f}}} ${mean:.3f}$")
            
        print(f"{' & '.join(to_print)} \\\\ ", end='')
        if gnn == 'GCN':
            print('\\cline{2-7}')
        else:
            print('\\hline')