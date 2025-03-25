import argparse
import os
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats

from utils.parse_logs import parse_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--node', action='store_true')
parser.add_argument('--graph', action='store_true')
args = parser.parse_args()

if args.node:
    datasets = ('Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel', 'TwitchDE')
elif args.graph:
    datasets = ('Proteins', 'Mutag', 'Enzymes')
else:
    raise ValueError('At least one of args.node and args.graph needs to be true.')
gnns = ('GCN', 'GAT')
dropouts = ('DropSens',)

'''
for name in ('PROTEINS', 'MUTAG', 'ENZYMES', 'REDDIT-BINARY', 'IMDB-BINARY', 'COLLAB'): 
    dataset = TUDataset(root='./data/TUDataset', name=name, use_node_attr=True)
    print(dataset.y.unique(return_counts=True)[1].max().item() / dataset.y.size(0))
'''
cutoffs = {
    'Cora': 0.3021,
    'CiteSeer': 0.2107,
    'PubMed': None,
    'Chameleon': 0.2288,
    'Squirrel': 0.1203,
    'TwitchDE': 0.6045,
    'Proteins': 0.5957,
    'Mutag': 0.6649,
    'Enzymes': 0.1667,
    'Reddit': 0.5000,
    'IMDb': 0.5000,
    'Collab': 0.5200,
}

metric = 'Accuracy'
drop_ps = (0.2, 0.3, 0.5, 0.8)
info_loss_ratios = (0.5, 0.8, 0.9, 0.95)
exp_dir = './results/{dropout}/{dataset}/{gnn}/L=4/P={drop_p}/C={info_loss_ratio}'


def get_samples(dataset, gnn, dropout, drop_p, info_loss_ratio):

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
        for info_loss_ratio in info_loss_ratios:
            samples = get_samples(dataset, gnn, dropout, drop_p, info_loss_ratio)
            mean = np.mean(samples) if len(samples) >= 10 else np.nan
            if mean > best_mean:
                best_mean, best_samples = mean, samples
    
    return best_samples

def is_normal(samples):

    # Failed to reject the null hypothesis of normal distribution of data at 90% confidence
    return stats.shapiro(samples)[0] > 0.1

def compare_samples(no_drop_samples, best_drop_samples):

    '''
    Testing the hypothesis that NoDrop performs worse than the given Dropout method
        under the null hypothesis of equal means.
    '''

    assert is_normal(no_drop_samples) and is_normal(best_drop_samples)

    statistic, pvalue = stats.ttest_ind(
        no_drop_samples,
        best_drop_samples,
        equal_var=False,    # Dropout samples should have a higher variance
        alternative='less'  # The mean of NoDrop samples is less than the mean of BestDrop samples
    )

    return statistic, pvalue

def color_effect_size(value):

    if value < -0.65:      # Between -inf and -0.65, includes -0.8
        out = '\\cellcolor{\\negative!80}'
    elif value < -0.35:    # Between -0.65 and -0.35, includes -0.5
        out = '\\cellcolor{\\negative!50}'
    elif value < -0.10:    # Between -0.35 and -0.10, includes -0.2
        out = '\\cellcolor{\\negative!20}'
    elif value < +0.10:    # Between -0.10 and +0.10, includes 0.0
        out = ''
    elif value < +0.35:    # Between +0.10 and +0.35, includes +0.2
        out = '\\cellcolor{\\positive!20}'
    elif value < +0.65:    # Between +0.35 and +0.65, includes +0.5
        out = '\\cellcolor{\\positive!50}'
    else:                  # Between +0.65 and +inf, includes +0.8
        out = '\\cellcolor{\\positive!80}'

    return out+' ' if out else out

data = dict()

for dataset in tqdm(datasets):    
    for gnn in gnns:
        no_drop_samples = get_samples(dataset, gnn, 'NoDrop', 0.0, None)
        no_drop_mean, no_drop_std = np.mean(no_drop_samples), np.std(no_drop_samples, ddof=1)
        for dropout in dropouts:
            best_drop_samples = get_best(dataset, gnn, dropout)
            if best_drop_samples is None:
                continue
            best_drop_mean, best_drop_std = np.mean(best_drop_samples), np.std(best_drop_samples, ddof=1)
            # cell_value = f'{100*(best_drop_mean-no_drop_mean):.3f}'
            # if cell_value[0].isdigit():
            #     cell_value = f'+{cell_value}'
            # statistic, pvalue = compare_samples(no_drop_samples, best_drop_samples)
            # if pvalue > 0.1:
            #     cell_value = f'\\cellcolor{{\\negative!{100*(pvalue-0.1)/0.9:.3f}}} ${cell_value}$'
            # else:
            #     cell_value = f'\\cellcolor{{\\positive!{100*(0.1-pvalue)/0.1:.3f}}} ${cell_value}$'
            # data[(dropout, gnn, dataset)] = cell_value
            cohens_d = (best_drop_mean-no_drop_mean) / np.sqrt((best_drop_std**2+no_drop_std**2)/2)
            hedges_correction = 1 - 3 / (4*(len(best_drop_samples)+len(no_drop_samples))-9)
            value = f'{hedges_correction*cohens_d:.3f}'
            if value[0].isdigit():
                value = f'+{value}'
            data[(dropout, gnn, dataset)] = f'{color_effect_size(float(value))}{value}'

for dropout in dropouts:
    print(f'\\multirow{{2}}{{*}}{{{dropout}}}', end='')
    for gnn in gnns:
        print(f' & {gnn} & ', end='')
        to_print = list()
        for dataset in datasets:
            to_print.append(data.get((dropout, gnn, dataset), ''))
        print(f"{' & '.join(to_print)} \\\\ ", end='')
        if gnn != gnns[-1]:
            print('\\cline{2-8}')
        else:
            print('\\hline')