import argparse
import os
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

import numpy as np

from utils.parse_logs import parse_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--node', action='store_true')
parser.add_argument('--graph', action='store_true')
parser.add_argument('--significance', action='store_true')
parser.add_argument('--effect_size', action='store_true')
parser.add_argument('--best_prob', action='store_true')
args = parser.parse_args()

assert sum((args.node, args.graph)) == 1, 'Exactly one must be true.'
assert sum((args.significance, args.effect_size, args.best_prob)) == 1, 'Exactly one must be true.'

if args.significance:
    from tables.significance import cell_value
elif args.effect_size:
    from tables.effect_size import cell_value
elif args.best_prob:
    from tables.best_prob import cell_value

if args.node:
    datasets = ('Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel', 'TwitchDE')
elif args.graph:
    datasets = ('Proteins', 'Mutag', 'Enzymes', 'Reddit', 'IMDb', 'Collab')
gnns = ('GCN', 'GAT')
dropouts = ('DropEdge', 'DropNode', 'DropAgg', 'DropGNN', 'Dropout', 'DropMessage', 'DropSens')

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

    best_mean, best_samples, best_config = float('-inf'), None, None

    for drop_p in ((0.2, 0.3, 0.5, 0.8) if dropout == 'DropSens' else drop_ps):
        for info_loss_ratio in (info_loss_ratios if dropout == 'DropSens' else (None,)):
            samples = get_samples(dataset, gnn, dropout, drop_p, info_loss_ratio)
            # Use at least 10 samples and at most 20 samples for computing the best config
            mean = np.mean(samples[:20]) if len(samples) >= 10 else np.nan
            if mean > best_mean:
                best_mean, best_samples, best_config = mean, samples, (drop_p, info_loss_ratio)
    
    # Return all samples (more than 20 for the best config)
    return best_samples, best_config

data = dict()
for dataset in tqdm(datasets):
    for gnn in gnns:
        base_drop_samples = get_samples(dataset, gnn, 'NoDrop', 0.0)
        for dropout in dropouts:
            best_drop_samples, best_config = get_best(dataset, gnn, dropout)
            if best_drop_samples is None:
                continue
            data[(dropout, gnn, dataset)] = cell_value(base_drop_samples, best_drop_samples, best_config)

for dropout in dropouts:
    print(f"\\multirow{{2}}{{*}}{{{dropout}}}", end='')
    for gnn in gnns:
        print(f' & {gnn} & ', end='')
        to_print = list()
        for dataset in datasets:
            to_print.append(data.get((dropout, gnn, dataset), ''))
        print(f"{' & '.join(to_print)} \\\\ ", end='')
        if gnn != gnns[-1]:
            print(f"\\hhline{{|~|{'-'*7}|}}")
        else:
            print('\\hline')