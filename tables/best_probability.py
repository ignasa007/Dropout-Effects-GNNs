import argparse
import os
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

import numpy as np

from utils.parse_logs import parse_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--node', action='store_true')
parser.add_argument('--graph', action='store_true')
args = parser.parse_args()

if args.node:
    datasets = ('Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel', 'TwitchDE')
elif args.graph:
    datasets = ('Proteins', 'Mutag', 'Enzymes', 'Reddit', 'IMDb', 'Collab')
else:
    raise ValueError('At least one of args.node and args.graph needs to be true.')

gnns = ('GCN', 'GAT')
dropouts = ('DropEdge', 'DropNode', 'DropAgg', 'DropGNN', 'Dropout', 'DropMessage')

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
exp_dir = './results/{dropout}/{dataset}/{gnn}/L=4/P={drop_p}'


def get_samples(dataset, gnn, dropout, drop_p):

    exp_dir_format = exp_dir.format(dropout=dropout, dataset=dataset, gnn=gnn, drop_p=drop_p)
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
        print(dataset, gnn, dropout, drop_p)

    return samples

def find_best_drop_p(dataset, gnn, dropout):

    best_mean, best_drop_p = float('-inf'), None

    for drop_p in drop_ps:
        samples = get_samples(dataset, gnn, dropout, drop_p)
        # Use at least 10 samples and at most 20 samples for computing the best config
        mean = np.mean(samples[:20]) if len(samples) >= 10 else np.nan
        if mean > best_mean:
            best_mean, best_drop_p = mean, drop_p
    
    return best_drop_p

data = dict()

for dataset in tqdm(datasets):    
    for gnn in gnns:
        for dropout in dropouts:
            data[(dropout, gnn, dataset)] = find_best_drop_p(dataset, gnn, dropout)

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