import argparse
import os
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from utils.parse_logs import parse_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--p_value', action='store_true')
parser.add_argument('--test_acc', action='store_true')
parser.add_argument('--effect_size', action='store_true')
parser.add_argument('--best_prob', action='store_true')
parser.add_argument('--node', action='store_true')
parser.add_argument('--graph', action='store_true')
args = parser.parse_args()

assert sum((args.p_value, args.test_acc, args.effect_size, args.best_prob)) == 1, 'Exactly one must be true.'

if args.p_value:
    from tables.p_value import *
elif args.test_acc:
    from tables.test_acc import *
elif args.effect_size:
    from tables.effect_size import *
elif args.best_prob:
    from tables.best_prob import *

node_datasets = ('Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel', 'TwitchDE', 'Actor',)
graph_datasets = ('Mutag', 'Proteins', 'Enzymes', 'Reddit', 'IMDb', 'Collab',)
gnns = ('GCN', 'GAT', 'GIN',)
dropouts = ('DropEdge', 'DropNode', 'DropAgg', 'DropGNN', 'Dropout', 'DropMessage', 'DropSens')
datasets, gnns, dropouts = values(node_datasets, graph_datasets, gnns, dropouts, args)

metric = 'Accuracy'
drop_ps = np.round(np.arange(0.1, 1, 0.1), decimals=1)
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
        samples.append(round(sample, 12))

    if len(samples) < 20:
        print(dataset, gnn, drop_p, info_loss_ratio)

    return samples

def get_best(dataset, gnn, dropout):

    best_mean, best_samples, best_config = float('-inf'), None, None

    for drop_p in drop_ps:
        for info_loss_ratio in (info_loss_ratios if dropout == 'DropSens' else (None,)):
            if (drop_p, info_loss_ratio) in ((0.2, 0.5), (0.3, 0.5), (0.5, 0.5), (0.2, 0.8)):
                continue
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
            # Don't use more than 50 samples for comparison
            data[(dataset, gnn, dropout)] = cell_value(base_drop_samples[:50], best_drop_samples[:50], best_config)

indices1, indices2, columns, key = make_key(datasets, gnns, dropouts)
for index1 in indices1:
    print(f'\\multirow{{{len(indices2)}}}{{*}}{{{index1}}}', end='')
    for index2 in indices2:
        print(f' & {index2} & ', end='')
        to_print = list()
        for column in columns:
            to_print.append(data.get(key(index1, index2, column), ''))
        print(f"{' & '.join(to_print)} \\\\ ", end='')
        if index2 != indices2[-1]:
            print(f"\\hhline{{|~|{'-'*(1+len(columns))}|}}")
        else:
            print('\\hline')