import argparse
import os
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from utils.parse_logs import parse_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--node', action='store_true')
parser.add_argument('--graph', action='store_true')
args = parser.parse_args()

if args.node:
    # 'Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel', 'TwitchDE', 
    datasets = ('Actor',)
    driver = 'experiments/node.sh'
elif args.graph:
    datasets = ('Proteins', 'Mutag', 'Enzymes', 'Reddit', 'IMDb', 'Collab')
    driver = 'experiments/graph.sh'
else:
    raise ValueError('At least one of args.node and args.graph needs to be true.')

gnns = ('GCN', 'GAT')
dropouts = ('DropEdge', 'DropNode', 'DropAgg', 'DropGNN', 'Dropout', 'DropMessage')

metric = 'Accuracy'
drop_ps = np.round(np.arange(0.1, 1, 0.1), decimals=1)
exp_dir = './results/{dataset}/{gnn}/L=4/{dropout}/P={drop_p}'


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
        sample = test[metric][np.argmax(val[metric])]
        samples.append(sample)

    if len(samples) < 20:
        print(dataset, gnn, dropout, drop_p)

    return samples[:20]

def find_best_drop_p(dataset, gnn, dropout):

    best_mean, best_drop_p = float('-inf'), None

    for drop_p in drop_ps:
        samples = get_samples(dataset, gnn, dropout, drop_p)
        mean = np.mean(samples)
        if mean > best_mean:
            best_mean, best_drop_p = mean, drop_p
    
    return best_drop_p

device_index = 0
for dataset in datasets:
    for gnn in gnns:
        for dropout in dropouts:
            config_dir = f"./results/{dropout}/{dataset}/{gnn}/L=4/P={find_best_drop_p(dataset, gnn, dropout)}"
            print(f'bash {driver} --datasets {dataset} --gnns {gnn} --dropouts {dropout} --drop_ps {find_best_drop_p(dataset, gnn, dropout)} --device_index {device_index} 50', end='; ')
            # print(f'echo {config_dir} $(find {config_dir} -mindepth 1 -type d 2>/dev/null | wc -l)', end='; ')
print()