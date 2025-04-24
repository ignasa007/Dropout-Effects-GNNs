import warnings
import argparse
import os
import shutil
from tqdm import tqdm
import numpy as np
from utils.parse_logs import parse_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--node', action='store_true')
parser.add_argument('--graph', action='store_true')
args = parser.parse_args()

assert sum((args.node, args.graph)) == 1, 'Exactly one must be true.'

if args.node:
    datasets = ('Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel', 'TwitchDE', 'Actor',)
elif args.graph:
    datasets = ('Proteins', 'Mutag', 'Enzymes', 'Reddit', 'IMDb', 'Collab',)
else:
    raise ValueError('At least one of args.node and args.graph needs to be true.')

gnns = ('GCN', 'GAT', 'GIN',)
dropouts = ('DropEdge', 'DropNode', 'DropAgg', 'DropGNN', 'Dropout', 'DropMessage', 'DropSens',)

metric = 'Accuracy'
drop_ps = np.round(np.arange(0.1, 1, 0.1), decimals=1)
info_loss_ratios = (0.5, 0.8, 0.9, 0.95)
exp_dir = './results/{dataset}/{gnn}/L=4/{dropout}/P={drop_p}/C={info_loss_ratio}'


def get_samples(dataset, gnn, dropout, drop_p, info_loss_ratio):

    exp_dir_format = exp_dir.format(dropout=dropout, dataset=dataset, gnn=gnn, drop_p=drop_p, info_loss_ratio=info_loss_ratio)
    if info_loss_ratio is None:
        exp_dir_format = os.path.dirname(exp_dir_format)
    
    samples = list()
    if not os.path.isdir(exp_dir_format):
        return exp_dir_format, samples
    
    for timestamp in os.listdir(exp_dir_format):
        train, val, test = parse_metrics(f'{exp_dir_format}/{timestamp}/logs')
        if len(test.get(metric, [])) < 300:
            print(f'Incomplete training run: {exp_dir_format}/{timestamp}')
            # shutil.rmtree(f'{exp_dir_format}/{timestamp}')
            continue
        sample = test[metric][np.argmax(val[metric])]
        samples.append(round(sample, 12))

    return exp_dir_format, samples

def find_best_config(dataset, gnn, dropout):

    best_mean = float('-inf')
    best_drop_p, best_info_loss_ratio = None, None
    best_samples = None

    for drop_p in ((0.2, 0.3, 0.5, 0.8) if dropout == 'DropSens' else drop_ps):
        for info_loss_ratio in (info_loss_ratios if dropout == 'DropSens' else (None,)):
            if (drop_p, info_loss_ratio) in ((0.2, 0.5), (0.3, 0.5), (0.5, 0.5), (0.2, 0.8), (0.3, 0.8)):
                continue
            exp_dir_format, samples = get_samples(dataset, gnn, dropout, drop_p, info_loss_ratio)
            if len(samples) < 20:
                warnings.warn('Not enough samples to find the best config. '
                    f'Only {len(samples)} samples for {exp_dir_format}.')
                return None, None, None
            mean = np.mean(samples[:20])
            if mean > best_mean:
                best_mean = mean
                best_drop_p, best_info_loss_ratio = drop_p, info_loss_ratio
                best_samples = samples

    return best_drop_p, best_info_loss_ratio, best_samples

device_index = 0
for dataset in tqdm(datasets):
    for gnn in gnns:
        for dropout in dropouts:
            best_drop_p, best_info_loss_ratio, best_samples = find_best_config(dataset, gnn, dropout)
            if best_drop_p is None:
                continue
            if len(best_samples) >= 50:
                continue
            if dropout != 'DropSens':
                driver = 'experiments/dropout.sh'
                print(f'bash {driver} --datasets {dataset} --gnns {gnn} --dropouts {dropout} --drop_ps {best_drop_p} --device_index {device_index} --total_samples 50', end='; ')
            else:
                driver = 'experiments/drop_sens.sh'
                print(f'bash {driver} --datasets {dataset} --gnns {gnn} --drop_ps {best_drop_p} --info_loss_ratios {best_info_loss_ratio} --device_index {device_index} --total_samples 50', end='; ')
print()