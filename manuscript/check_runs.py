import os
import shutil
from itertools import product
from collections import defaultdict
import numpy as np
from utils.parse_logs import parse_metrics, parse_configs


datasets = ('Cora', 'CiteSeer', 'Chameleon', 'Squirrel', 'TwitchDE')
gnns = ('GCN', 'GAT')
dropouts = ('DropEdge', 'DropNode', 'DropAgg', 'DropGNN', 'Dropout', 'DropMessage', 'SkipNode')

metric = 'Accuracy'
depths = range(2, 9, 2)
drop_ps = np.round(np.arange(0.0, 1, 0.1), decimals=1)

for dropout, dataset in product(dropouts, datasets):

    results_dir = f'./results/{dropout}/{dataset}'
    exp_dir = results_dir + '/{gnn}/L={depth}/P={drop_p}'
    test_metrics = defaultdict(list)

    for gnn in gnns:
        for depth in depths:
            for drop_p in drop_ps:
                exp_dir_format = exp_dir.format(gnn=gnn, depth=depth, drop_p=drop_p)
                if not os.path.exists(exp_dir_format):
                    continue
                if len(os.listdir(exp_dir_format)) != 5:
                    print(exp_dir_format, len(os.listdir(exp_dir_format)))
                    for sample_dir in os.listdir(exp_dir_format)[:-5]:
                        shutil.rmtree(f'{exp_dir_format}/{sample_dir}')
                for sample_dir in os.listdir(exp_dir_format):
                    config = parse_configs(f'{exp_dir_format}/{sample_dir}/logs')
                    if not all([x == 64 for x in eval(config['gnn_layer_sizes'])]):
                        print(f'{exp_dir_format}/{sample_dir}', f"eval(config['gnn_layer_sizes']) = {eval(config['gnn_layer_sizes'])}")
                        continue
                    train, val, test = parse_metrics(f'{exp_dir_format}/{sample_dir}/logs')
                    if len(train.get('Epoch', list())) != 300:
                        print(f'{exp_dir_format}/{sample_dir}', f"len(train.get('Epoch', list())) = {len(train.get('Epoch', list()))}")
                        continue
                    test_metrics[(gnn, depth, drop_p)].append(test[metric][np.argmax(val[metric])])
                if len(test_metrics[(gnn, depth, drop_p)]) != 5:
                    print(f'{exp_dir_format}', f"len(test_metrics[(gnn, depth, drop_p)]) = {len(test_metrics[(gnn, depth, drop_p)])}")