import tables.main as utils


device_index = 0
utils.node_datasets = ('Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel', 'TwitchDE', 'Actor',)
utils.graph_datasets = ('Mutag', 'Proteins', 'Enzymes', 'IMDb', 'Reddit', 'Collab')
utils.gnns = ('GCN', 'GIN', 'GAT',)
utils.dropout_methods = ('NoDrop', 'DropSens', 'DropEdge', 'DropNode', 'DropAgg', 'DropGNN', 'Dropout', 'DropMessage')
utils.metric, utils.higher_is_better = 'Accuracy', True
# utils.metric, utils.higher_is_better = 'Mean Absolute Error', False
total_samples = 50

data = dict()
for dataset in utils.node_datasets + utils.graph_datasets:
    for gnn in utils.gnns:
        for method in utils.dropout_methods:
            best_args, best_samples = utils.get_best(dataset, gnn, method)
            if best_samples is None:
                print(f'\n{utils.exp_dir.format(dataset=dataset, gnn=gnn, method=method)}')
                continue
            if len(best_samples) > total_samples:
                print(
                    f'\nGot len(best_samples) = {len(best_samples)} for '
                    f'{utils.exp_dir.format(dataset=dataset, gnn=gnn, method=method)}/{"/".join(best_args)}.'
                )
                continue
            if len(best_samples) == total_samples:
                continue
            best_args = map(lambda arg: arg.split('=')[1], best_args)
            if method == 'DropSens':
                driver = 'experiments/drop_sens.sh'
                best_drop_p, best_info_save_ratio = best_args
                print(f'bash {driver} --datasets {dataset} --gnns {gnn} --drop_ps {best_drop_p} --info_save_ratios {best_info_save_ratio} --device_index {device_index} --total_samples {total_samples}', end='; ')
            else:
                driver = 'experiments/dropout.sh'
                best_drop_p, = best_args
                print(f'bash {driver} --datasets {dataset} --gnns {gnn} --dropouts {method} --drop_ps {best_drop_p} --device_index {device_index} --total_samples {total_samples}', end='; ')
print()