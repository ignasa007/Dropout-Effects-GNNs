def values(node_datasets, graph_datasets, gnns, dropouts, args):

    datasets = node_datasets + graph_datasets

    return datasets, gnns, dropouts

def cell_value(base_drop_samples, best_drop_samples, best_config):
    
    if not hasattr(best_config, '__len__'):
        best_config = (best_config,)
    
    out = ', '.join(map(lambda x: f'${x}$', (config for config in best_config if config is not None)))

    return out

def make_key(datasets, gnns, dropouts):

    indices1, indices2 = gnns, datasets
    columns = dropouts
    key = lambda index1, index2, column: (index2, index1, column)

    return indices1, indices2, columns, key