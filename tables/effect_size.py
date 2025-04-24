import numpy as np


def values(node_datasets, graph_datasets, gnns, dropouts, args):

    datasets = node_datasets + graph_datasets
    dropouts = tuple(dropout for dropout in dropouts if dropout != 'DropSens')
    
    return datasets, gnns, dropouts

def color_effect_size(value):

    if value < -0.65:      # Between -inf and -0.65, includes -0.8
        return '\\cellcolor{\\negative!80}'
    elif value < -0.35:    # Between -0.65 and -0.35, includes -0.5
        return '\\cellcolor{\\negative!50}'
    elif value < -0.10:    # Between -0.35 and -0.10, includes -0.2
        return '\\cellcolor{\\negative!20}'
    elif value < +0.10:    # Between -0.10 and +0.10, includes 0.0
        return '\\cellcolor{white!00}'
    elif value < +0.35:    # Between +0.10 and +0.35, includes +0.2
        return '\\cellcolor{\\positive!20}'
    elif value < +0.65:    # Between +0.35 and +0.65, includes +0.5
        return '\\cellcolor{\\positive!50}'
    else:                  # Between +0.65 and +inf, includes +0.8
        return '\\cellcolor{\\positive!80}'

def cell_value(base_drop_samples, best_drop_samples, best_config):
    
    base_drop_mean, base_drop_std = np.mean(base_drop_samples), np.std(base_drop_samples, ddof=1)
    best_drop_mean, best_drop_std = np.mean(best_drop_samples), np.std(best_drop_samples, ddof=1)
    
    s_pool = np.sqrt((
        (len(best_drop_samples)-1) * best_drop_std**2 + 
        (len(base_drop_samples)-1) * base_drop_std**2
    ) / (len(best_drop_samples) + len(base_drop_samples) - 2))
    cohens_d = (best_drop_mean-base_drop_mean) / s_pool
    hedges_g = (1 - 3 / (4*(len(best_drop_samples)+len(base_drop_samples))-9)) * cohens_d
    
    out = f'{hedges_g:.3f}'
    if out[0].isdigit():
        out = f'+{out}'
    
    return f'{color_effect_size(float(out))} ${out}$'

def make_key(datasets, gnns, dropouts):

    indices1, indices2 = datasets, gnns
    columns = dropouts
    key = lambda index1, index2, column: (index1, index2, column)

    return indices1, indices2, columns, key