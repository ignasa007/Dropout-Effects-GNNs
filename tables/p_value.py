import numpy as np
from scipy import stats


def values(node_datasets, graph_datasets, gnns, dropouts, args):

    assert args.node or args.graph, 'Exactly one must be true.'
    
    dropouts = tuple(dropout for dropout in dropouts if dropout != 'DropSens')
    
    if args.node:
        node_datasets = tuple(dataset for dataset in node_datasets if dataset != 'Actor')
        return node_datasets, gnns, dropouts
    elif args.graph:
        return graph_datasets, gnns, dropouts

def is_normal(samples):

    # Failed to reject the null hypothesis of normal distribution of data at 90% confidence
    return stats.shapiro(samples)[0] > 0.1

def compare_samples(base_drop_samples, best_drop_samples):

    '''
    Testing the hypothesis that NoDrop performs worse than the given Dropout method
        under the null hypothesis of equal means.
    '''

    assert is_normal(base_drop_samples) and is_normal(best_drop_samples)

    statistic, pvalue = stats.ttest_ind(
        base_drop_samples,
        best_drop_samples,
        equal_var=False,    # Dropout samples should have a higher variance
        alternative='less'  # The mean of NoDrop samples is less than the mean of BestDrop samples
    )

    return statistic, pvalue

def cell_value(base_drop_samples, best_drop_samples, best_config):

    base_drop_mean = np.mean(base_drop_samples)
    best_drop_mean = np.mean(best_drop_samples)

    _, pvalue = compare_samples(base_drop_samples, best_drop_samples)
    out = f'{100*(best_drop_mean-base_drop_mean):.3f}'
    if out[0].isdigit():
        out = f'+{out}'
        
    if pvalue > 0.1:
        out = f'\\cellcolor{{\\negative!{100*(pvalue-0.1)/0.9:.3f}}} ${out}$'
    else:
        out = f'\\cellcolor{{\\positive!{100*(0.1-pvalue)/0.1:.3f}}} ${out}$'
    
    return out

def make_key(datasets, gnns, dropouts):

    indices1, indices2 = gnns, dropouts
    columns = datasets
    key = lambda index1, index2, column: (column, index1, index2)

    return indices1, indices2, columns, key