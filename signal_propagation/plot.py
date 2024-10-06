import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smooth_plot(x, y=None, ax=None, label='', halflife=10):
    
    y_int = y if y is not None else x
    
    x_ewm = pd.Series(y_int).ewm(halflife=halflife)
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    if y is None:
        ax.plot(x_ewm.mean(), label=label, color=color)
        # ax.fill_between(np.arange(x_ewm.mean().shape[0]), x_ewm.mean() + x_ewm.std(), x_ewm.mean() - x_ewm.std(), color=color, alpha=0.15)
    else:
        ax.plot(x, x_ewm.mean(), label=label, color=color)
        # ax.fill_between(x, y_int + x_ewm.std(), y_int - x_ewm.std(), color=color, alpha=0.15)


def main(results_dir, assets_dir, dataset_name, versus, halflife):

    with open(f'{results_dir}/{dataset_name}.pkl', 'rb') as f:
        pairs = pickle.load(f)

    Ps = (0.0, 0.8)
    fig, ax = plt.subplots(1, 1)
    
    for P in Ps:
        data = np.array(pairs[P]).T
        data = data[:, data[0].argsort()]
        data = (data - data.min(axis=1, keepdims=True)) / (data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True))
        smooth_plot(*data, ax=ax, label=f'P={P:.1f}', halflife=halflife)

    ax.set_xlabel(versus, fontsize=14)
    ax.set_ylabel('Signal Propagation', fontsize=14)
    ax.set_title(dataset_name, fontsize=16)
    ax.grid()
    ax.legend()
    fig.tight_layout()

    fn = f'{assets_dir}/propagation-distance/{dataset_name}.png'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['Proteins', 'MUTAG', 'PTC_MR'])
    parser.add_argument('--new_implementation', action='store_true')
    parser.add_argument('--old_implementation', dest='new_implementation', action='store_false')
    parser.add_argument('--use_commute_time', action='store_true')
    parser.add_argument('--use_total_resistance', dest='use_commute_time', action='store_false')
    parser.add_argument('--halflife', type=int, default=20)
    args = parser.parse_args()

    implementation = 'new_implementation' if args.new_implementation else 'old_implementation'
    versus = 'Commute Time' if args.use_commute_time else 'Total Resistance'

    dataset_name = args.dataset.split('_')[0]
    results_dir = f'./results/signal-propagation/{implementation}/{versus}'
    assets_dir = results_dir.replace('results', 'assets')
    
    main(
        dataset_name=dataset_name,
        results_dir=results_dir,
        assets_dir=assets_dir,
        halflife=args.halflife,
        versus=versus,
    )