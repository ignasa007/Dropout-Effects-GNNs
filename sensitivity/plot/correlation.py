import argparse
import os
import pickle
from collections import defaultdict
from tqdm import tqdm

import torch
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from sensitivity.utils import bin_jac_norms
from utils.format import format_dataset_name


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['Cora', 'CiteSeer', 'Proteins', 'MUTAG'])
parser.add_argument('--agg', type=str, default='mean', choices=['mean', 'sum'])
args = parser.parse_args()

L = 6
models_dir = f'./results/sensitivity/model-store/{format_dataset_name[args.dataset.lower()]}'
jac_norms_dir = f'./results/sensitivity/jac-norms-store/{format_dataset_name[args.dataset.lower()]}'

fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
fig.suptitle(f'{args.dataset}, L={L}')
metric_names = ('train_ce', 'eval_ce')

for P_dir in tqdm(os.listdir(models_dir)):
    
    P = float(P_dir.split('=')[1])
    ys = defaultdict(list)
    
    for timestamp in os.listdir(f'{models_dir}/P={P}'):

        with open(f'{models_dir}/P={P}/{timestamp}/indices.pkl', 'rb') as f:
            indices = pickle.load(f)

        # 2nd dim = L+1 because we trained a L-layer GNN, so it can reach nodes at distances from 0 and L 
        binned_jac_norms = torch.full((len(indices), L+1), torch.nan)
        for i, idx in enumerate(indices):
            with open(f'{jac_norms_dir}/i={idx}/shortest_distances.pkl', 'rb') as f:
                shortest_distances = pickle.load(f)
            x_sd = shortest_distances.unique().int()
            with open(f'{jac_norms_dir}/i={idx}/P={P}/{timestamp}/trained.pkl', 'rb') as f:
                jac_norms = pickle.load(f)
            y_sd = bin_jac_norms(jac_norms, shortest_distances, x_sd, args.agg)
            mask, = torch.where(x_sd<=L)
            binned_jac_norms[i, x_sd[mask]] = y_sd[mask]

        for metric_name in metric_names:

            with open(f'{models_dir}/P={P}/{timestamp}/{metric_name}.pkl', 'rb') as f:
                metrics = pickle.load(f)

            x = np.arange(binned_jac_norms.size(1))
            y, std, markers = list(), list(), list()
            for norms in binned_jac_norms.transpose(0, 1):
                tensor1, tensor2 = map(lambda tensor: tensor[~norms.isnan()], (norms, metrics))
                if torch.allclose(tensor1, tensor1.mean()):
                    print(P, timestamp)
                if torch.allclose(tensor2, tensor2.mean()):
                    print(P, timestamp, metric_name)
                corr = spearmanr(tensor1, tensor2)
                y.append(corr.statistic)
            ys[metric_name].append(y)

    for metric_name, ax in zip(metric_names, axs.flatten()):
        
        if not ys[metric_name]:
            print(f'Skipping {metric_name} for P = {P}')
            continue
        mean, std = np.mean(ys[metric_name], axis=0), np.std(ys[metric_name], axis=0)
        p = ax.plot(x, mean, label=f'q = {P}')
        ax.fill_between(x, mean-std, mean+std, color=p[-1].get_color(), alpha=0.2)
        # ax.scatter(x[markers], y[markers], color=p[-1].get_color())
        # ax.scatter(x[~markers], y[~markers], facecolors='none', edgecolors=p[-1].get_color())

for metric_name, ax in zip(metric_names, axs.flatten()):

    mode, error = metric_name.split('_')
    ax.set_title(f'{mode.capitalize()} {error.upper()}')
    ax.set_xlabel('Shortest Distance')
    ax.set_ylabel('Correlation with Mean Sensitivity')
    ax.grid()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor = (0, -0.15, 1, 1))
fig.tight_layout()
fn = f'./assets/correlation/Cross Entropy/{args.dataset}.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')