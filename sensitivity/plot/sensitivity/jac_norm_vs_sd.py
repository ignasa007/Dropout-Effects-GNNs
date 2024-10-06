import argparse
import os
import pickle
from tqdm import tqdm

import torch
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

for trained in ('untrained', 'trained'):

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for P_dir in tqdm(os.listdir(models_dir)):

        P = float(P_dir.split('=')[1])
        mean_jac_norms = list()
        
        for timestamp in os.listdir(f'{models_dir}/{P_dir}'):

            sum_jac_norms = torch.zeros(L+1)
            count_jac_norms = torch.zeros_like(sum_jac_norms)

            with open(f'{models_dir}/{P_dir}/{timestamp}/indices.pkl', 'rb') as f:
                indices = pickle.load(f)

            for i, idx in enumerate(indices):
                with open(f'{jac_norms_dir}/i={idx}/shortest_distances.pkl', 'rb') as f:
                    shortest_distances = pickle.load(f)
                x_sd = shortest_distances.unique().int()
                with open(f'{jac_norms_dir}/i={idx}/{P_dir}/{timestamp}/{trained}.pkl', 'rb') as f:
                    jac_norms = pickle.load(f)
                y_sd = bin_jac_norms(jac_norms, shortest_distances, x_sd, args.agg)
                mask, = torch.where(x_sd<=L)
                sum_jac_norms[x_sd[mask]] += y_sd[mask]
                count_jac_norms[x_sd[mask]] += 1

            # average over molecules
            mean_jac_norms.append(sum_jac_norms/count_jac_norms)

        # average over models
        if len(mean_jac_norms) == 0:
            continue
        elif len(mean_jac_norms) == 1:
            std, mean = torch.zeros_like(mean_jac_norms[0]), mean_jac_norms[0]
        else:
            std, mean = torch.std_mean(torch.stack(mean_jac_norms, dim=0), dim=0)
        x = torch.arange(mean.size(0))
        p = ax.plot(x, mean, label=f'q = {P}')
        ax.fill_between(x, mean-std, mean+std, alpha=0.2)

    ax.set_xlabel('Shortest Distances')
    ax.set_ylabel('Mean Sensitivity')
    ax.set_yscale('log')
    ax.grid()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor = (0, -0.07, 1, 1))
    fig.tight_layout()

    fn = f'./assets/sensitivity/shortest-distance/{args.dataset}/{trained}.png'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn, bbox_inches='tight')