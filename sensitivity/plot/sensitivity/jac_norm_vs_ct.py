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
parser.add_argument('--bin_size', type=int, default=40.)
args = parser.parse_args()
args.dataset = format_dataset_name[args.dataset.lower()]

L = 6
models_dir = f'./results/sensitivity/model-store/{format_dataset_name[args.dataset.lower()]}'
jac_norms_dir = f'./results/sensitivity/jac-norms-store/{format_dataset_name[args.dataset.lower()]}'

for trained in ('untrained', 'trained'):

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.8))
    xlim = 0

    for P_dir in tqdm(os.listdir(models_dir)):

        P = float(P_dir.split('=')[1])
        if P not in (0.0, 0.2, 0.4, 0.6, 0.8): continue
        mean_jac_norms = list()
        
        for timestamp in os.listdir(f'{models_dir}/{P_dir}'):

            if not os.path.isfile(f'{models_dir}/{P_dir}/{timestamp}/indices.pkl'):
                print(f'{models_dir}/{P_dir}/{timestamp}/indices.pkl')
                continue
            with open(f'{models_dir}/{P_dir}/{timestamp}/indices.pkl', 'rb') as f:
                indices = pickle.load(f)

            max_commute_time = 0
            for i, idx in enumerate(indices):
                with open(f'{jac_norms_dir}/i={idx}/commute_times.pkl', 'rb') as f:
                    commute_times = pickle.load(f)
                max_commute_time = max(max_commute_time, commute_times.max())
            
            sum_jac_norms = torch.zeros(int(torch.round(max_commute_time/args.bin_size).item())+1)
            count_jac_norms = torch.zeros_like(sum_jac_norms)

            for i, idx in enumerate(indices):
                with open(f'{jac_norms_dir}/i={idx}/commute_times.pkl', 'rb') as f:
                    commute_times = pickle.load(f)
                binned_commute_times = torch.round(commute_times/args.bin_size).flatten()
                x_ct = binned_commute_times.unique().int()
                with open(f'{jac_norms_dir}/i={idx}/{P_dir}/{timestamp}/{trained}.pkl', 'rb') as f:
                    jac_norms = pickle.load(f)
                y_ct = bin_jac_norms(jac_norms, binned_commute_times, x_ct, args.agg)
                sum_jac_norms[x_ct] += y_ct
                count_jac_norms[x_ct] += 1

            # average over molecules
            mean_jac_norms.append(sum_jac_norms/count_jac_norms)

        # average over models
        if len(mean_jac_norms) == 0:
            continue
        elif len(mean_jac_norms) == 1:
            std, mean = torch.zeros_like(mean_jac_norms[0]), mean_jac_norms[0]
        else:
            std, mean = torch.std_mean(torch.stack(mean_jac_norms, dim=0), dim=0)
        x_ct = torch.arange(mean.size(0)) * args.bin_size
        
        mask = mean-std > 0
        x_ct, mean, std = map(lambda x: x[mask], (x_ct, mean, std))
        xlim = max(xlim, x_ct.max().item())
        p = ax.plot(x_ct, mean, label=f'q = {P}')
        # ax.fill_between(x_ct, mean-std, mean+std, alpha=0.2)

    ax.set_xlim(xmax=min(600, xlim))
    ax.set_ylim(ymin=1e-6)
    ax.set_xlabel('Commute Times', fontsize=14)
    ax.set_ylabel('Mean Sensitivity', fontsize=14)
    ax.set_title(args.dataset, fontsize=14)
    ax.set_yscale('log')
    ax.grid()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor = (0, -0.07, 1, 1))
    fig.tight_layout()

    fn = f'./assets/sensitivity/commute-times/{args.dataset}/{trained}.png'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn, bbox_inches='tight')