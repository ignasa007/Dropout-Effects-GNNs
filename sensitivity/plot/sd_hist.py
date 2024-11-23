import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from sensitivity.utils import bin_jac_norms


L = 6
sensitivity_dir = f'./results/sensitivity/Proteins/L={L}'
agg = 'mean'

Ps = np.arange(0, 1.0, 0.1).round(decimals=1)
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

for trained, ax in zip(('untrained', 'trained'), axs):
    
    ax.set_title(f'{trained.capitalize()} Models')
    mean_jac_norms = dict()

    for P in Ps:

        sum_jac_norms = torch.zeros(L+1)
        sum_bin_sizes = torch.zeros_like(sum_jac_norms)
        molecules_count = torch.zeros_like(sum_jac_norms)

        for i_dir in os.listdir(sensitivity_dir):

            i_dir = f'{sensitivity_dir}/{i_dir}'
            if not os.path.isdir(i_dir) or 'copy' in i_dir: continue
            
            with open (f'{i_dir}/shortest_distances.pkl', 'rb') as f:
                shortest_distances = pickle.load(f)
            x_sd, y_hist = map(lambda x: x.int(), shortest_distances.unique(return_counts=True))
            y_hist = y_hist.float() / y_hist.sum()  # convert counts to ratios
            x_sd, y_hist = map(lambda x: x[x_sd <= L], (x_sd, y_hist))

            with open(f'{i_dir}/jac-norms/P={P}/{trained}.pkl', 'rb') as f:
                jac_norms = pickle.load(f)
            y_sd = bin_jac_norms(jac_norms, shortest_distances, x_sd, agg)  # expectation of jac-norms over bins
            
            sum_jac_norms[x_sd] += y_sd
            sum_bin_sizes[x_sd] += y_hist   # update ratios of edge-pairs at diff distances
            molecules_count[x_sd] += 1
                            
        mean_jac_norms[P] = sum_jac_norms / molecules_count     # expectation of binned jac-norms over graphs
        mean_bin_sizes = sum_bin_sizes / molecules_count
        
    for k, v in mean_jac_norms.items():
        if k != 0.:
            delta = v / mean_jac_norms[0] - 1.
            delta[delta>1000.] = torch.nan
            x_sd, = torch.where(~delta.isnan())
            ax.plot(x_sd, delta[x_sd], label=f'q = {k/100:.1f}')

    ax.set_xlabel('Shortest Distances')
    ax.set_ylabel('Sensitivity Change')
    # ax.set_yscale('log')
    ax.grid()
    ax.legend()

    twin_ax = ax.twinx()
    twin_ax.bar(x_sd, np.cumsum(mean_bin_sizes[x_sd]), color='black', alpha=0.2)
    twin_ax.set_ylabel('Fraction of Edge Pairs', rotation=270, labelpad=16)

fig.tight_layout()
plt.show()