import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from sensitivity.utils import bin_jac_norms


L = 6
sensitivity_dir = f'./results/sensitivity/Proteins/L={L}'

# max_commute_time = 0.
# for i_dir in os.listdir(sensitivity_dir):
#     if 'copy' in i_dir: continue
#     i_dir = f'{sensitivity_dir}/{i_dir}'
#     with open (f'{i_dir}/commute_times.pkl', 'rb') as f:
#         commute_times = pickle.load(f)
#     max_commute_time = max(max_commute_time, commute_times.max())

BIN_SIZE = 40.
Ps = np.arange(0, 1.0, 0.1)
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

for trained, ax in zip(('untrained', 'trained'), axs):
    
    ax.set_title(f'{trained.capitalize()} Models')
    mean_jac_norms = dict()

    for P in Ps:

        sum_jac_norms = torch.zeros(int(torch.round(max_commute_time/BIN_SIZE).item())+1)
        count_jac_norms = torch.zeros_like(sum_jac_norms)

        for i_dir in os.listdir(sensitivity_dir):

            i_dir = f'{sensitivity_dir}/{i_dir}'
            if not os.path.isdir(i_dir) or 'copy' in i_dir: continue
            
            with open (f'{i_dir}/commute_times.pkl', 'rb') as f:
                commute_times = pickle.load(f)
            binned_commute_times = torch.round(commute_times/BIN_SIZE) * BIN_SIZE
            binned_commute_times = binned_commute_times.flatten()
            x_ct = binned_commute_times.unique()

            with open(f'{i_dir}/jac-norms/{trained}/p={P}.pkl', 'rb') as f:
                jac_norms = pickle.load(f)
            y_ct = bin_jac_norms(jac_norms, binned_commute_times, x_ct)
            
            indices = (x_ct/BIN_SIZE).int()
            sum_jac_norms[indices] += y_ct
            count_jac_norms[indices] += 1
                            
        end = torch.where(count_jac_norms>0.)[0][-1]
        sum_jac_norms = sum_jac_norms[:end+1]
        count_jac_norms = count_jac_norms[:end+1]

        mean_jac_norms[P] = sum_jac_norms / count_jac_norms
        
    for k, v in mean_jac_norms.items():
        if k != 0.:
            delta = 100 * (v / mean_jac_norms[0] - 1.)
            delta[delta>1000.] = torch.nan
            ax.plot(torch.arange(v.size(0))*BIN_SIZE, delta, label=f'q = {k/100:.1f}')

    ax.set_xlabel('Commute Times')
    ax.set_ylabel('Sensitivity Change (%)')
    # ax.set_yscale('log')
    ax.grid()
    ax.legend()

fig.tight_layout()
plt.show()