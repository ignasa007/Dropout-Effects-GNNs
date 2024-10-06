from collections import defaultdict

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected, degree
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt

from signal_propagation.plot import smooth_plot


datasets = ('Proteins', 'MUTAG') # , 'PTC_MR')
Ps = (0.2, 0.4, 0.6, 0.8)
fig, axs = plt.subplots(1, len(datasets), figsize=(6.4*len(datasets), 4.8), sharey=True)
if not hasattr(axs, '__len__'): axs = (axs,)

for dataset_name, ax in zip(datasets, axs):

    dataset = TUDataset(root='./data/TUDataset', name=dataset_name.upper(), use_node_attr=True)
    beta_0s = defaultdict(list)
    ratios = defaultdict(list)

    for molecule in dataset:

        edge_index = to_undirected(molecule.edge_index)
        if connected_components(to_scipy_sparse_matrix(edge_index), directed=False, return_labels=False) > 1:
            continue
        degrees = degree(edge_index[0])
        beta_0 = torch.sum(degrees)

        for P in Ps:
            beta_P = torch.sum(degrees / (1-P**degrees))
            ratio = beta_P / beta_0
            beta_0s[P].append(beta_0)
            ratios[P].append(ratio)

    beta_0s = torch.Tensor([beta_0s[P] for P in Ps])
    ratios = torch.Tensor([ratios[P] for P in Ps])

    for P, x, y in zip(Ps, beta_0s, ratios):
        x, args = torch.sort(x)
        y = y[args]
        smooth_plot(x, y, ax=ax, label=f'P={P}', halflife=10)

    ax.set_title(dataset_name.split('_')[0], fontsize=16)
    ax.set_xlabel(r'$\beta_0$', fontsize=14)
    axs[0].set_ylabel(r'$\beta_p/\beta_0$', fontsize=14)
    # ax.set_xscale('log')
    ax.grid()
    ax.legend()

fig.tight_layout()
plt.savefig('assets/commute-times/ratio-vs-vol.png')