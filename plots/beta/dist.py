from collections import defaultdict

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected, degree
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt

datasets = ('Proteins', 'MUTAG') # , 'PTC_MR')
Ps = (0.2, 0.4, 0.6, 0.8)
fig, axs = plt.subplots(1, len(datasets), figsize=(6.4*len(datasets), 4.8), sharey=True)
if not hasattr(axs, '__len__'): axs = (axs,)

for dataset_name, ax in zip(datasets, axs):

    dataset = TUDataset(root='./data/TUDataset', name=dataset_name.upper(), use_node_attr=True)
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
            ratios[P].append(ratio)

    ratios = torch.Tensor([ratios[P] for P in Ps])
    range = (ratios.min().item(), ratios.max().item())

    for P, ratio in zip(Ps, ratios):
        counts, bin_edges = torch.histogram(ratio, bins=50, range=range)
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_mids, counts/len(ratio), width=bin_edges.diff(), label=f'q = {P}', alpha=0.8)

    ax.set_title(dataset_name.split('_')[0], fontsize=16)
    ax.set_xlabel(r'$\beta^{\left(q\right)}/\beta^{\left(0\right)}$', fontsize=14)
    axs[0].set_ylabel('Proportion of Graphs', fontsize=14)

    ax.grid()
    ax.legend()

fig.tight_layout()
plt.savefig('assets/commute-times/beta-dist.png')