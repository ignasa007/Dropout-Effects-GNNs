'''
Poor naming of the file :(
    considers the sensitivity in the case the propagation matrix is D^{-1}A
'''

import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import matplotlib.pyplot as plt

from sensitivity.utils import to_adj_mat, compute_shortest_distances, bin_jac_norms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['Proteins', 'MUTAG'])
args = parser.parse_args()

L = 6; ls = range(L+1)
MOLECULE_SAMPLES = 100
dataset = TUDataset(root='./data/TUDataset', name=args.dataset.upper(), use_node_attr=True)
indices = np.where(np.array([molecule.num_nodes for molecule in dataset]) <= 60)[0]
ps = np.arange(0, 1, 0.1)

sum_sensitivity = {p: torch.zeros(MOLECULE_SAMPLES, L+1) for p in ps}
count_sensitivity = {p: torch.zeros_like(sum_sensitivity[p]) for p in ps}
fig, (axp, axl) = plt.subplots(1, 2, figsize=(12.8, 4.8))

for m in tqdm(range(MOLECULE_SAMPLES)):

    while True:
        i = np.random.choice(indices)
        edge_index = dataset[i].edge_index
        try:
            shortest_distances = compute_shortest_distances(edge_index).flatten()
        except AssertionError:
            continue
        break

    # edge_index = torch.Tensor([[0, 1, 1, 1, 2, 2, 3, 3], [1, 0, 2, 3, 1, 3, 1, 2]]).type(torch.int64)
    degrees = degree(edge_index[0])
    A = to_adj_mat(edge_index, assert_connected=False)
    x_sd = shortest_distances.unique().int()
    x_sd = x_sd[x_sd<=L]
    
    for p in ps:
            
        diag = (1-p**(degrees+1)) / ((1-p)*(degrees+1))
        non_diag = (1 / degrees) * (1 - diag)
        
        non_diag = non_diag.unsqueeze(dim=1).repeat(1, degrees.size(0)) * A
        diag = torch.diag(diag)
        P = torch.where(diag>0., diag, non_diag)
        P_l = torch.eye(P.size(0))
        P_L = P_l.clone()
        for l in range(1, L+1):
            P_l = P @ P_l
            P_L += P_l
        
        y_sd = bin_jac_norms(P_L, shortest_distances, x_sd, agg='mean')
        sum_sensitivity[p][m, x_sd] += y_sd
        count_sensitivity[p][m, x_sd] += 1


data = torch.nan * torch.zeros(len(ps), L+1)

for i, p in enumerate(ps):

    # to avoid zero division error in case no graph hits shortest distance L
    dim_to_keep = (count_sensitivity[p]>0).any(dim=0)
    x = torch.where(dim_to_keep)[0]
    y = sum_sensitivity[p][:, dim_to_keep].sum(dim=0) / count_sensitivity[p][:, dim_to_keep].sum(dim=0)
    data[i, x] = y

for p, datap in list(zip(ps, data))[::2]:
    axp.plot(ls, datap, label=f'q = {p:.1f}')
axp.set_xlabel(rf'Shortest Distance, $d_{{\mathsf{{G}}}}(i,j)$', fontsize=20)
axp.set_ylabel(rf'$\sum_{{\ell=0}}^{{6}} \left(\mathbb{{E}}\left[\hat{{A}}^{{asym}}\right]^{{\ell}}\right)_{{ij}}$', fontsize=20)
axp.set_yscale('log')
axp.grid()
axp.legend(loc='lower left', bbox_to_anchor=(0.05, 0.05), fontsize=18)

for l, datal in list(zip(ls, data.T))[::2]:
    axl.plot(ps, datal, label=fr'$d_{{\mathsf{{G}}}}(i,j) = {l}$')
axl.set_xlabel(r'DropEdge Probability, $q$', fontsize=20)
axl.set_yscale('log')
axl.grid()
axl.legend(loc='lower left', bbox_to_anchor=(0.05, 0.05), fontsize=18)

# fig.suptitle(args.dataset, fontsize=16)
fig.tight_layout()
fn = f'./assets/linear-gcn/black-extension/{args.dataset}.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')