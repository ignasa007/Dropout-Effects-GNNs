'''
Poor naming of the file :(
    considers the sensitivity in the case the propagation matrix is D^{-1/2}AD^{-1/2}
'''

import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, add_self_loops, dropout_edge
import matplotlib.pyplot as plt

from sensitivity.utils import to_adj_mat, compute_shortest_distances, bin_jac_norms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['Proteins', 'MUTAG'])
args = parser.parse_args()

L = 6; ls = range(L+1)
MOLECULE_SAMPLES = 100
DROPEDGE_SAMPLES = 20
dataset = TUDataset(root='./data/TUDataset', name=args.dataset.upper(), use_node_attr=True)
indices = np.where(np.array([molecule.num_nodes for molecule in dataset]) <= 60)[0]
ps = np.arange(0, 1, 0.1)

sum_sensitivity = {p: torch.zeros(MOLECULE_SAMPLES, L+1) for p in ps}
count_sensitivity = {p: torch.zeros_like(sum_sensitivity[p]) for p in ps}
fig, (axp, axl) = plt.subplots(1, 2, figsize=(12.8, 4.8))

for m in tqdm(range(MOLECULE_SAMPLES)):

    while True:
        i = np.random.choice(indices)
        molecule = dataset[i]
        edge_index = molecule.edge_index
        try:
            shortest_distances = compute_shortest_distances(edge_index).flatten()
        except AssertionError:
            continue
        break

    # edge_index = torch.Tensor([[0, 1, 1, 1, 2, 2, 3, 3], [1, 0, 2, 3, 1, 3, 1, 2]]).type(torch.int64)
    x_sd = shortest_distances.unique().int()
    x_sd = x_sd[x_sd<=L]
    
    for p in ps:
            
        P_p = torch.zeros(molecule.num_nodes, molecule.num_nodes)
        
        for _ in range(DROPEDGE_SAMPLES):
        
            dropped_edge_index = add_self_loops(dropout_edge(edge_index, p, force_undirected=False)[0])[0]
            A = to_adj_mat(dropped_edge_index, num_nodes=molecule.num_nodes, undirected=False, assert_connected=False)
        
            out_deg_inv_sqrt = degree(dropped_edge_index[0], num_nodes=molecule.num_nodes).pow(-0.5)
            out_deg_inv_sqrt[out_deg_inv_sqrt == float('inf')] = 0
        
            in_deg_inv_sqrt = degree(dropped_edge_index[1], num_nodes=molecule.num_nodes).pow(-0.5)
            in_deg_inv_sqrt[in_deg_inv_sqrt == float('inf')] = 0
        
            P_p += torch.diag(in_deg_inv_sqrt) @ A @ torch.diag(out_deg_inv_sqrt)
        
        P_p /= DROPEDGE_SAMPLES
        P_p_L = torch.matrix_power(P_p, L).flatten()
        
        y_sd = bin_jac_norms(P_p_L, shortest_distances, x_sd, agg='mean')
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
axp.set_ylabel(rf'$\left(\mathbb{{E}}\left[\hat{{A}}^{{sym}}\right]^{{{L}}}\right)_{{ij}}$', fontsize=20)
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
fn = f'./assets/linear-gcn/symmetric/{args.dataset}.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')