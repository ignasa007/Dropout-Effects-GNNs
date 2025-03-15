import warnings; warnings.filterwarnings('ignore')
import os

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree, remove_self_loops, add_remaining_self_loops, dropout_edge
from torch_geometric.utils.num_nodes import maybe_num_nodes
import matplotlib.pyplot as plt

from sensitivity.utils import to_adj_mat, compute_shortest_distances, aggregate

L = 6; ls = range(L+1)
DROPEDGE_SAMPLES = 20
dataset = Planetoid(root='./data/Planetoid', name='Cora')
ps = np.arange(0, 1, 0.1)

edge_index = remove_self_loops(dataset.edge_index)[0]
num_nodes = maybe_num_nodes(edge_index)

shortest_distances = compute_shortest_distances(edge_index, num_nodes).flatten()
degrees = degree(edge_index[1], num_nodes)  # needs to be without self-loops
x_sd = shortest_distances.unique().int()
x_sd = x_sd[torch.logical_and(0<=x_sd, x_sd<=L)]

fig, (axp, axl) = plt.subplots(1, 2, figsize=(12.8, 4.8))
data = torch.nan * torch.zeros(len(ps), len(ls))

for i, p in enumerate(ps):
        
    P_p = torch.zeros((num_nodes, num_nodes))
    for _ in range(DROPEDGE_SAMPLES):
        dropped_edge_index = add_remaining_self_loops(dropout_edge(edge_index, p, force_undirected=False)[0], num_nodes=num_nodes)[0]
        A = to_adj_mat(dropped_edge_index, num_nodes=num_nodes)
        out_deg_inv_sqrt = degree(dropped_edge_index[0], num_nodes=num_nodes).pow(-0.5)
        out_deg_inv_sqrt[out_deg_inv_sqrt == float('inf')] = 0
        in_deg_inv_sqrt = degree(dropped_edge_index[1], num_nodes=num_nodes).pow(-0.5)
        in_deg_inv_sqrt[in_deg_inv_sqrt == float('inf')] = 0
        P_p += torch.diag(in_deg_inv_sqrt) @ A @ torch.diag(out_deg_inv_sqrt)
    P_p /= DROPEDGE_SAMPLES # Expected propogation matrix under symmetric normalization

    P_p_L = torch.matrix_power(P_p, L).flatten()
    y_sd = aggregate(P_p_L, shortest_distances, x_sd, agg='mean')
    data[i] = y_sd

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

fig.tight_layout()
fn = f'./assets/linear-gcn/symmetric.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')