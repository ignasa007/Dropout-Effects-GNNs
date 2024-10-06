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
from torch_geometric.utils import degree, add_self_loops, dropout_edge, dropout_node
import matplotlib.pyplot as plt

from sensitivity.utils import to_adj_mat, compute_shortest_distances, bin_jac_norms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['Proteins', 'MUTAG'])
args = parser.parse_args()

L = 6
MOLECULE_SAMPLES = 100
DROPEDGE_SAMPLES = 20
dataset = TUDataset(root='./data/TUDataset', name=args.dataset.upper(), use_node_attr=True)
indices = np.where(np.array([molecule.num_nodes for molecule in dataset]) <= 100)[0]
ps = np.arange(0, 1, 0.1)


def no_drop(edge_index, num_nodes, p):
    dropped_edge_index = add_self_loops(edge_index)[0]
    A = to_adj_mat(dropped_edge_index, num_nodes=num_nodes, undirected=False, assert_connected=False)
    in_deg_inv_sqrt = degree(dropped_edge_index[1], num_nodes=num_nodes).pow(-1)
    in_deg_inv_sqrt[in_deg_inv_sqrt == float('inf')] = 0
    sample = torch.diag(in_deg_inv_sqrt) @ A
    return sample

def drop_edge(edge_index, num_nodes, p):
    dropped_edge_index = add_self_loops(dropout_edge(edge_index, p, force_undirected=False)[0])[0]
    A = to_adj_mat(dropped_edge_index, num_nodes=num_nodes, undirected=False, assert_connected=False)
    in_deg_inv_sqrt = degree(dropped_edge_index[1], num_nodes=num_nodes).pow(-1)
    in_deg_inv_sqrt[in_deg_inv_sqrt == float('inf')] = 0
    sample = torch.diag(in_deg_inv_sqrt) @ A
    return sample

def drop_node(edge_index, num_nodes, p):
    edge_index = add_self_loops(edge_index)[0]
    A = to_adj_mat(edge_index, num_nodes=num_nodes, undirected=False, assert_connected=False)
    in_deg_inv_sqrt = degree(edge_index[1], num_nodes=num_nodes).pow(-1)
    in_deg_inv_sqrt[in_deg_inv_sqrt == float('inf')] = 0
    node_mask = torch.bernoulli((1-p)*torch.ones(num_nodes))
    sample = (1/(1-p)) * node_mask.unsqueeze(0) * (torch.diag(in_deg_inv_sqrt) @ A)
    return sample

def drop_agg(edge_index, num_nodes, p):
    node_mask = torch.bernoulli((1-p)*torch.ones(num_nodes)).bool(); edge_mask = node_mask[edge_index[1]]
    dropped_edge_index = add_self_loops(edge_index[:, edge_mask])[0]
    A = to_adj_mat(dropped_edge_index, num_nodes=num_nodes, undirected=False, assert_connected=False)
    in_deg_inv_sqrt = degree(dropped_edge_index[1], num_nodes=num_nodes).pow(-1)
    in_deg_inv_sqrt[in_deg_inv_sqrt == float('inf')] = 0
    sample = torch.diag(in_deg_inv_sqrt) @ A
    return sample

def drop_gnn(edge_index, num_nodes, p):
    dropped_edge_index = add_self_loops(dropout_node(edge_index, p)[0])[0]
    A = to_adj_mat(dropped_edge_index, num_nodes=num_nodes, undirected=False, assert_connected=False)
    in_deg_inv_sqrt = degree(dropped_edge_index[1], num_nodes=num_nodes).pow(-1)
    in_deg_inv_sqrt[in_deg_inv_sqrt == float('inf')] = 0
    sample = torch.diag(in_deg_inv_sqrt) @ A
    return sample


fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

for name, drop in zip(('NoDrop', 'DropEdge', 'DropNode', 'DropAgg', 'DropGNN'), (no_drop, drop_edge, drop_node, drop_agg, drop_gnn)):

    sum_sensitivity = torch.zeros(len(ps), MOLECULE_SAMPLES)

    for m in tqdm(range(MOLECULE_SAMPLES)):

        while True:
            i = np.random.choice(indices)
            molecule = dataset[i]
            edge_index = molecule.edge_index
            try:
                shortest_distances = compute_shortest_distances(edge_index).flatten()
            except AssertionError:
                continue
            if shortest_distances.max() < L:
                continue
            break
        
        for i, p in enumerate(ps):
                
            P = torch.zeros(molecule.num_nodes, molecule.num_nodes)
            for _ in range(DROPEDGE_SAMPLES):
                sample = torch.eye(molecule.num_nodes)
                for l in range(L):
                    sample @= drop(edge_index, molecule.num_nodes, p)
                P += sample
            P /= DROPEDGE_SAMPLES
            
            y_sd = bin_jac_norms(P.flatten(), shortest_distances, (L,), agg='mean')
            sum_sensitivity[i, m] += y_sd.item()

    mean_sensitivity = sum_sensitivity.mean(dim=1)
    mask = mean_sensitivity > 0.
    ax.plot(ps[mask], mean_sensitivity[mask], label=name)

ax.set_xlabel(r'Drop Probability, $q$', fontsize=20)
ax.set_yscale('log')
ax.grid()
ax.legend(loc='lower left', bbox_to_anchor=(0.05, 0.05), fontsize=18)

fig.tight_layout()
fn = f'./assets/linear-gcn/compare-drop/{args.dataset}.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')