import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from scipy.sparse.csgraph import connected_components


dataset = TUDataset(root='./data/TUDataset', name='Proteins', use_node_attr=True)
betas_store = list()

for molecule in dataset:
    
    edge_index = molecule.edge_index
    
    num_nodes = (edge_index.max()+1).item()
    A = torch.zeros((num_nodes, num_nodes))
    A[edge_index[0], edge_index[1]] = 1.
    if not connected_components(A, directed=False, return_labels=False) == 1:
        continue

    degrees = degree(edge_index[0], num_nodes=edge_index.max()+1)[:, None]
    beta_0 = torch.sum(degrees)
    Ps = torch.round(torch.arange(0.0, 1.0, 0.05), decimals=2)
    beta_Ps = torch.sum(degrees / (1 - torch.pow(Ps, degrees)), dim=0) / beta_0

    betas_store.append(beta_Ps)

betas_store = torch.stack(betas_store)
print(betas_store.size())

torch.save(betas_store, './results/betas_store.pkl')