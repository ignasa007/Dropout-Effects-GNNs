'''
Copied from Lim et al. (2021)
URL: https://github.com/CUAI/Non-Homophily-Benchmarks/blob/main/homophily.py
'''

from warnings import filterwarnings; filterwarnings('ignore')
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops


def edge_homophily(edge_idx, labels):
    edge_index = remove_self_loops(edge_idx)[0]
    return torch.mean((labels[edge_index[0,:]] == labels[edge_index[1,:]]).float())
    
def compat_matrix(edge_idx, labels):
    edge_index = remove_self_loops(edge_idx)[0]
    src_node, targ_node = edge_index[0,:], edge_index[1,:]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max()+1
    H = torch.zeros((c,c)).to(edge_index.device)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    label_idx = torch.cat((src_label.unsqueeze(0), targ_label.unsqueeze(0)), axis=0)
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        scatter_add(torch.ones_like(add_idx).to(H.dtype), add_idx, out=H[k,:], dim=-1)
    H = H / torch.sum(H, axis=1, keepdims=True)
    return H

def new_measure(edge_index, label):
    label = label.squeeze()
    c = label.max()+1
    H = compat_matrix(edge_index, label)
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k,k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c-1
    return val


if __name__ == '__main__':

    from torch_geometric.datasets import *

    dataset = Amazon(root='./data/Amazon', name='Computers')
    print(f'Edge Homophily: {edge_homophily(dataset.edge_index, dataset.y):.3f}')
    print(f'New Homophily Measure: {new_measure(dataset.edge_index, dataset.y):.3f}')