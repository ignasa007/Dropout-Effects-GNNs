import warnings; warnings.filterwarnings('ignore')
import os
from argparse import Namespace
from time import perf_counter
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Twitch, Actor, TUDataset
from torch_geometric.utils import degree, dropout_edge
import matplotlib.pyplot as plt

from model.dropout import DropSens


node_datasets = (
    ('Cora', Planetoid(root='./data/Planetoid', name='Cora', split='full')),
    ('CiteSeer', Planetoid(root='./data/Planetoid', name='CiteSeer', split='full')),
    ('PubMed', Planetoid(root='./data/Planetoid', name='PubMed', split='full')),
    ('Chameleon', WikipediaNetwork(root='./data/Wikipedia', name='Chameleon')),
    ('Squirrel', WikipediaNetwork(root='./data/Wikipedia', name='Squirrel')),
    ('TwitchDE', Twitch(root='./data/Twitch', name='DE')),
    ('Actor', Actor(root='./data/Actor')),
)
graph_datasets = (
    ('Mutag', TUDataset(root='./data/TUDataset', name='MUTAG')),
    ('Proteins', TUDataset(root='./data/TUDataset', name='PROTEINS')),
    ('Enzymes', TUDataset(root='./data/TUDataset', name='ENZYMES')),
    ('Reddit', TUDataset(root='./data/TUDataset', name='REDDIT-BINARY')),
    ('IMDb', TUDataset(root='./data/TUDataset', name='IMDB-BINARY')),
    ('Collab', TUDataset(root='./data/TUDataset', name='COLLAB')),
)

dataset_names = list()
for name, dataset in node_datasets+graph_datasets:
    dataset_names.append(name)
    dataset.edge_index = dataset.edge_index.to(device='cuda:1')

runs = 10
fig, ax = plt.subplots(1, 1, figsize=(11.2, 4.8))
width = 0.8/3
displacements = np.arange(-2, 3, 2) * (width/2)

###

init_times = [[None]*(len(node_datasets)+len(graph_datasets)), [None]*(len(node_datasets)+len(graph_datasets))]
sampling_times = [[None]*(len(node_datasets)+len(graph_datasets)), [None]*(len(node_datasets)+len(graph_datasets))]

def compute_runtime(datasets, task_name):
    
    global init_times, sampling_times
    
    for name, dataset in tqdm(datasets):
    
        idx = dataset_names.index(name)
        inits, samplings = list(), list()
    
        for _ in range(runs):
            
            others = Namespace(dropsens_info_save_ratio=0.8, task_name=task_name)
            drop_sens = DropSens(0.5, others)
            
            start = perf_counter()
            drop_sens.compute_edge_dropping_probs(dataset.edge_index)
            inits.append(perf_counter()-start)
            
            start = perf_counter()
            degrees = degree(dataset.edge_index[1]).int()
            drop_sens.mapper[degrees[dataset.edge_index[1]].to('cpu')] <= torch.rand(dataset.edge_index.size(1))
            samplings.append(perf_counter()-start)
        
        init_times[0][idx], init_times[1][idx] = np.mean(inits), np.std(inits)
        sampling_times[0][idx], sampling_times[1][idx] = np.mean(samplings), np.std(samplings)

compute_runtime(node_datasets, task_name='node-c')
compute_runtime(graph_datasets, task_name='graph-c')

print(' | '.join(map(lambda x: f'${x[0]*(10**3):.0f} \\pm {x[1]*(10**3):.0f}$', zip(*init_times))))
print(' | '.join(map(lambda x: f'${x[0]*(10**3):.0f} \\pm {x[1]*(10**3):.0f}$', zip(*sampling_times))))

plt.bar(np.arange(len(dataset_names))+displacements[0], init_times[0], width=width, label='DropSens Initialization')
plt.bar(np.arange(len(dataset_names))+displacements[1], sampling_times[0], width=width, label='DropSens Sampling')

###

sampling_times = [[None]*(len(node_datasets)+len(graph_datasets)), [None]*(len(node_datasets)+len(graph_datasets))]

def compute_runtime(datasets):
    
    global init_times, sampling_times
    
    for name, dataset in datasets:
    
        idx = dataset_names.index(name)
        inits, samplings = list(), list()
    
        for _ in range(runs):
            
            start = perf_counter()
            dropout_edge(dataset.edge_index, p=0.5)
            samplings.append(perf_counter()-start)
        
        init_times[0][idx], init_times[1][idx] = np.mean(inits), np.std(inits)
        sampling_times[0][idx], sampling_times[1][idx] = np.mean(samplings), np.std(samplings)

compute_runtime(node_datasets+graph_datasets)

print(' | '.join(map(lambda x: f'${x[0]*(10**3):.0f} \\pm {x[1]*(10**3):.0f}$', zip(*sampling_times))))
plt.bar(np.arange(len(dataset_names))+displacements[2], sampling_times[0], width=width, label='DropEdge Sampling')

plt.yscale('log')
plt.xticks(np.arange(len(dataset_names)), dataset_names, rotation=30, fontsize=15)
plt.ylabel('Computation Time (s)', fontsize=18)
plt.legend(fontsize=15)
plt.grid()

fig.tight_layout()
fn = f'./assets/DropSens/init_sampling_time.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)