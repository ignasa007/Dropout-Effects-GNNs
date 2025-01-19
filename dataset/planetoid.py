from typing import Tuple, Dict

import torch
from torch_geometric.datasets import Planetoid as PlanetoidTorch
from torch.optim import Optimizer

from dataset.constants import root
from dataset.base import Transductive
from model import Model


class Planetoid(Transductive):

    def __init__(self, name: str, device: torch.device, **kwargs):

        dataset = PlanetoidTorch(root=f'{root}/Planetoid', name=name, split='full').to(device)

        # don't normalize features since they are indicator variables

        self.x = dataset.x
        self.edge_index = dataset.edge_index
        self.y = dataset.y
        
        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask

        self.task_name = 'node-c'
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(Planetoid, self).__init__(self.task_name, device)
    

class Cora(Planetoid):
    def __init__(self, **kwargs):
        super(Cora, self).__init__(name='Cora', **kwargs)

class CiteSeer(Planetoid):
    def __init__(self, **kwargs):
        super(CiteSeer, self).__init__(name='CiteSeer', **kwargs)

class PubMed(Planetoid):
    def __init__(self, **kwargs):
        super(PubMed, self).__init__(name='PubMed', **kwargs)