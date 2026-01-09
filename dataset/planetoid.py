from argparse import Namespace

import torch
from torch_geometric.datasets import Planetoid as PlanetoidTorch

from dataset.constants import root
from dataset.base import Transductive


class Planetoid(Transductive):

    def __init__(self, name: str, config: Namespace, others: Namespace, device: torch.device):

        dataset = PlanetoidTorch(root=f'{root}/Planetoid', name=name, split='full').to(device)
        dataset = self.rewire(dataset, config=config, others=others, device=device)
        # don't normalize features since they are indicator variables
        self.graph = dataset[0]
        
        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask

        self.task_name = 'node-c'
        self.num_features = dataset.num_features
        self.num_targets = dataset.num_classes
        super(Planetoid, self).__init__(device=device)


class Cora(Planetoid):
    def __init__(self, **kwargs):
        super(Cora, self).__init__(name='Cora', **kwargs)

class CiteSeer(Planetoid):
    def __init__(self, **kwargs):
        super(CiteSeer, self).__init__(name='CiteSeer', **kwargs)

class PubMed(Planetoid):
    def __init__(self, **kwargs):
        super(PubMed, self).__init__(name='PubMed', **kwargs)