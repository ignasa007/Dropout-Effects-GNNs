from typing import Tuple, Dict

import torch
from torch_geometric.datasets import Planetoid as PlanetoidTorch
from torch.optim import Optimizer

from dataset.constants import root
from dataset.base import BaseDataset
from model import Model


class Planetoid(BaseDataset):

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
        super(Planetoid, self).__init__(self.task_name)

    def train(self, model: Model, optimizer: Optimizer) -> Dict[str, float]:

        model.train()
        
        optimizer.zero_grad()
        out = model(self.x, self.edge_index, self.train_mask)
        train_loss = self.compute_loss(out, self.y[self.train_mask])
        train_loss.backward()
        optimizer.step()

        train_metrics = self.compute_metrics()
        return train_metrics
    
    @torch.no_grad()
    def eval(self, model: Model) -> Tuple[Dict[str, float], Dict[str, float]]:

        model.eval()
        out = model(self.x, self.edge_index, mask=None)
        
        self.compute_loss(out[self.val_mask], self.y[self.val_mask])
        val_metrics = self.compute_metrics()
        self.compute_loss(out[self.test_mask], self.y[self.test_mask])
        test_metrics = self.compute_metrics()

        return val_metrics, test_metrics
    

class Cora(Planetoid):
    def __init__(self, **kwargs):
        super(Cora, self).__init__(name='Cora', **kwargs)

class CiteSeer(Planetoid):
    def __init__(self, **kwargs):
        super(CiteSeer, self).__init__(name='CiteSeer', **kwargs)

class PubMed(Planetoid):
    def __init__(self, **kwargs):
        super(PubMed, self).__init__(name='PubMed', **kwargs)