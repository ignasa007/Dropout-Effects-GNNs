from typing import Dict

import torch
from torch_geometric.datasets import TUDataset as TUDatasetTorch
from torch.optim import Optimizer

from dataset.constants import root, batch_size
from dataset.base import BaseDataset
from dataset.utils import split_dataset, create_loaders
from model import Model


class TUDataset(BaseDataset):

    def __init__(self, name: str, device: torch.device, **kwargs):

        dataset = TUDatasetTorch(root=f'{root}/TUDataset', name=name, use_node_attr=True).to(device)
        dataset = dataset.shuffle()

        self.train_loader, self.val_loader, self.test_loader = create_loaders(
            ### normalize features(*split_dataset(...))?
            split_dataset(dataset),
            batch_size=batch_size,
            shuffle=True
        )

        self.task_name = 'graph-c'
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(TUDataset, self).__init__(self.task_name)

    def train(self, model: Model, optimizer: Optimizer):

        model.train()

        for batch in self.train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            train_loss = self.compute_loss(out, batch.y)
            train_loss.backward()
            optimizer.step()

        train_metrics = self.compute_metrics()
        return train_metrics
    
    @torch.no_grad()
    def eval(self, model: Model) -> Dict[str, float]:

        model.eval()
        
        for batch in self.val_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            self.compute_loss(out, batch.y)
        val_metrics = self.compute_metrics()

        for batch in self.test_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            self.compute_loss(out, batch.y)
        test_metrics = self.compute_metrics()

        return val_metrics, test_metrics
    

class Proteins(TUDataset):
    def __init__(self, **kwargs):
        super(Proteins, self).__init__(name='PROTEINS', **kwargs)

class PTC(TUDataset):
    def __init__(self, **kwargs):
        super(PTC, self).__init__(name='PTC_MR', **kwargs)

class MUTAG(TUDataset):
    def __init__(self, **kwargs):
        super(MUTAG, self).__init__(name='MUTAG', **kwargs)