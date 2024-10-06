from typing import Tuple, Dict

import torch
from torch_geometric.datasets import WebKB as WebKBTorch
from torch.optim import Optimizer

from dataset.constants import root
from dataset.base import BaseDataset
from dataset.utils import split_dataset
from model import Model


class WebKB(BaseDataset):

    def __init__(self, name: str, device: torch.device, **kwargs):

        dataset = WebKBTorch(root=f'{root}/WebKB', name=name).to(device)

        self.x = dataset.x
        self.edge_index = dataset.edge_index
        self.y = dataset.y

        indices = torch.randperm(self.x.size(0))
        self.train_mask, self.val_mask, self.test_mask = split_dataset(indices)
    
        self.task_name = 'node-c'
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(WebKB, self).__init__(self.task_name)

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
    

class Cornell(WebKB):
    def __init__(self, **kwargs):
        super(Cornell, self).__init__(name='Cornell', **kwargs)

class Texas(WebKB):
    def __init__(self, **kwargs):
        super(Texas, self).__init__(name='Texas', **kwargs)

class Wisconsin(WebKB):
    def __init__(self, **kwargs):
        super(Wisconsin, self).__init__(name='Wisconsin', **kwargs)