from typing import Tuple, Dict
import torch
from torch.optim import Optimizer
from metrics import Metrics, Classification, Regression
from model import Model

    
def set_metrics(task_name: str, num_classes: int, device: torch.device) -> Tuple[Metrics, int]:

    formatted_name = task_name.replace('_', '-').lower()
    
    output_dim = num_classes  # must be set in child classes
    if formatted_name.endswith('-c'):
        metrics = Classification(num_classes, device)
        if num_classes == 2: output_dim = 1
    elif formatted_name.endswith('-r'):
        metrics = Regression(num_classes, device)
    else:
        raise ValueError('Parameter `task_name` not identified.' +
            ' ' + f'Expected `classification` or `regression`, but got `{task_name}`.')
    
    return metrics, output_dim


class BaseDataset:

    def __init__(self, task_name: str, device: torch.device):

        self.metrics, self.output_dim = set_metrics(task_name, self.num_classes, device)
        
    def reset_metrics(self):

        return self.metrics.reset()
    
    def compute_loss(self, out, target):

        return self.metrics.compute_loss(out, target)

    def compute_metrics(self):

        return self.metrics.compute_metrics()
        
    def train(self, model, optimizer):

        raise NotImplementedError
    
    @torch.no_grad()
    def eval(self, model):

        raise NotImplementedError
    

class Transductive(BaseDataset):

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


class Inductive(BaseDataset):

    def train(self, model: Model, optimizer: Optimizer) -> Dict[str, float]:

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
    def eval(self, model: Model) -> Tuple[Dict[str, float], Dict[str, float]]:

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