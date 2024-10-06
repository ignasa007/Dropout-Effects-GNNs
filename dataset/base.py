from typing import Tuple
import torch
from metrics import Metrics, Classification, Regression

    
def set_metrics(task_name: str, num_classes: int) -> Tuple[Metrics, int]:

    formatted_name = task_name.replace('_', '-').lower()
    
    output_dim = num_classes  # must be set in child classes
    if formatted_name.endswith('-c'):
        metrics = Classification(num_classes)
        if num_classes == 2: output_dim = 1
    elif formatted_name.endswith('-r'):
        metrics = Regression(num_classes)
    else:
        raise ValueError('Parameter `task_name` not identified.' +
            ' ' + f'Expected `classification` or `regression`, but got `{task_name}`.')
    
    return metrics, output_dim


class BaseDataset:

    def __init__(self, task_name: str):

        self.metrics, self.output_dim = set_metrics(task_name, num_classes=self.num_classes)
        
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