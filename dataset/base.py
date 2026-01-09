from typing import Any, Tuple, List, Iterator
from argparse import Namespace

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer
from torch_geometric.data import Data, InMemoryDataset
from metrics import Metrics, Classification, Regression

from model import Model


def set_metrics(task_name: str, num_targets: int, device: torch.device) -> Tuple[Metrics, int]:

    formatted_name = task_name.replace('_', '-').lower()

    output_dim = num_targets    # must be set in child classes
    if formatted_name.endswith('-c'):
        metrics = Classification(num_targets, device)
        if num_targets == 2: output_dim = 1
    elif formatted_name.endswith('-r'):
        metrics = Regression(num_targets, device)
    else:
        raise ValueError('Parameter `task_name` not identified.' +
            ' ' + f'Expected `classification` or `regression`, but got `{task_name}`.')

    return metrics, output_dim


class BaseDataset:

    task_name: str
    num_features: int
    num_targets: int

    def __init__(self, device: torch.device):
        self.metrics, self.output_dim = set_metrics(self.task_name, self.num_targets, device)

    @staticmethod
    def rewire(
        dataset: InMemoryDataset,
        config: Namespace,
        others: Namespace,
        device: torch.device,
    ) -> InMemoryDataset:

        kwargs = dict(config=config, others=others, device=device)
        
        # https://github.com/pyg-team/pytorch_geometric/blob/1648a0c320f67fba8cb583dfefa14c3bc2741fc9/torch_geometric/datasets/tu_dataset.py#L202
        data_list = [dataset.get(idx) for idx in range(len(dataset))]
        dataset.data, dataset.slices = dataset.collate(data_list)
        dataset._data_list = None   # Reset cache

        return dataset

    def parameters(self) -> Iterator[Parameter]:
        yield from ()       # No parameters to optimize

    def encode(self, graph: Data) -> Data:
        return graph        # Like an embedding model, used before starting message-passing

    def forward(self, graph: Data, model: Model, **kwargs: Any) -> Tensor:
        
        graph = self.encode(graph)
        out = model(graph, **kwargs)
        
        return out

    def train(self, model: Model, optimizer: Optimizer) -> List[Tuple[str, Tensor]]:
        raise NotImplementedError

    @torch.no_grad()
    def eval(self, model: Model) -> Tuple[List[Tuple[str, Tensor]], ...]:
        raise NotImplementedError

    def reset_metrics(self):
        return self.metrics.reset()

    def compute_loss(self, out: Tensor, target: Tensor) -> Tensor:
        return self.metrics.compute_loss(out, target)

    def aggregate_metrics(self) -> List[Tuple[str, Tensor]]:
        return self.metrics.aggregate_metrics()


class Transductive(BaseDataset):

    def train(self, model: Model, optimizer: Optimizer) -> List[Tuple[str, Tensor]]:

        model.train()

        graph = self.graph
        mask = self.train_mask
        
        preds = self.forward(graph, model, mask=mask)
        optimizer.zero_grad()
        self.compute_loss(preds, graph.y[mask]).backward()
        optimizer.step()
        
        return self.aggregate_metrics()

    @torch.no_grad()
    def eval(self, model: Model) -> Tuple[List[Tuple[str, Tensor]], ...]:
        
        model.eval()

        graph = self.graph
        preds = self.forward(graph, model, mask=None)

        out = ()
        for mask in (self.val_mask, self.test_mask):
            self.compute_loss(preds[mask], graph.y[mask])
            out += (self.aggregate_metrics(),)
        
        return out


class Inductive(BaseDataset):

    def train(self, model: Model, optimizer: Optimizer) -> List[Tuple[str, Tensor]]:
        
        model.train()
        
        self.reset_metrics()    # metrics are reset upon aggregation as well
        for graph in self.train_loader:
            preds = self.forward(graph, model, mask=graph.batch)
            optimizer.zero_grad()
            self.compute_loss(preds, graph.y).backward()
            optimizer.step()
        
        return self.aggregate_metrics()

    @torch.no_grad()
    def eval(self, model: Model) -> Tuple[List[Tuple[str, Tensor]], ...]:
        
        model.eval()
        
        out = ()
        for loader in (self.val_loader, self.test_loader):
            self.reset_metrics()
            for graph in loader:
                preds = self.forward(graph, model, mask=graph.batch)
                self.compute_loss(preds, graph.y)
            out += (self.aggregate_metrics(),)
        
        return out