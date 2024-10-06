from typing import Tuple, Dict

import torch
from torch_geometric.datasets import QM9 as QM9Torch
from torch.optim import Optimizer

from dataset.constants import root, batch_size
from dataset.base import BaseDataset
from dataset.utils import split_dataset, normalize_features, normalize_labels, create_loaders
from model import Model


class QM9(BaseDataset):

    def __init__(self, device: torch.device, **kwargs):

        dataset = QM9Torch(root=f'{root}/QM9').to(device)
        dataset = dataset.shuffle()

        self.train_loader, self.val_loader, self.test_loader = create_loaders(
            normalize_labels(*normalize_features(*split_dataset(dataset))),
            batch_size=batch_size,
            shuffle=True
        )

        self.task_name = 'graph-r'
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(QM9, self).__init__(self.task_name)

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