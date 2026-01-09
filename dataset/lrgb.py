from typing import Callable, Iterator
from argparse import Namespace

import torch
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.datasets import LRGBDataset as LRGBDatasetTorch

from dataset.constants import root, batch_size
from dataset.base import Inductive
from dataset.utils import create_loaders


class LRGBDataset(Inductive):

    collate_fn: Callable

    def __init__(self, name: str, config: Namespace, others: Namespace, device: torch.device):

        datasets = tuple(
            LRGBDatasetTorch(root=root, name=name, split=split).to(device)
            for split in ('train', 'val', 'test')
        )

        datasets = tuple(map(
            lambda dataset: self.rewire(dataset, config=config, others=others, device=device),
            datasets,
        ))

        self.train_loader, self.val_loader, self.test_loader = create_loaders(
            datasets,
            batch_size=batch_size,
            shuffle=True,
        )

        # For Peptides, node embeddings need to be constructed, so `num_features` is set manually in its case; see below
        if not hasattr(self, 'num_features'):
            self.num_features = datasets[0].num_features
        self.num_targets = datasets[0].num_classes
        super(LRGBDataset, self).__init__(device=device)


class Pascal(LRGBDataset):
    def __init__(self, **kwargs):
        self.task_name = 'node-c'
        super(Pascal, self).__init__(name='PascalVOC-SP', **kwargs)

class PeptidesStruct(LRGBDataset):
    
    def __init__(self, **kwargs):
    
        self.num_features = emb_dim = 16
        self.task_name = 'graph-r'
        super(PeptidesStruct, self).__init__(name='Peptides-struct', **kwargs)

        from torch_geometric.graphgym.models.encoder import AtomEncoder
        self.encoder = AtomEncoder(emb_dim=emb_dim).to(kwargs['device'])

    def parameters(self) -> Iterator[Parameter]:

        return self.encoder.parameters()

    def encode(self, graph: Data) -> Data:

        return self.encoder(graph)