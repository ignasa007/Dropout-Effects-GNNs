from argparse import Namespace

import torch
from torch_geometric.datasets import DeezerEurope as DeezerTorch

from dataset.constants import root
from dataset.base import Transductive
from dataset.utils import split_dataset


class Deezer(Transductive):

    def __init__(self, config: Namespace, others: Namespace, device: torch.device):

        dataset = DeezerTorch(root=f'{root}/Deezer').to(device)
        dataset = self.rewire(dataset, config=config, others=others, device=device)
        self.graph = dataset[0]

        indices = torch.randperm(self.graph.num_nodes)
        self.train_mask, self.val_mask, self.test_mask = split_dataset(indices)

        self.task_name = 'node-c'
        self.num_features = dataset.num_features
        self.num_targets = dataset.num_classes
        super(Deezer, self).__init__(device=device)