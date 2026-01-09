from argparse import Namespace

import torch
from torch_geometric.datasets import Twitch as TwitchTorch

from dataset.constants import root
from dataset.base import Transductive
from dataset.utils import split_dataset, normalize_features


class Twitch(Transductive):

    def __init__(self, name: str, config: Namespace, others: Namespace, device: torch.device):

        dataset = TwitchTorch(root=f'{root}/Twitch', name=name).to(device)
        dataset = self.rewire(dataset, config=config, others=others, device=device)
        dataset, = normalize_features(dataset)
        self.graph = dataset[0]

        indices = torch.randperm(self.graph.num_nodes)
        self.train_mask, self.val_mask, self.test_mask = split_dataset(indices)

        self.task_name = 'node-c'
        self.num_features = dataset.num_features
        self.num_targets = dataset.num_classes
        super(Twitch, self).__init__(device=device)


class TwitchDE(Twitch):
    def __init__(self, **kwargs):
        super(TwitchDE, self).__init__(name='DE', **kwargs)