from argparse import Namespace

import torch
from torch_geometric.datasets import WebKB as WebKBTorch

from dataset.constants import root
from dataset.base import Transductive
from dataset.utils import split_dataset


class WebKB(Transductive):

    def __init__(self, name: str, config: Namespace, others: Namespace, device: torch.device):

        dataset = WebKBTorch(root=f'{root}/WebKB', name=name).to(device)
        dataset = self.rewire(dataset, config=config, others=others, device=device)
        self.graph = dataset[0]

        indices = torch.randperm(self.graph.num_nodes)
        self.train_mask, self.val_mask, self.test_mask = split_dataset(indices)

        self.task_name = 'node-c'
        self.num_features = dataset.num_features
        self.num_targets = dataset.num_classes
        super(WebKB, self).__init__(device=device)


class Cornell(WebKB):
    def __init__(self, **kwargs):
        super(Cornell, self).__init__(name='Cornell', **kwargs)

class Texas(WebKB):
    def __init__(self, **kwargs):
        super(Texas, self).__init__(name='Texas', **kwargs)

class Wisconsin(WebKB):
    def __init__(self, **kwargs):
        super(Wisconsin, self).__init__(name='Wisconsin', **kwargs)