import torch
from torch_geometric.datasets import Twitch as TwitchTorch

from dataset.constants import root
from dataset.base import Transductive
from dataset.utils import split_dataset, normalize_features


class Twitch(Transductive):

    def __init__(self, name: str, device: torch.device, **kwargs):

        dataset = TwitchTorch(root=f'{root}/Twitch', name=name).to(device)
        dataset, = normalize_features(dataset)

        self.x = dataset.x
        self.edge_index = dataset.edge_index
        self.y = dataset.y
        
        indices = torch.randperm(dataset.x.size(0))
        self.train_mask, self.val_mask, self.test_mask = split_dataset(indices)

        self.task_name = 'node-c'
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(Twitch, self).__init__(self.task_name, device)
    

class TwitchDE(Twitch):
    def __init__(self, **kwargs):
        super(TwitchDE, self).__init__(name='DE', **kwargs)