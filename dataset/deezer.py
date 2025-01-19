import torch
from torch_geometric.datasets import DeezerEurope as DeezerTorch

from dataset.constants import root
from dataset.base import Transductive
from dataset.utils import split_dataset


class Deezer(Transductive):

    def __init__(self, device: torch.device, **kwargs):

        dataset = DeezerTorch(root=f'{root}/Deezer').to(device)

        self.x = dataset.x
        self.edge_index = dataset.edge_index
        self.y = dataset.y

        indices = torch.randperm(self.x.size(0))
        self.train_mask, self.val_mask, self.test_mask = split_dataset(indices)
    
        self.task_name = 'node-c'
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(Deezer, self).__init__(self.task_name, device)