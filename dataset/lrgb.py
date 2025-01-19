import torch
from torch_geometric.datasets import LRGBDataset as LRGBDatasetTorch

from dataset.constants import root
from dataset.base import Inductive
from dataset.utils import normalize_features, create_loaders


class LRGBDataset(Inductive):

    def __init__(self, name: str, device: torch.device, **kwargs):

        train, val, test = (
            LRGBDatasetTorch(root=root, name=name, split=split).to(device).shuffle()
            for split in ('train', 'val', 'test')
        )

        sizes = 1500, 250, 250
        batch_size = 20 

        self.train_loader, self.val_loader, self.test_loader = create_loaders(
            tuple(map(lambda enum: enum[1][:sizes[enum[0]]], enumerate(normalize_features(train, val, test)))),
            batch_size=batch_size,
            shuffle=True
        )

        self.task_name = 'node-c'
        self.num_features = train.num_features
        self.num_classes = train.num_classes
        super(LRGBDataset, self).__init__(self.task_name, device)
    

class Pascal(LRGBDataset):
    def __init__(self, **kwargs):
        super(Pascal, self).__init__(name='PascalVOC-SP', **kwargs)