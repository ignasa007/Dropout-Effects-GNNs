import torch
from torch_geometric.datasets import TUDataset as TUDatasetTorch

from dataset.constants import root, batch_size
from dataset.base import Inductive
from dataset.utils import split_dataset, create_loaders


class TUDataset(Inductive):

    def __init__(self, name: str, device: torch.device, **kwargs):

        dataset = TUDatasetTorch(root=f'{root}/TUDataset', name=name, use_node_attr=True).to(device)
        dataset = dataset.shuffle()

        self.train_loader, self.val_loader, self.test_loader = create_loaders(
            ### normalize features(*split_dataset(...))?
            split_dataset(dataset),
            batch_size=batch_size,
            shuffle=True
        )

        self.task_name = 'graph-c'
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(TUDataset, self).__init__(self.task_name, device)
    

class Proteins(TUDataset):
    def __init__(self, **kwargs):
        super(Proteins, self).__init__(name='PROTEINS', **kwargs)

class PTC(TUDataset):
    def __init__(self, **kwargs):
        super(PTC, self).__init__(name='PTC_MR', **kwargs)

class MUTAG(TUDataset):
    def __init__(self, **kwargs):
        super(MUTAG, self).__init__(name='MUTAG', **kwargs)