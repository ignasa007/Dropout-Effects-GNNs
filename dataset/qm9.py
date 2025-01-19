import torch
from torch_geometric.datasets import QM9 as QM9Torch

from dataset.constants import root, batch_size
from dataset.base import Inductive
from dataset.utils import split_dataset, normalize_features, normalize_labels, create_loaders


class QM9(Inductive):

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
        super(QM9, self).__init__(self.task_name, device)