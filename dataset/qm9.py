from argparse import Namespace

import torch
from torch_geometric.datasets import QM9 as QM9Torch

from dataset.constants import root, batch_size
from dataset.base import Inductive
from dataset.utils import split_dataset, normalize_features, normalize_labels, create_loaders


class QM9(Inductive):

    def __init__(self, config: Namespace, others: Namespace, device: torch.device):

        dataset = QM9Torch(root=f'{root}/QM9').to(device)
        dataset = self.rewire(dataset, config=config, others=others, device=device)

        # Need to shuffle here because `split_dataset` does not shuffle
        self.train_loader, self.val_loader, self.test_loader = create_loaders(
            normalize_labels(*normalize_features(*split_dataset(dataset.shuffle()))),
            batch_size=batch_size,
            shuffle=True,
        )

        self.task_name = 'graph-r'
        self.num_features = dataset.num_features
        self.num_targets = dataset.num_classes
        super(QM9, self).__init__(device=device)