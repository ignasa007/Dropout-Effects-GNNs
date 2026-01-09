from argparse import Namespace

import torch
from torch_geometric.datasets import TUDataset as TUDatasetTorch

from dataset.constants import root, batch_size
from dataset.base import Inductive
from dataset.utils import split_dataset, create_loaders
from model import Model


root = f'{root}/Synthetics'


class PreTransform:

    def __init__(self, config: Namespace, others: Namespace):

        config.gnn_layer_sizes = [config.gnn_layer_sizes[0]] * int(others.syn_mutag_gt_depth)
        config.drop_p = 0.0

        self.model = Model(config, others)
        state_dict = torch.load(f'{root}/Mutag/L={others.syn_mutag_gt_depth}/ckpt-500.pt')
        self.model.load_state_dict(state_dict)

        self.model.eval()

    def __call__(self, data):

        data.y = self.model(data.x, data.edge_index)

        return data


class SyntheticMutag(Inductive):

    def __init__(self, config: Namespace, others: Namespace, device: torch.device):

        others.task_name = self.task_name = 'node-r'
        others.input_dim = self.num_features = 7
        others.output_dim = self.num_targets = 1

        dataset = TUDatasetTorch(
            root=root,
            name='Mutag',
            use_node_attr=True,
            pre_transform=PreTransform(config, others)
        ).to(device)

        dataset = self.rewire(dataset, config=config, others=others, device=device)

        # Need to shuffle here because `split_dataset` does not shuffle
        self.train_loader, self.val_loader, self.test_loader = create_loaders(
            split_dataset(dataset.shuffle()),
            batch_size=batch_size,
            shuffle=True
        )

        super(SyntheticMutag, self).__init__(device=device)