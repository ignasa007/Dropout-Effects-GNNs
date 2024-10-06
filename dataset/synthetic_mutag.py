from argparse import Namespace
from typing import Dict

import torch
from torch_geometric.datasets import TUDataset as TUDatasetTorch
from torch.optim import Optimizer

from dataset.constants import root, batch_size
from dataset.base import BaseDataset
from dataset.utils import split_dataset, create_loaders
from model import Model


root = f'{root}/Synthetics'


class PreTransform:

    def __init__(self, config: Namespace, others: Namespace):

        config.gnn_layer_sizes = [config.gnn_layer_sizes[0]] * int(others.gt_depth)
        config.drop_p = 0.0
        
        self.model = Model(config, others)
        state_dict = torch.load(f'{root}/MUTAG/L={others.gt_depth}/ckpt-500.pt')
        self.model.load_state_dict(state_dict)
        
        self.model.train()
        
    def __call__(self, data):
        
        data.y = self.model(data.x, data.edge_index)

        return data


class SyntheticMUTAG(BaseDataset):

    def __init__(self, config: Namespace, others: Namespace, device: torch.device):

        config, others = map(lambda x: Namespace(**vars(x)), (config, others))

        others.task_name = self.task_name = 'node-r'
        others.input_dim = self.num_features = 7
        others.output_dim = self.num_classes = 1

        dataset = TUDatasetTorch(
            root=root,
            name='MUTAG',
            use_node_attr=True,
            pre_transform=PreTransform(config, others)
        ).to(device)
        dataset = dataset.shuffle()

        self.train_loader, self.val_loader, self.test_loader = create_loaders(
            split_dataset(dataset),
            batch_size=batch_size,
            shuffle=True
        )
        
        super(SyntheticMUTAG, self).__init__(self.task_name)

    def train(self, model: Model, optimizer: Optimizer):

        model.train()

        for batch in self.train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            train_loss = self.compute_loss(out, batch.y)
            train_loss.backward()
            optimizer.step()

        train_metrics = self.compute_metrics()
        return train_metrics
    
    @torch.no_grad()
    def eval(self, model: Model) -> Dict[str, float]:

        model.eval()
        
        for batch in self.val_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            self.compute_loss(out, batch.y)
        val_metrics = self.compute_metrics()

        for batch in self.test_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            self.compute_loss(out, batch.y)
        test_metrics = self.compute_metrics()

        return val_metrics, test_metrics