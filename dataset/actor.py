import torch
from torch_geometric.datasets import Actor as ActorTorch
from torch_geometric.utils import to_undirected, remove_self_loops

from dataset.constants import root
from dataset.base import Transductive
from dataset.utils import split_dataset


class Actor(Transductive):

    def __init__(self, device: torch.device, **kwargs):

        dataset = ActorTorch(root=f'{root}/Actor').to(device)

        self.x = dataset.x
        ### Important to make the graph undirected
        ### TODO: GCN still not learning
        ### TODO: Should not have to make undirected
        self.edge_index = to_undirected(remove_self_loops(dataset.edge_index)[0])
        self.y = dataset.y
        
        indices = torch.randperm(self.x.size(0))
        self.train_mask, self.val_mask, self.test_mask = split_dataset(indices)

        self.task_name = 'node-c'
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(Actor, self).__init__(self.task_name, device)