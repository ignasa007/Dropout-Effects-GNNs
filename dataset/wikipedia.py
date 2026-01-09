from argparse import Namespace

import torch
from torch_geometric.datasets import WikipediaNetwork as WikipediaTorch
from torch_geometric.utils import to_undirected, remove_self_loops

from dataset.constants import root
from dataset.base import Transductive
from dataset.utils import split_dataset


class Wikipedia(Transductive):

    def __init__(self, name: str, config: Namespace, others: Namespace, device: torch.device):

        dataset = WikipediaTorch(root=f'{root}/Wikipedia', name=name).to(device)
        dataset = self.rewire(dataset, config=config, others=others, device=device)
        self.graph = dataset[0]
        
        ### Important to make the graph undirected, GCN does not learn without it
        ### TODO: Should not have to make undirected
        self.graph.edge_index = to_undirected(remove_self_loops(self.graph.edge_index)[0])

        indices = torch.randperm(self.graph.num_nodes)
        self.train_mask, self.val_mask, self.test_mask = split_dataset(indices)

        self.task_name = 'node-c'
        self.num_features = dataset.num_features
        self.num_targets = dataset.num_classes
        super(Wikipedia, self).__init__(device=device)


class Chameleon(Wikipedia):
    def __init__(self, **kwargs):
        super(Chameleon, self).__init__(name='chameleon', **kwargs)

class Crocodile(Wikipedia):
    def __init__(self, **kwargs):
        super(Crocodile, self).__init__(name='crocodile', **kwargs)

class Squirrel(Wikipedia):
    def __init__(self, **kwargs):
        super(Squirrel, self).__init__(name='squirrel', **kwargs)