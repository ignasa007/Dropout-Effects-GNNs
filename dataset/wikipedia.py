import torch
from torch_geometric.datasets import WikipediaNetwork as WikipediaTorch
from torch_geometric.utils import to_undirected, remove_self_loops

from dataset.constants import root
from dataset.base import Transductive
from dataset.utils import split_dataset


class Wikipedia(Transductive):

    def __init__(self, name: str, device: torch.device, **kwargs):

        dataset = WikipediaTorch(root=f'{root}/Wikipedia', name=name).to(device)

        self.x = dataset[0].x
        ### Important to make the graph undirected, GCN does not learn without it
        ### TODO: Should not have to make undirected
        self.edge_index = to_undirected(remove_self_loops(dataset[0].edge_index)[0])
        self.y = dataset[0].y

        indices = torch.randperm(self.x.size(0))
        self.train_mask, self.val_mask, self.test_mask = split_dataset(indices)

        self.task_name = 'node-c'
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(Wikipedia, self).__init__(self.task_name, device)
    

class Chameleon(Wikipedia):
    def __init__(self, **kwargs):
        super(Chameleon, self).__init__(name='chameleon', **kwargs)

class Crocodile(Wikipedia):
    def __init__(self, **kwargs):
        super(Crocodile, self).__init__(name='crocodile', **kwargs)

class Squirrel(Wikipedia):
    def __init__(self, **kwargs):
        super(Squirrel, self).__init__(name='squirrel', **kwargs)