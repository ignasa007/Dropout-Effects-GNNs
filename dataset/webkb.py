import torch
from torch_geometric.datasets import WebKB as WebKBTorch

from dataset.constants import root
from dataset.base import Transductive
from dataset.utils import split_dataset


class WebKB(Transductive):

    def __init__(self, name: str, device: torch.device, **kwargs):

        dataset = WebKBTorch(root=f'{root}/WebKB', name=name).to(device)

        self.x = dataset.x
        self.edge_index = dataset.edge_index
        self.y = dataset.y

        indices = torch.randperm(self.x.size(0))
        self.train_mask, self.val_mask, self.test_mask = split_dataset(indices)
    
        self.task_name = 'node-c'
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(WebKB, self).__init__(self.task_name, device)
    

class Cornell(WebKB):
    def __init__(self, **kwargs):
        super(Cornell, self).__init__(name='Cornell', **kwargs)

class Texas(WebKB):
    def __init__(self, **kwargs):
        super(Texas, self).__init__(name='Texas', **kwargs)

class Wisconsin(WebKB):
    def __init__(self, **kwargs):
        super(Wisconsin, self).__init__(name='Wisconsin', **kwargs)