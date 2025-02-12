import torch
from torch_geometric.datasets import TUDataset as TUDatasetTorch

from dataset.constants import root, batch_size
from dataset.base import Inductive
from dataset.utils import split_dataset, create_loaders


def pre_transform(datum):
    
    # Cannot learn datasets Mutag, Enzymes and Proteins without features (F1 score = 0)
    if datum.x is None:
        # Following FoSR (Karhadkar et al., 2022) and GTR (Black et al., 2023)
        # https://github.com/kedar2/FoSR/blob/1a7360c2c77c42624bdc7ffef1490a2eb0a8afd0/run_graph_classification.py#L27
        datum.x = torch.ones((datum.num_nodes, 1))

    return datum


class TUDataset(Inductive):

    def __init__(self, name: str, device: torch.device, **kwargs):

        dataset = TUDatasetTorch(
            root=f'{root}/TUDataset',
            name=name,
            use_node_attr=True,
            pre_transform=pre_transform,
        ).to(device)
        dataset = dataset.shuffle()
        
        # (80, 10, 10) splits and batch_size=64 following Karhadkar et al. (2022)
        self.train_loader, self.val_loader, self.test_loader = create_loaders(
            ### normalize features(*split_dataset(...))?
            split_dataset(dataset, train_split=0.8, val_split=0.1, test_split=0.1),
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

class Mutag(TUDataset):
    def __init__(self, **kwargs):
        super(Mutag, self).__init__(name='MUTAG', **kwargs)

class Enzymes(TUDataset):
    def __init__(self, **kwargs):
        super(Enzymes, self).__init__(name='ENZYMES', **kwargs)

class Reddit(TUDataset):
    def __init__(self, **kwargs):
        super(Reddit, self).__init__(name='REDDIT-BINARY', **kwargs)

class IMDb(TUDataset):
    def __init__(self, **kwargs):
        super(IMDb, self).__init__(name='IMDB-BINARY', **kwargs)

class Collab(TUDataset):
    def __init__(self, **kwargs):
        super(Collab, self).__init__(name='COLLAB', **kwargs)