if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')

from argparse import Namespace
import os
import random
from tqdm import tqdm
import pickle

import torch
from torch_geometric.datasets import ZINC as ZINCTorch

from dataset.constants import root, batch_size
from dataset.base import Inductive
from dataset.utils import create_loaders


root = f'{root}/ZINC'
batch_size = 250    # Divisible by 1k


class SyntheticZINC(Inductive):

    def __init__(self, node_pairs_fn: str, device: torch.device, others: Namespace, **kwargs):

        self.distance = self.process_args(others)
        
        splits, sizes = ('train', 'val', 'test'), (None, None, None)
        datasets = (
            self.make_dataset(node_pairs_fn, split, size, device)
            for split, size in zip(splits, sizes)
        )
        self.train_loader, self.val_loader, self.test_loader = create_loaders(
            datasets,
            batch_size=batch_size,
            shuffle=True,
        )

        self.task_name = 'graph-r'
        self.num_features = 1
        self.num_classes = 1
        super(SyntheticZINC, self).__init__(self.task_name, device)

    def process_args(self, others: Namespace):
        
        assert others.pooler == 'max', f'For SyntheticZINC, the `pooler` argument must be `max`.'

    def make_dataset(self, node_pairs_fn, split, size, device):

        dataset = ZINCTorch(root=root, subset=True, split=split)
        dataset = list(enumerate(dataset))
        if size is not None:
            random.shuffle(dataset)
            dataset = dataset[:size]
        
        # Get sampled node pairs separated by `distance` hops
        node_pairs = self.get_node_pairs(node_pairs_fn, split)
        # Create node-level features, and graph-level labels
        data_list = [self.make_features_and_labels(datum, node_pairs[index]) for index, datum in dataset]
        # Filter out molecules with no two nodes separated by `distance`
        data_list = [datum.to(device) for datum in data_list if datum is not None]
        # TODO: Create custom dataset (just to be able to use GPUs efficiently)
        # dataset = CustomDataset(data_list)
        
        return data_list

    def get_node_pairs(self, node_pairs_fn: str, split: str):

        fn = f'{root}/{node_pairs_fn}/distance={self.distance}/{split}.pkl'
        if os.path.isfile(fn): 
            with open(fn, 'rb') as f:
                node_pairs = pickle.load(f)
            return node_pairs

        dataset = ZINCTorch(root=root, subset=True, split=split)
        node_pairs = list()

        # For each molecule, sample a node pair separated by `distance`
        for datum in tqdm(dataset):
            choices = self.get_node_pair_choices(datum.edge_index)          # Tuple[row indices, column indices]
            try:
                sample = random.randint(0, choices[0].size(0)-1)            # Int in range [0, num_mathces-1]
                node_pair = list(map(lambda x: x[sample].item(), choices))  # List[row, column]
            except ValueError:
                node_pair = None                                            # No pair separated by `distance`
            node_pairs.append(node_pair)

        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, 'wb') as f:
            pickle.dump(node_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

        return node_pairs

    def make_features_and_labels(self, datum, node_pair):
                
        datum.x = torch.zeros_like(datum.x, dtype=torch.float)  # Set all node features to 0
        if node_pair is not None:
            features = torch.rand(len(node_pair))               # Sample random features
            datum.x[node_pair, :] = features.unsqueeze(1)       # Set features for the pair of nodes
            datum.y = torch.tanh(features.sum())                # Set graph-level label
            return datum
        else:
            return None

    def get_node_pair_choices(self, edge_index):

        raise NotImplementedError


class SyntheticZINC_SD(SyntheticZINC):

    def __init__(self, device: torch.device, others: Namespace, **kwargs):

        super(SyntheticZINC_SD, self).__init__(self.__class__.__name__, device, others)

    def process_args(self, others: Namespace):

        super(SyntheticZINC_SD, self).process_args(others)
        distance = float(others.distance)
        assert distance >= 0

        return distance

    def get_node_pair_choices(self, edge_index):

        global compute_shortest_distances
        from sensitivity.utils import compute_shortest_distances

        shortest_distances = compute_shortest_distances(edge_index) # Tensor(|E|x|E|)
        choices = torch.where(shortest_distances == self.distance)  # Tuple[row indices, column indices]

        return choices


class SyntheticZINC_CT(SyntheticZINC):

    def __init__(self, device: torch.device, others: Namespace, **kwargs):

        super(SyntheticZINC_CT, self).__init__(self.__class__.__name__, device, others)

    def process_args(self, others: Namespace):

        super(SyntheticZINC_CT, self).process_args(others)
        distance = float(others.distance)
        assert 0 <= distance <= 1

        return distance
    
    def get_node_pair_choices(self, edge_index):

        global compute_commute_times
        from sensitivity.utils import compute_commute_times

        commute_times = compute_commute_times(edge_index)   # Tensor(|E|x|E|)
        quantile = torch.quantile(commute_times.flatten(), q=self.distance, interpolation='nearest')
        choices = torch.where(commute_times == quantile)    # Tuple[row indices, column indices]

        return choices


if __name__ == '__main__':

    import argparse
    import os
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=float, required=True)
    args = parser.parse_args()

    device = torch.device('cpu')
    distances = list(map(lambda x: round(x, str(args.step)[::-1].find('.')), np.arange(0.0, 1+args.step, args.step)))

    for distance in distances:
        others = Namespace(distance=distance, pooler='max')
        SyntheticZINC_CT(device, others)