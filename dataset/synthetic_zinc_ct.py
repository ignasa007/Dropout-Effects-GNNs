from argparse import Namespace
import os
import random
from tqdm import tqdm
import pickle

import torch
from torch_geometric.datasets import ZINC as ZINCTorch

from dataset.constants import root, batch_size
from dataset.base import Inductive
from dataset.synthetic_zinc_sd import Transform_SD
from dataset.utils import CustomDataset, create_loaders


root = f'{root}/Synthetics'


class Transform_CT(Transform_SD):

    def save_node_pairs(self, root, alpha, split):

        fn = f'{root}/node-pairs-sd/alpha={alpha}/{split}.pkl'
        if os.path.isfile(fn): 
            with open(fn, 'rb') as f:
                node_pairs = pickle.load(f)
            return node_pairs

        from sensitivity.utils import compute_commute_times

        dataset = ZINCTorch(root=root, subset=True, split=split)
        node_pairs = list()

        # For each molecule, sample a node pair separated by `distance` hops
        for datum in tqdm(dataset):
            commute_times = compute_commute_times(datum.edge_index)
            quantile = torch.quantile(commute_times.flatten(), alpha, interpolation='nearest')
            choices = torch.where(commute_times == quantile)                # Tuple[row indices, column indices]
            try:
                sample = random.randint(0, choices[0].size(0)-1)            # Int in range [0, num_mathces-1]
                node_pair = list(map(lambda x: x[sample].item(), choices))  # List[row, column]
            except ValueError:
                node_pair = None                                            # No pair separated by `distance` hops
            node_pairs.append(node_pair)

        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, 'wb') as f:
            pickle.dump(node_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

        return node_pairs


class SyntheticZINC_CT(Inductive):

    def __init__(self, device: torch.device, others: Namespace, **kwargs):

        assert others.pooler == 'max', f"For SyntheticZINC, the `pooler` argument must be 'max'."

        zinc_root = f'{root}/ZINC'
        datasets, sizes = list(), (None, None, None)
        for split, size in zip(('train', 'val', 'test'), sizes):
            # Save node pairs separated by `distance` hops
            transform = Transform_CT(zinc_root, others.alpha, split)
            dataset = ZINCTorch(root=zinc_root, subset=True, split=split)
            dataset = enumerate(dataset)
            if size is not None:
                random.shuffle(dataset)
                dataset = dataset[:size]
            # Create node-level features, and graph-level labels
            data_list = [transform(index, datum) for index, datum in dataset]
            # Filter out molecules with no two nodes separated by `distance` hops
            data_list = [datum.to(device) for datum in data_list if datum is not None]
            datasets.append(CustomDataset(data_list))
        train, val, test = datasets
        
        self.train_loader, self.val_loader, self.test_loader = create_loaders(
            (train, val, test),
            batch_size=batch_size,
            shuffle=True
        )

        self.task_name = 'graph-r'
        self.num_features = 1
        self.num_classes = 1
        super(SyntheticZINC_CT, self).__init__(self.task_name, device)