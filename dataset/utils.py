import torch
from torch_geometric.loader import DataLoader

from dataset.constants import Splits, batch_size


def split_dataset(dataset, train_split=Splits.train_split, val_split=Splits.val_split, test_split=Splits.test_split):

    train_end = int(train_split*len(dataset))
    val_end = train_end + int(val_split*len(dataset))

    train, val, test = dataset[:train_end], dataset[train_end:val_end], dataset[val_end:]

    return train, val, test


def normalize_features(train, *others):

    out = ()

    std, mean = torch.std_mean(train.x, dim=0, keepdim=True)
    train.x = (train.x - mean) / std
    out += (train,)

    if not isinstance(others, (tuple, list)):
        raise TypeError(f'Expected `others` to be an instance of List or Tuple, but received {type(others)}.')
    for other in others:
        other.x = (other.x - mean) / std
        out += (other,)

    return out
    
    
def normalize_labels(train, *others):

    # TODO: 1. controversial choice, 2. some labels in QM9 still way too large -- clip?

    out = ()

    std, mean = torch.std_mean(train.y, dim=0, keepdim=True)
    train.y = (train.y - mean) / std
    out += (train,)

    if not isinstance(others, (tuple, list)):
        raise TypeError(f'Expected `others` to be an instance of List or Tuple, but received {type(others)}.')
    for other in others:
        other.y = (other.y - mean) / std
        out += (other,)

    return out


def create_loaders(splits, batch_size=batch_size, shuffle=True):

    out = ()
    
    for split in splits:
        out += (DataLoader(split, batch_size=batch_size, shuffle=shuffle),)
    
    return out