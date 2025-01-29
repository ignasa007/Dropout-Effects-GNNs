from dataset.base import BaseDataset
from dataset.planetoid import Cora, CiteSeer, PubMed
from dataset.qm9 import QM9
from dataset.tudataset import Proteins, PTC, Mutag, Enzymes, Reddit, IMDb, Collab
from dataset.lrgb import Pascal
from dataset.synthetic_zinc_ct import SyntheticZINC_CT
from dataset.synthetic_zinc_sd import SyntheticZINC_SD
from dataset.synthetic_mutag import SyntheticMutag
from dataset.twitch import TwitchDE
from dataset.actor import Actor
from dataset.wikipedia import Chameleon, Crocodile, Squirrel
from dataset.webkb import Cornell, Texas, Wisconsin
from dataset.deezer import Deezer


def get_dataset(dataset_name: str, **kwargs) -> BaseDataset:

    dataset_map = {
        # citation networks
        'cora': Cora, 'citeseer': CiteSeer, 'pubmed': PubMed,
        # biomed networks
        'qm9': QM9, 'proteins': Proteins, 'ptc': PTC, 'mutag': Mutag, 'enzymes': Enzymes,
        # social networks
        'reddit': Reddit, 'imdb': IMDb, 'collab': Collab,
        # synthetic networks
        'syntheticzinc_ct': SyntheticZINC_CT, 'syntheticzinc_sd': SyntheticZINC_SD, 'syntheticmutag': SyntheticMutag,
        # wikipedia networks
        'chameleon': Chameleon, 'crocodile': Crocodile, 'squirrel': Squirrel,
        # web-kb networks
        'cornell': Cornell, 'texas': Texas, 'wisconsin': Wisconsin,
        # twitch networks
        'twitchde': TwitchDE,
        # others
        'pascal': Pascal, 'actor': Actor, 'deezer': Deezer,
    }
    
    formatted_name = dataset_name.lower()
    if formatted_name not in dataset_map:
        raise ValueError(f'Parameter `dataset_name` not recognised (got `{dataset_name}`).')
    
    dataset = dataset_map.get(formatted_name)
    
    return dataset(**kwargs)