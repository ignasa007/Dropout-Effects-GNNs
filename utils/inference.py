import pickle
from tqdm import tqdm

import torch
from torch.optim import Adam

from dataset import get_dataset, BaseDataset
from model import Model
from utils.config import parse_arguments
from utils.logger import Logger
from utils.format import *


def load_model(config_fn):

    with open(config_fn, 'rb') as f:
        config = pickle.load(config_fn)

    DEVICE = torch.device(f'cuda:{config.device_index}' if torch.cuda.is_available() and config.device_index is not None else 'cpu')
    model = Model(config=config).to(device=DEVICE)