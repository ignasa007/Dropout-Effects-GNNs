'''
Generate model samples for SyntheticZINC runs.
'''

import warnings; warnings.filterwarnings('ignore')
import os
import torch
from model import Model
from utils.config import parse_arguments

config, others = parse_arguments(return_others=True)
others.input_dim = 1
others.output_dim = 1
others.task_name = 'Graph-R'

for sample in range(1, int(others.num_samples)+1):
    model = Model(config, others)
    sample_fn = f'./results/synthetic-zinc_state-dicts/{config.gnn}/sample={sample}.pt'
    if not os.path.isfile(sample_fn):
        os.makedirs(os.path.dirname(sample_fn), exist_ok=True)
        torch.save(model.state_dict(), sample_fn)