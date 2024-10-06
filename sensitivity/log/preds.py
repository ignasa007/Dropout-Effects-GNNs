import os
import pickle
import argparse
from tqdm import tqdm

import torch

from model import Model
from utils.format import format_dataset_name


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['Cora', 'CiteSeer', 'Proteins', 'MUTAG'])
args = parser.parse_args()

NODE_SAMPLES = 100
MODEL_SAMPLES = 10
models_dir = f'./results/sensitivity/model-store/{format_dataset_name[args.dataset.lower()]}'
jac_norms_dir = f'./results/sensitivity/jac-norms-store/{format_dataset_name[args.dataset.lower()]}'
assert os.path.isdir(jac_norms_dir)


if args.dataset in ('Cora', 'CiteSeer'):
    
    from torch_geometric.datasets import Planetoid
    
    dataset = Planetoid(root='./data/Planetoid', name=format_dataset_name[args.dataset.lower()], split='full')
    indices = [int(i_dir.split('=')[1]) for i_dir in os.listdir(jac_norms_dir)]
    
    input = dataset
    mask = indices
    target = input.y[mask]

elif args.dataset in ('Proteins', 'MUTAG'):

    from torch_geometric.datasets import TUDataset
    from dataset.utils import normalize_features
    from torch_geometric.loader.dataloader import Collater

    dataset = TUDataset(root='./data/TUDataset', name=format_dataset_name[args.dataset.lower()], use_node_attr=True)
    dataset, = normalize_features(dataset)
    indices = [int(i_dir.split('=')[1]) for i_dir in os.listdir(jac_norms_dir)]

    input = Collater(dataset)(dataset[indices])
    mask = input.batch
    target = input.y

if dataset.num_classes == 2:

    from torch.nn import BCEWithLogitsLoss

    nonlinearity = torch.sigmoid
    ce_loss = lambda logits, target: BCEWithLogitsLoss(reduction='none')(logits, target.float())
    mae_loss = lambda logits, target: torch.abs(nonlinearity(logits) - target)

else:
    
    from torch.nn import CrossEntropyLoss
    
    nonlinearity = lambda probs: torch.softmax(probs, dim=-1)
    ce_loss = CrossEntropyLoss(reduction='none')
    mae_loss = lambda logits, target: torch.abs(1 - nonlinearity(logits)[torch.arange(target.size(0)), target])

for P_dir in tqdm(os.listdir(models_dir)):
    
    P = float(P_dir.split('=')[1])
    P_dir = f'{models_dir}/{P_dir}'
    
    for timestamp in os.listdir(P_dir):
    
        model_dir = f'{P_dir}/{timestamp}'
        with open(f'{model_dir}/config.pkl', 'rb') as f:
            config = pickle.load(f)
        model = Model(config)
        state_dict = torch.load(f'{model_dir}/ckpt-400.pt')
        model.load_state_dict(state_dict)
    
        # averaging logits to improve confidence calibration, since GNNs are usually underconfident
        # https://ceur-ws.org/Vol-3215/19.pdf
        n_samples = MODEL_SAMPLES if P > 0. else 1
        model.train()
        logits = torch.zeros(len(indices), config.output_dim)
        for _ in range(n_samples):
            logits += model(input.x, input.edge_index, mask).detach()
        logits = logits.squeeze() / n_samples
        train_ce = ce_loss(logits, target)
        train_mae = mae_loss(logits, target)

        model.eval()
        logits = model(input.x, input.edge_index, mask).detach().squeeze()
        eval_ce = ce_loss(logits, target)
        eval_mae = mae_loss(logits, target)

        with open(f'{model_dir}/indices.pkl', 'wb') as f:
            pickle.dump(indices, f, pickle.HIGHEST_PROTOCOL)
        with open(f'{model_dir}/train_ce.pkl', 'wb') as f:
            pickle.dump(train_ce, f, pickle.HIGHEST_PROTOCOL)
        with open(f'{model_dir}/train_mae.pkl', 'wb') as f:
            pickle.dump(train_mae, f, pickle.HIGHEST_PROTOCOL)
        with open(f'{model_dir}/eval_ce.pkl', 'wb') as f:
            pickle.dump(eval_ce, f, pickle.HIGHEST_PROTOCOL)
        with open(f'{model_dir}/eval_mae.pkl', 'wb') as f:
            pickle.dump(eval_mae, f, pickle.HIGHEST_PROTOCOL)