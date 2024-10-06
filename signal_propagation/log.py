import argparse
import os
from glob import glob
import pickle
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.datasets import TUDataset

from model import Model as Base
from sensitivity.utils import compute_shortest_distances, compute_commute_times


GRAPHS_SAMPLES = 200
NUM_VERTICES_SAMPLES = 10
DROPEDGE_SAMPLES = 10


class Model(Base):
    
    def forward(self, x, edge_index):
    
        for mp_layer in self.message_passing:
            x = mp_layer(x, edge_index)
    
        return x

def initialize_architecture(config):

    model = Model(config)
    for mp_layer in model.message_passing:
        mp_layer.bias = None
    model.train()
    
    return model


def process_graph_datum(datum, config, Ps, new_implementation, use_commute_time):

    distances = compute_shortest_distances(datum.edge_index)
    ct_or_er = compute_commute_times(datum.edge_index)
    if not use_commute_time:
        ct_or_er =  ct_or_er / datum.edge_index.size(1)
    
    L = len(config.gnn_layer_sizes)
    sources = np.random.choice(datum.num_nodes, min(NUM_VERTICES_SAMPLES, datum.num_nodes), replace=False)
    pairs_runaway = {P: list() for P in Ps}

    for source in sources:

        if new_implementation:
            mask = distances[source] <= L   ### consider only the L-hop neighborhood, or not, that is the question
        else:
            mask = torch.Tensor([True]*distances.size(0)).type(torch.bool)
        total_ct_or_er = torch.sum(ct_or_er[mask, source])

        x = torch.zeros_like(datum.x)
        x[source] = torch.randn_like(datum.x[source])
        x[source] = x[source].softmax(dim=-1)
        datum.x.data = x
            
        for P in Ps:

            config.drop_p = P
            model = initialize_architecture(config)

            out = torch.mean(torch.stack([
                model(datum.x, datum.edge_index).detach()
                for _ in range(DROPEDGE_SAMPLES if P>0 else 1)
            ]), dim=0).detach()
            assert torch.all(out[~mask, :] == 0.)
            out = out[mask, :]

            if new_implementation:  # normalize over each feature dimension
                out = out.abs() / out.abs().sum(dim=0, keepdims=True)
            else:                   # normalize over each node
                out = out / out.abs().sum(dim=1, keepdims=True)
                # out = out / out.sum()
            out = torch.nan_to_num(out, nan=0.0)
            propagation_distance = (out * distances[mask, source][:, None]).sum() \
                / (out.size(1) * distances[mask, source].max())
            
            pairs_runaway[P].append((total_ct_or_er, propagation_distance))

    # averaging over sampled source nodes
    pairs_runaway = {P: tuple(np.mean(pairs_runaway[P], axis=0)) for P in pairs_runaway}
    
    return pairs_runaway


def main(dataset_name, results_dir, new_implementation, use_commute_time):

    global GRAPHS_SAMPLES

    dataset = TUDataset(root='./data/TUDataset', name=dataset_name, use_node_attr=True).shuffle()
    dataset_name = dataset_name.split('_')[0]
    model_dir = glob(f'./results/drop-edge/{dataset_name}/GCN/L=2/P=0.1/*')[0]
    with open(f'{model_dir}/config.pkl', 'rb') as f:
        config = pickle.load(f)
    config.gnn_layer_sizes = [5] * 10
    config.task_name = 'Graph-C'

    Ps = np.arange(0.0, 1.0, 0.1)
    pairs = {P: list() for P in Ps}

    for datum in tqdm(dataset):
        if not GRAPHS_SAMPLES:
            break
        try:
            arch_runaway = process_graph_datum(datum, config, Ps, new_implementation, use_commute_time)
            for P in arch_runaway:
                pairs[P].append(arch_runaway[P])
            GRAPHS_SAMPLES -= 1
        except AssertionError:
            continue

    fn = f'{results_dir}/{dataset_name}.pkl'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, 'wb') as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['Proteins', 'MUTAG', 'PTC_MR'])
    parser.add_argument('--new_implementation', action='store_true')
    parser.add_argument('--old_implementation', dest='new_implementation', action='store_false')
    parser.add_argument('--use_commute_time', action='store_true')
    parser.add_argument('--use_total_resistance', dest='use_commute_time', action='store_false')
    args = parser.parse_args()

    implementation = 'new_implementation' if args.new_implementation else 'old_implementation'
    versus = 'Commute Time' if args.use_commute_time else 'Total Resistance'
    
    main(
        dataset_name=args.dataset,
        results_dir=f'./results/signal-propagation/{implementation}/{versus}',
        new_implementation=args.new_implementation,
        use_commute_time=args.use_commute_time,
    )