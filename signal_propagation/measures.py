from glob import glob
import pickle

import torch

from model import Model as Base
from sensitivity.utils import compute_shortest_distances, compute_commute_times


class Model(Base):
    
    def forward(self, x, edge_index):
    
        for mp_layer in self.message_passing:
            x = mp_layer(x, edge_index)
    
        return x

def initialize_architecture(config):

    model = Model(config)
    for mp_layer in model.message_passing:
        mp_layer.lin.weight.data = torch.ones_like(mp_layer.lin.weight.data)
        mp_layer.bias = None
    model.eval()
    
    return model


def process_graph_datum(x, edge_index, source, config, new_implementation, use_commute_time, print_out=False):

    distances = compute_shortest_distances(edge_index)
    ct_or_er = compute_commute_times(edge_index)
    if not use_commute_time:
        ct_or_er =  ct_or_er / edge_index.size(1)
    
    if new_implementation:
        mask = distances[source] <= len(config.gnn_layer_sizes)
    else:
        mask = torch.Tensor([True]*distances.size(0)).type(torch.bool)
    total_ct_or_er = torch.sum(ct_or_er[mask, source])
        
    model = initialize_architecture(config)
    out = model(x, edge_index).detach()
    if print_out:
        print(out)
    assert torch.all(out[~mask, :] == 0.)
    out = out[mask, :]

    if new_implementation:  # normalize over each feature dimension
        out = (out.abs() / out.abs().sum(dim=0, keepdims=True))
    else:                   # normalize over each node
        out = (out / out.abs().sum(dim=1, keepdims=True))
    out = torch.nan_to_num(out, nan=0.0)
    propagation_distance = (out * distances[mask, source][:, None]).sum() \
        / (out.size(1) * distances[mask, source].max())

    return total_ct_or_er.item(), propagation_distance.item()


def main(edge_index, source):

    edge_index = edge_index.type(torch.int64)
    x = torch.zeros(edge_index.max()+1, dim)
    x[source] = 1/dim

    model_dir = glob(f'./results/drop-edge/MUTAG/GCN/L=2/P=0.1/*')[0]
    with open(f'{model_dir}/config.pkl', 'rb') as f:
        config = pickle.load(f)
    config.input_dim = x.size(1)
    # we need L>1 to see the effect of connectivity
    # let's say L=1, then a K-3 graph fails to send messages between the two non-source nodes
    config.gnn_layer_sizes = [x.size(1)] * 2
    config.task_name = 'Graph-C'

    total_resistance, old_implementation = process_graph_datum(x, edge_index, source, config, False, False, False)
    total_commute_time, new_implementation = process_graph_datum(x, edge_index, source, config, True, True, False)
    print(f'Total resistance   = ${total_resistance:.2f}$ \\\\')
    print(f'Total commute time = ${total_commute_time:.2f}$ \\\\')
    print(f'Original measure   = ${old_implementation:.2f}$ \\\\')
    print(f'Our measure        = ${new_implementation:.2f}$')

if __name__ == '__main__':

    dim = 2

    print('2-chain')
    edge_index = torch.Tensor([
        [0, 1],
        [1, 0]
    ])
    main(edge_index, source=0)
    
    print('3-chain with source at a corner')
    edge_index = torch.Tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ])
    main(edge_index, source=0)
    print()

    print('3-cycle')
    edge_index = torch.Tensor([
        [0, 0, 1, 1, 2, 2],
        [1, 2, 2, 0, 0, 1]
    ])
    main(edge_index, source=0)

    print('3-chain with source in the middle')
    edge_index = torch.Tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ])
    main(edge_index, source=1)

    print('4-cycle')
    edge_index = torch.Tensor([
        [0, 0, 1, 1, 2, 2, 3, 3],
        [1, 3, 0, 2, 1, 3, 0, 2]
    ])
    main(edge_index, source=0)
    print()

    print('4-complete')
    edge_index = torch.Tensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]
    ])
    main(edge_index, source=0)
    
    print('4-cycle with a source connected to diagonally opposite nodes')
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 0, 3, 1, 4],
        [1, 0, 2, 1, 3, 2, 4, 3, 3, 0, 4, 1]
    ])
    main(edge_index, source=2)

    print('5-chain with source in the middle')
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ])
    main(edge_index, source=2)
    print()

    print('5-complete graph')
    edge_index = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
        [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]
    ])
    main(edge_index, source=0)

    print('4-cycle with a source connected to all nodes')
    edge_index = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [1, 2, 3, 4, 0, 2, 4, 0, 1, 3, 0, 2, 4, 0, 1, 3]
    ])
    main(edge_index, source=0)

    print('5-chain with the source connected to the corners')
    edge_index = torch.tensor([
        [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        [1, 2, 3, 4, 0, 2, 0, 1, 0, 4, 0, 3]
    ])
    main(edge_index, source=0)