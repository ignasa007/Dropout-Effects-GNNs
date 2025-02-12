import argparse
from utils.format import format_dataset_name, format_layer_name, \
    format_dropout_name, format_activation_name

def layer_sizes(args):

    out = list()
    for arg in args:
        if isinstance(arg, str) and '*' in arg:
            size, mult = map(int, arg.split('*'))
            out.extend([size]*mult)
        elif isinstance(arg, str) and arg.isdigit() or isinstance(arg, int):
            out.append(int(arg))
        else:
            raise ValueError(f'arg = {arg}, type(arg) = {type(arg)}')

    return out


def parse_arguments(return_others=False):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', type=str, required=True,
        # action=lambda dataset: format_dataset_name.get(dataset.lower()),
        help='The dataset to be trained on.'
    )
    parser.add_argument(
        '--add_self_loops', type=bool, default=True,
        help='Boolean value indicating whether to add self-loops during message passing.'
    )
    parser.add_argument(
        '--normalize', type=bool, default=True,
        help='Boolean value indicating whether to normalize edge weights during message passing.'
    )

    parser.add_argument(
        '--gnn', type=str, required=True,
        # action=lambda gnn: format_layer_name.get(gnn.lower()),
        help='The backbone model.'
    )
    parser.add_argument(
        '--gnn_layer_sizes', type=str, nargs='+', default=[16, 16],
        help="Hidden layers' sizes for the GNN."
    )
    parser.add_argument(
        '--gnn_activation', type=str, default='ReLU',
        # action=lambda activation: format_activation_name.get(activation.lower()),
        help='The non-linearity to use for message-passing.'
    )
    parser.add_argument(
        '--bias', type=bool, default=True,
        help='Boolean value indicating whether to add bias after message aggregation.'
    )

    parser.add_argument(
        '--ffn_layer_sizes', type=str, nargs='*', default=[],
        help="Hidden layers' sizes for the readout FFN."
    )
    parser.add_argument(
        '--ffn_activation', type=str, default='ReLU',
        # action=lambda activation: format_activation_name.get(activation.lower()),
        help='The non-linearity to use for readout.'
    )

    parser.add_argument(
        '--dropout', type=str, default='NoDrop',
        # action=lambda dropout: format_dropout_name.get(dropout.lower()),
        help='The dropping method.'
    )
    parser.add_argument(
        '--drop_p', type=float, default=0.0,
        help='The dropping probability used with the dropout method.'
    )

    parser.add_argument(
        '--n_epochs', type=int, default=300,
        help='Number of epochs to train the model for.'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-3,
        help='Learning rate for Adam optimizer.'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0,
        help='Weight decay for Adam optimizer.'
    )

    parser.add_argument(
        '--device_index', type=int, default=None,
        help="Index of the GPU to use; skip if you're using CPU."
    )
    parser.add_argument(
        '--test_every', type=int, default=1,
        help='Number of epochs of training to test after.\n' \
            '\tSpecial cases: -1 to test only at the last epoch.'
    )
    parser.add_argument(
        '--save_every', type=int, default=None,
        help='Number of epochs of training to save the model after.\n' \
            '\tSpecial cases: skip to never save and -1 to save at the last epoch.'
    )

    parser.add_argument(
        '--exp_dir', type=str, required=True,
        help='Directory to log the experiment in.'
    )

    config, _ = parser.parse_known_args()
    config.gnn_layer_sizes = layer_sizes(config.gnn_layer_sizes)
    config.ffn_layer_sizes = layer_sizes(config.ffn_layer_sizes)

    if not return_others:
        return config
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--gt_depth', type=int,
        help='Depth of the ground-truth function when dataset is SyntheticMutag.'
    )
    parser.add_argument(
        '--distance', type=float,
        help='Shortest distance between node pairs when dataset is SyntheticZINC_SD.\n \
            Commute times percentile when dataset is SyntheticZINC_CT.'
    )

    parser.add_argument(
        '--attention_heads', type=int,
        help='Number of attention heads when GNN is GAT.'
    )
    parser.add_argument(
        '--eps', type=float,
        help='Extra weight of self-loops when GNN is GIN.'
    )
    parser.add_argument(
        '--power_iter', type=int,
        help='Number of power iteration steps when GNN is APPNP.'
    )
    parser.add_argument(
        '--teleport_p', type=float,
        help='Teleport probability to use when GNN is APPNP.'
    )
    parser.add_argument(
        '--pooler', type=str, choices=['mean', 'add', 'max'],
        help='Method used to pool node embeddings when task is at the graph-level.'
    )

    parser.add_argument(
        '--info_loss_ratio', type=float,
        help='Ratio of information to preserve per edge when dropout is DropSens.'
    )

    parser.add_argument(
        '--model_sample', type=int,
        help='Model sample to load weights from when dataset is SyntheticZINC.'
    )
    
    others, unknown = parser.parse_known_args()

    i = 0
    while i < len(unknown):
        assert unknown[i].startswith('--')
        key = unknown[i].removeprefix('--'); i += 1
        if i == len(unknown):
            value = None
        else:
            value = list()
            while i < len(unknown) and not unknown[i].startswith('--'):
                value.append(unknown[i]); i += 1
            if len(value) == 0: value = None
            elif len(value) == 1: value = value[0]
        if not hasattr(config, key):
            setattr(others, key, value)
    
    return config, others