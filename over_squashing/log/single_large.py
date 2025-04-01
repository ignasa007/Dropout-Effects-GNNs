import warnings; warnings.filterwarnings('ignore')
import os
from tqdm import tqdm

import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
import torch
from torch_geometric.utils import subgraph, to_scipy_sparse_matrix, degree, remove_self_loops, add_remaining_self_loops

from utils.config import parse_arguments
from dataset import get_dataset
from model import Model as BaseModel
from model.message_passing.pretreatment import ModelPretreatment as BaseModelPretreatment
from over_squashing.utils import get_jacobian_norms


NODE_SAMPLES = 25
MASK_SAMPLES = 5
INIT_SAMPLES = 5

config, others = parse_arguments(return_others=True)
jac_norms_dir = f'./jac-norms/{config.dataset}'
os.makedirs(jac_norms_dir, exist_ok=True)

if config.gnn_activation == 'Identity':
    # With no activation, a bias term (even if in each layer) does not affect sensitivity to input features
    config.bias = False
if config.dropout == 'NoDrop':
    config.drop_p = 0.0
if config.drop_p == 0.0:
    INIT_SAMPLES = INIT_SAMPLES * MASK_SAMPLES
    MASK_SAMPLES = 1


class ModelPretreatment(BaseModelPretreatment):

    def pretreatment(self, num_nodes, edge_index, dtype):
        if self.add_self_loops:
            edge_index = add_remaining_self_loops(edge_index, num_nodes=num_nodes)[0]
        edge_weight = None
        if self.normalize:
            col = edge_index[1]
            deg = degree(col, num_nodes, dtype=dtype)
            # Performing asymmetric normalization because subgraph sampling changes the in-degree 
            #   of source nodes at distance L, making symmetric normalization inaccurate
            # NOTE: the in-degrees of nodes at distance <L remains the same, since all their
            #   neighbors will be at distance <=L from the target, including them in the subgraph
            # NOTE: this is an implementation hack, not a scientifically motivated decision
            deg_inv = deg.pow(-1)
            edge_weight = deg_inv[col]
        return edge_index, edge_weight


class Model(BaseModel):

    def __init__(self, edge_index, config, others):
        super(Model, self).__init__(config, others)
        for mp_layer in self.message_passing:
            mp_layer.pt = ModelPretreatment(config.add_self_loops, config.normalize)
        if config.dropout == 'DropSens':
            mp_layer.drop_strategy.init_mapper(remove_self_loops(edge_index)[0])
    
    def forward(self, mask, edge_index, x):
        for mp_layer in self.message_passing:
            x = mp_layer(x, edge_index)    
        return x if mask is None else x[mask, ...]


DEVICE = torch.device(f'cuda:{config.device_index}' if torch.cuda.is_available() and config.device_index is not None else 'cpu')
# Will put x and edge_index on device after sampling the subgraph
dataset = get_dataset(config.dataset, device=torch.device('cpu'), config=config, others=others)
others.input_dim = dataset.num_features
others.output_dim = dataset.output_dim
others.task_name = dataset.task_name

num_nodes = dataset.x.size(0)
# Doesn't matter if self-loops are not removed here since A is only used for computing connected components and shortest distances
A = to_scipy_sparse_matrix(dataset.edge_index)

# Sample nodes from the largest component
assignments = connected_components(A, return_labels=True)[1]
cc_labels, sizes = np.unique(assignments, return_counts=True)
# All indices assigned to the largest component
all_indices = np.where(assignments == cc_labels[np.argmax(sizes)])[0]
# Indices for which a directory is already created (in case we have incomplete logging for some nodes)
logged_indices = set((int(x.split('=')[1]) for x in os.listdir(jac_norms_dir)))
assert logged_indices.issubset(all_indices)
# Set the sample of nodes to be the set of nodes that have been partially logged and the remaining being random nodes
node_samples = np.random.choice(all_indices[~np.isin(all_indices, logged_indices)], NODE_SAMPLES-len(logged_indices), replace=False)
node_samples = logged_indices.union(node_samples)

for i in tqdm(node_samples):

    i_dir = f'{jac_norms_dir}/i={i}/L={len(config.gnn_layer_sizes)}'
    # Initializing here, instead of the for-loop below, to avoid having to recompute node-level drop-prob for DropSens
    model = Model(dataset.edge_index, config, others).to(device=DEVICE)

    shortest_distances = torch.from_numpy(shortest_path(A, method='D', indices=i)).int()
    # Source nodes we will compute the sensitivity of target node i to
    subset = torch.where(torch.logical_and(
        0 <= shortest_distances,    # Needed because `shortest_distances` is -inf between disconnected nodes
        shortest_distances <= len(config.gnn_layer_sizes)
    ))[0]

    fn = f'{i_dir}/shortest_distances.pkl'
    if not os.path.isfile(fn):
        os.makedirs(i_dir, exist_ok=True)
        # Save the distance from the source nodes lying within the receptive field of node i
        torch.save(shortest_distances[subset], fn)

    edge_index, _ = subgraph(subset, dataset.edge_index, relabel_nodes=True, num_nodes=dataset.x.size(0))
    # Checked the implementation -- relabelling is such that subset[i] is relabelled as i
    x = dataset.x[subset, :]
    new_i = torch.where(subset == i)[0].item()

    # Now move x and edge_index to device
    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)

    # Joint sampling model params and dropout masks
    for model_sample in range(1, INIT_SAMPLES*MASK_SAMPLES+1): 
        save_fn = f'{i_dir}/{config.gnn}/{config.dropout}/P={config.drop_p}/sample={model_sample}.pkl'
        if not os.path.exists(save_fn):
            torch.manual_seed(model_sample)
            model.reset_parameters()
            jac_norms = get_jacobian_norms(x, edge_index, model, config, mask=new_i, n_samples=1).flatten()
            os.makedirs(os.path.dirname(save_fn), exist_ok=True)
            torch.save(jac_norms, save_fn)