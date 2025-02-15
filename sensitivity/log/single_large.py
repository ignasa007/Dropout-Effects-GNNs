import warnings; warnings.filterwarnings('ignore')
import os
from tqdm import tqdm

import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
import torch
from torch_geometric.utils import subgraph, to_scipy_sparse_matrix

from utils.config import parse_arguments
from dataset import get_dataset
from model import Model as Base
from sensitivity.utils import get_jacobian_norms


NODE_SAMPLES = 20
MASK_SAMPLES = 5
INIT_SAMPLES = 5

config, others = parse_arguments(return_others=True)
jac_norms_dir = f'./jac-norms/{config.dataset}'
os.makedirs(jac_norms_dir, exist_ok=True)

if config.dropout == 'NoDrop':
    config.drop_p = 0.0 
if config.drop_p == 0.0:
    INIT_SAMPLES = INIT_SAMPLES * MASK_SAMPLES
    MASK_SAMPLES = 1


class Model(Base):
    
    def forward(self, mask, edge_index, x):
    
        for mp_layer in self.message_passing:
            x = mp_layer(x, edge_index)
    
        return x if mask is None else x[mask, ...]    # self.readout(x, mask=mask)


DEVICE = torch.device(f'cuda:{config.device_index}' if torch.cuda.is_available() and config.device_index is not None else 'cpu')
# will put x and edge_index on device after sampling the subgraph
dataset = get_dataset(config.dataset, device=torch.device('cpu'), config=config, others=others)
others.input_dim = dataset.num_features
others.output_dim = dataset.output_dim
others.task_name = dataset.task_name

num_nodes = dataset.x.size(0)
A = to_scipy_sparse_matrix(dataset.edge_index)

# sample nodes from the largest component
assignments = connected_components(A, return_labels=True)[1]
cc_labels, sizes = np.unique(assignments, return_counts=True)
# all indices assigned to the largest component
all_indices = np.where(assignments == cc_labels[np.argmax(sizes)])[0]
# indices for which a directory is already created (in case we have incomplete logging for some nodes)
logged_indices = set((int(x.split('=')[1]) for x in os.listdir(jac_norms_dir)))
# set the sample of nodes to be the set of nodes that have been partially logged and the remaining being random nodes
node_samples = np.random.choice(all_indices[~np.isin(all_indices, logged_indices)], NODE_SAMPLES-len(logged_indices), replace=False)
node_samples = logged_indices.union(node_samples)

for i in tqdm(node_samples):

    i_dir = f'{jac_norms_dir}/i={i}/L={len(config.gnn_layer_sizes)}'
    model = Model(config, others).to(device=DEVICE)     # initializing here to avoid having to reset DropSens

    shortest_distances = torch.from_numpy(shortest_path(A, method='D', indices=i)).int()
    # source nodes we will compute the sensitivity of representation of node i to
    subset = torch.where(torch.logical_and(0 <= shortest_distances, shortest_distances <= len(config.gnn_layer_sizes)))[0]
    fn = f'{i_dir}/shortest_distances.pkl'
    if not os.path.exists(fn):
        os.makedirs(i_dir, exist_ok=True)
        # save the distance from the source nodes lying within the receptive field of node i
        torch.save(shortest_distances[subset], fn)

    edge_index, _ = subgraph(subset, dataset.edge_index, relabel_nodes=True, num_nodes=dataset.x.size(0))
    # checked the implementation -- relabelling is such that subset[i] is relabelled as i
    x = dataset.x[subset, :]
    new_i = torch.where(subset == i)[0].item()

    # now move x and edge_index to device
    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)

    for init_sample in range(1, INIT_SAMPLES*MASK_SAMPLES+1):   # joint sampling 

        model.reset_parameters()
        save_fn = f'{i_dir}/{config.gnn}/{config.dropout}/P={config.drop_p}/sample={init_sample}.pkl'
        if not os.path.exists(save_fn):
            jac_norms = get_jacobian_norms(x, edge_index, new_i, model, 1, config, others).flatten()
            os.makedirs(os.path.dirname(save_fn), exist_ok=True)
            torch.save(jac_norms, save_fn)