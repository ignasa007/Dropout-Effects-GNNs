import pickle

import torch
from torch.func import jacrev

from model import Model as Base


class Model(Base):
    
    def forward(self, i, edge_index, x):
    
        for mp_layer in self.message_passing:
            x = mp_layer(x, edge_index)
    
        return self.readout(x, mask=i)


def get_jacobian_norms(x, edge_index, i, dir_name, n_samples, use_trained):

    with open(f'{dir_name}/config.pkl', 'rb') as f:
        config = pickle.load(f)

    model = Model(config)
    if use_trained:
        state_dict = torch.load(f'{dir_name}/ckpt-400.pt')
        model.load_state_dict(state_dict)
    model.train()

    jacobians = torch.zeros((others.output_dim, x.size(0), others.input_dim))
    n_samples = n_samples if config.drop_p > 0. else 1
    for _ in range(n_samples):
        jacobians += jacrev(model, argnums=2)(i, edge_index, x)
    jacobians /= n_samples
    jacobian_norms = jacobians.transpose(0, 1).flatten(start_dim=1).norm(dim=1, p=1)

    return jacobian_norms