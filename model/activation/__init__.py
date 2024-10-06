from torch.nn import Identity, ReLU, ELU, Sigmoid, Tanh


def get_activation(activation_name: str):

    activation_map = {
        'identiy': Identity,
        'relu': ReLU,
        'elu': ELU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
    }
    
    formatted_name = activation_name.lower()
    if formatted_name not in activation_map:
        raise ValueError(f'Parameter `activation_name` not recognised (got `{activation_name}`).')
    
    activation_class = activation_map.get(formatted_name)
    
    return activation_class