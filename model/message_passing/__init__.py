from model.message_passing.gcn import GCNLayer
from model.message_passing.gat import GATLayer
from model.message_passing.appnp import APPNPLayer


def get_layer(layer_name: str):

    layer_map = {
        'gcn': GCNLayer,
        'gat': GATLayer,
        'appnp': APPNPLayer,
    }
    
    formatted_name = layer_name.lower()
    if formatted_name not in layer_map:
        raise ValueError(f'Parameter `layer_name` not recognised (got `{layer_name}`).')
    
    layer_class = layer_map.get(formatted_name)
    
    return layer_class