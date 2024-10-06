from model.readout.base import BaseHead
from model.readout.node_r import NodeRegression
from model.readout.node_c import NodeClassification
from model.readout.graph_r import GraphRegression
from model.readout.graph_c import GraphClassification 


def get_head(task_name: str) -> BaseHead:

    head_map = {
        'node-r': NodeRegression,
        'node-c': NodeClassification,
        'graph-r': GraphRegression,
        'graph-c': GraphClassification,
    }

    formatted_name = task_name.replace('_', '-').lower()
    if formatted_name not in head_map:
        raise ValueError(f'Parameter `task_name` not recognised (got `{task_name}`).')
    
    model_head = head_map.get(formatted_name)
    
    return model_head