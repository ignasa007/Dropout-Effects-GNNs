from model.readout.base import BaseHead
from model.readout.node import NodeLevel
from model.readout.graph import GraphLevel


def get_readout(task_name: str) -> BaseHead:

    readout_map = {
        'node': NodeLevel,
        'graph': GraphLevel,
    }

    formatted_name = task_name.replace('_', '-').split('-')[0].lower()
    if formatted_name not in readout_map:
        raise ValueError(f'Parameter `task_name` not recognised (got `{task_name}`).')

    model_head = readout_map.get(formatted_name)

    return model_head