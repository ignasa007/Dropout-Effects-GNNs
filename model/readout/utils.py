from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


def get_pooler(pooler_name: str):

    pooler_map = {
        'mean': global_mean_pool,
        'add': global_add_pool,
        'max': global_max_pool,
    }

    formatted_name = pooler_name.lower()
    if formatted_name not in pooler_map:
        raise ValueError(f'Parameter `pooler_name` not recognised (got `{pooler_name}`).')
    
    pooler = pooler_map.get(formatted_name)
    
    return pooler