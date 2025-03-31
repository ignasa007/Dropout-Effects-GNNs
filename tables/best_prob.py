def cell_value(base_drop_samples, best_drop_samples, best_config):
    
    if not hasattr(best_config, '__len__'):
        best_config = (best_config,)
    
    out = ', '.join(map(str, (config in best_config if config is not None)))

    return out