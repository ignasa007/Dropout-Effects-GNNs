format_dataset_name = {
    'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed',
    'qm9': 'QM9', 'proteins': 'Proteins', 'ptc': 'PTC', 'mutag': 'MUTAG',
    'syntheticzinc_ct': 'SyntheticZINC_CT', 'syntheticzinc_sd': 'SyntheticZINC_SD', 'syntheticmutag': 'SyntheticMUTAG',
    'chameleon': 'Chameleon', 'crocodile': 'Crocodile', 'squirrel': 'Squirrel',
    'cornell': 'Cornell', 'texas': 'Texas', 'wisconsin': 'Wisconsin',
    'twitchde': 'TwitchDE',
    'pascal': 'Pascal', 'actor': 'Actor', 'deezer': 'Deezer',
}

format_task_name = {
    'node-c': 'Node-C', 'node_c': 'Node-C',
    'graph-c': 'Graph-C', 'graph_c': 'Graph-C',
    'graph-r': 'Graph-R', 'graph_r': 'Graph-R',
}

format_layer_name = {
    'gcn': 'GCN',
    'gat': 'GAT',
    'appnp': 'APPNP',
}

format_dropout_name = {
    'nodrop': 'NoDrop',
    'dropout': 'Dropout',
    'drop-edge': 'DropEdge', 'dropedge': 'DropEdge',
    'drop-node': 'DropNode', 'dropnode': 'DropNode',
    'drop-message': 'DropMessage', 'dropmessage': 'DropMessage',
    'drop-gnn': 'DropGNN', 'dropgnn': 'DropGNN',
    'drop-agg': 'DropAgg', 'dropagg': 'DropAgg',
    'drop-sens': 'DropSens', 'dropsens': 'DropSens',
}

format_activation_name = {
    'identiy': 'Identity',
    'relu': 'ReLU',
    'elu': 'ELU',
    'sigmoid': 'Sigmoid',
    'tanh': 'Tanh',
}

class FormatEpoch:

    def __init__(self, n_epochs: int):
        self.adj_len = len(str(n_epochs))

    def __call__(self, epoch: int):
        return str(epoch).rjust(self.adj_len, '0')