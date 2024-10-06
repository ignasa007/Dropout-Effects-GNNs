import argparse
from tqdm import tqdm

from scipy.stats import spearmanr
import torch
from torch_geometric.datasets import TUDataset
import matplotlib.pyplot as plt

from sensitivity.utils import compute_shortest_distances
from sensitivity.utils import compute_commute_times


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['Proteins', 'MUTAG', 'PTC_MR'])
args = parser.parse_args()

dataset = TUDataset('./data/TUDataset', args.dataset, use_node_attr=True)

xs = list()
ys = list()

for molecule in tqdm(dataset):

    edge_index, num_nodes = molecule.edge_index, molecule.num_nodes
    if num_nodes > 80:
        continue

    try:
        shortest_distances = compute_shortest_distances(edge_index)
        effective_resistances = compute_commute_times(edge_index) / edge_index.size(1)
    except AssertionError:
        continue

    # diameter = shortest_distances.max().item()
    # total_resistance = effective_resistances.sum().item()

    diameter = torch.mean(shortest_distances.amax(dim=1))
    total_resistance = torch.mean(effective_resistances.sum(dim=1))

    xs.append(diameter)
    ys.append(total_resistance)

plt.scatter(xs, ys, s=28, label=f'Rank Correlation = {spearmanr(xs, ys).statistic:.2f}')
plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

plt.xlabel('Diameter', fontsize=14)
plt.ylabel('Total Resistance', fontsize=14)
plt.title(args.dataset.split('_')[0], fontsize=16)

plt.grid()
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()