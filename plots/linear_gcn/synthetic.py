import os
import torch
from torch_geometric.utils import degree
import matplotlib.pyplot as plt

from sensitivity.utils import to_adj_mat

L = 2

# edge_index = torch.Tensor([[0, 1, 1, 1, 2, 2, 3, 3], [1, 0, 2, 3, 1, 3, 1, 2]]).type(torch.int64)
edge_index = torch.Tensor([[0, 1, 1, 2], [1, 0, 2, 1]]).type(torch.int64)
degrees = degree(edge_index[0])
A = to_adj_mat(edge_index)

ps = torch.arange(0.0, 1.0, 0.01).tolist()
ys = list()
fig, ax = plt.subplots(1, 1)

for p in ps:

    p = round(p, 2)
    diag = (1-p**(degrees+1)) / ((1-p)*(degrees+1))
    non_diag = (1 / degrees) * (1 - diag)

    non_diag = non_diag.unsqueeze(dim=1).repeat(1, degrees.size(0)) * A
    diag = torch.diag(diag)
    P_p = torch.where(diag>0., diag, non_diag)
    P_p_L = torch.matrix_power(P_p, L)

    ys.append(torch.diag(P_p_L)[:2])

ys = torch.stack(ys, dim=0).transpose(0, 1)
ax.plot(ps, ys[0], label=r'$i = 0, 2$')
ax.plot(ps, ys[1], label=r'$i = 1$')

ax.set_xlabel('DropEdge Probability', fontsize=14)
ax.set_ylabel(r'$(P^2)_{ii}$', fontsize=14)
ax.grid()
ax.legend()
fig.tight_layout()

fn = f'assets/linear-gcn/synthetic/sensitivities.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')