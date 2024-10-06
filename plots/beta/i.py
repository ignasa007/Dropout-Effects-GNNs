import os
import torch
import matplotlib.pyplot as plt

Ps = torch.arange(0, 0.94, 0.01)
degrees = torch.arange(1, 10, 2)
ys = degrees[:,None] / (1-Ps[None,:]**degrees[:,None])
yticks = torch.arange(0, 25, 5).tolist()

fig, axs = plt.subplots(1, 3, figsize=(15,4))

ax = axs[0]
for degree, y in zip(degrees, ys):
    ax.plot(Ps, y, label=rf'$d_i = {degree}$')

ax.set_yticks(yticks, yticks)
ax.set_xlabel('DropEdge Probability', fontsize=14)
ax.set_ylabel(r'$\beta_i^{(q)}$', fontsize=14)
ax.grid()
ax.legend()

ax = axs[1]
for degree, y in zip(degrees, ys-degrees[:,None]):
    ax.plot(Ps, y, label=rf'$d_i = {degree}$')

yticks = yticks[:-1]
ax.set_yticks(yticks, yticks)
ax.set_xlabel('DropEdge Probability', fontsize=14)
ax.set_ylabel(r'$\beta_i^{(q)}-\beta_i^{(0)}$', fontsize=14)
ax.grid()
ax.legend()

ax = axs[2]
for degree, y in zip(degrees, ys/degrees[:,None]):
    ax.plot(Ps, y, label=rf'$d_i = {degree}$')

ax.set_yticks(yticks, yticks)
ax.set_xlabel('DropEdge Probability', fontsize=14)
ax.set_ylabel(r'$\beta_i^{(q)}/\beta_i^{(0)}$', fontsize=14)
ax.grid()
ax.legend()

fig.tight_layout()
fn = f'./assets/commute-times/beta-vs-prob.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')