import os
import time
from tqdm import tqdm

import numpy as np
import sympy
from sympy.abc import x
import matplotlib.pyplot as plt

# Limit all relevant thread usage
os.environ["OMP_NUM_THREADS"] = "1"         # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"    # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"         # Intel MKL
os.environ["NUMEXPR_NUM_THREADS"] = "1"     # NumExpr

ds = range(1, 101)
cs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
reps = 1

fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, sharex=True, figsize=(6.4, 4.8),
    gridspec_kw={'height_ratios': [1, 1]}
)

times = np.zeros((reps*len(cs), len(ds)))
i = 0
for _ in range(reps):
    for c in cs:
        for j, d in tqdm(enumerate(ds)):
            start = time.perf_counter()
            float(sympy.N(sympy.real_roots(d*(1-c)*(1-x)-x+x**(d+1))[-2]))
            times[i, j] += time.perf_counter() - start
        i += 1
mean, std = np.mean(times, axis=0), np.std(times, axis=0, ddof=1)
print(' | '.join(map(lambda x: f'${x[0]*(10**2):.3f} \\pm {x[1]*(10**2):.3f}$', zip(mean[[0, 1, 4, 9, 19, 49, 99]], std[[0, 1, 4, 9, 19, 49, 99]]))))
ax_top.plot(ds, mean, color='blue', label='Exact')
ax_top.fill_between(ds, mean-std, mean+std, color='blue', alpha=0.2)

times = np.zeros((reps*len(cs), len(ds)))
i = 0
for _ in range(reps):
    for c in cs:
        for j, d in tqdm(enumerate(ds)):
            start = time.perf_counter()
            (1-c)*d / (1+(1-c)*d)
            times[i, j] += time.perf_counter() - start
        i += 1
mean, std = np.mean(times, axis=0), np.std(times, axis=0, ddof=1)
print(' | '.join(map(lambda x: f'${x[0]*(10**6):.3f} \\pm {x[1]*(10**6):.3f}$', zip(mean[[0, 1, 4, 9, 19, 49, 99]], std[[0, 1, 4, 9, 19, 49, 99]]))))
ax_bottom.plot(ds, mean, color='green', label='Approximation')
ax_bottom.fill_between(ds, mean-std, mean+std, color='green', alpha=0.2)

# Diagonal lines to show break
d = .015  # size of diagonal lines in axes coordinates
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
ax_top.plot((-d, +d), (-d, +d), **kwargs)           # top-left
ax_top.plot((1-d, 1+d), (-d, +d), **kwargs)         # top-right
kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1-d, 1+d), **kwargs)      # bottom-left
ax_bottom.plot((1-d, 1+d), (1-d, 1+d), **kwargs)    # bottom-right
ax_bottom.yaxis.get_offset_text().set_x(-0.1)

# Combine legend entries
handles_top, labels_top = ax_top.get_legend_handles_labels()
handles_bot, labels_bot = ax_bottom.get_legend_handles_labels()
ax_top.legend(handles_top+handles_bot, labels_top+labels_bot, loc='best', fontsize=12)

ax_top.tick_params(axis='x', which='both', length=0)
yticks = ax_top.get_yticks()
ax_top.set_yticks([tick for tick in yticks if tick != 0])
ax_top.set_ylim(ax_bottom.get_ylim()[1], ax_top.get_ylim()[1])
ax_bottom.set_xlabel(r'Node in-degree, $d_i$', fontsize=13)
fig.supylabel(r'$q_i$ computation time (s)', fontsize=13, x=0.04)
ax_top.grid()
ax_bottom.grid()

fig.tight_layout()
fig.subplots_adjust(hspace=0.12)
fn = './assets/DropSens/compute_time.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)