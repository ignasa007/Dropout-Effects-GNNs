import os
from tqdm import tqdm
import sympy
from sympy.abc import x
import matplotlib.pyplot as plt

ds = range(1, 101)
cs = [0.5, 0.8, 0.9]
lines = []

for c in cs:

    qs = list()
    for d in tqdm(ds):
        q = float(sympy.N(sympy.real_roots(d*(1-c)*(1-x)-x+x**(d+1))[-2]))
        qs.append(q)
    line, = plt.plot(ds, qs)
    lines.append(line)
    
    qs = list()
    for d in ds:
        q = (1-c)*d / (1+(1-c)*d)
        qs.append(q)
    approximation, = plt.plot(ds, qs, linestyle='--', color='black')

plt.xlabel(r'Node in-degree, $d_i$', fontsize=13)
plt.ylabel(r'Edge Dropping Probability, $q_i$', fontsize=13)

lines.append(approximation)
labels = [f'C = {c:.2f}' for c in cs] + ['Approximation']
plt.legend(lines, labels, fontsize=12)
plt.grid()

plt.tight_layout()
fn = './assets/DropSens/approximation.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)