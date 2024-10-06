import torch
import matplotlib.pyplot as plt


fig, (axs_bar, axs_plot, axs_violin) = plt.subplots(1, 3, figsize=(18, 4.8))

betas_store = torch.load('./results/betas_store.pkl').transpose(0, 1)
indices = torch.arange(0, betas_store.size(0), 2)
Ps = indices/betas_store.size(0)

for i, P in zip(indices, Ps):

    betas = betas_store[i]
    
    heights, bin_edges = torch.histogram(betas, bins=10)
    widths = torch.diff(bin_edges)
    axs_bar.bar(bin_edges[:-1], heights, width=widths, edgecolor='black', align='edge', label=f'q = {P:.2f}')
    
    bin_centers = bin_edges[:-1] + widths/2
    heights, bin_centers = map(lambda x: x[heights!=0.], (heights, bin_centers))
    axs_plot.plot(bin_centers, heights, label=f'q = {P:.2f}')

for axs in (axs_bar, axs_plot):
    axs.set_xlabel(r'$\beta_P/\beta_0$')
    axs.set_ylabel('Counts')
    axs.set_yscale('log')
    axs.grid()
    axs.legend()

axs_violin.violinplot(betas_store, showmeans=True)
axs_violin.set_xlabel('DropEdge Probability')
axs_violin.set_ylabel(r'$\beta_P/\beta_0$')
axs_violin.grid()

fig.tight_layout()
plt.show()