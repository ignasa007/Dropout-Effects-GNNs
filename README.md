This is the official code repository for the paper [**Effects of Dropout on Performance in Long-range Graph Learning Tasks**](https://arxiv.org/abs/2502.07364). If you use this work, kindly cite it as

```
@misc{
    singh2025effects,
    title={Effects of Dropout on Performance in Long-range Graph Learning Tasks},
    author={Jasraj Singh, Keyue Jiang, Brooks Paige, Laura Toni},
    year={2025},
    url={https://arxiv.org/abs/2502.07364}
}
```

## Directory Structure

- `assets` - root directory for storing plots and other images (.png files)
- `data` - root directory for storing raw (and transformed) datasets, eg. `./data/Planetoid/Cora/`
- `dataset` - classes for handling different datasets, making them suitable for training, eg. Cora
- `experiments` - driver code (.sh files) for running the experiments
- `jac-norms` - root directory for storing Jacobian norms for empirical sensitivity analysis
- `metrics` - classes for computing performance metrics for regression and classification tasks
- `model` - classes defining model architectures
    - `activation` - activation functions, eg. ReLU
    - `dropout` - dropping methods, eg. DropEdge
    - `message_passing` - message passing layers, eg. GCN
    - `readout` - task-dependent readout layer eg. for node-level classification
- `plots` - methods for plotting experimental results
    - `linear_gcn` - empirical results accompanying the theoretical analysis
    - `metrics` - plots of the performance metrics
- `results` - root directory for storing results of training runs
    - directory structure is not fixed, and must be passed using the `exp_dir` command-line argument
    - see `experiments` for the directory structure in use
- `sensitvity` - studying the raw sensitivity between nodes at different distances
    - `log` - log the sensitivity measures (takes a while to run)
    - `plot` - plot the sensitivity between nodes against eg. shortest distance
    - `utils` - utility functions for these experiments, eg. `compute_shortest_distances`, `aggregate`
- `tables` - methods for reporting the final experimental results
- `utils` - utility methods for model training and logging

## Setup

```bash
conda create --name ${env_name} python=3.9
conda activate ${env_name}
pip install -r requirements_${os}.txt
```

`PyG>=2.5.0` has an error in the file `torch_geometric.io.fs` at line 193 (see [issue](https://github.com/pyg-team/pytorch_geometric/issues/9330)). Change it to
```python
fs1.mv(path1, path2, recursive=recursive)
```

## Execution

To train a model, execute
```bash
python -m main
    --dataset ${dataset}
    --gnn ${gnn}
    --gnn_layer_sizes ${width}*${depth}
    --ffn_layer_sizes
    --dropout ${dropout}
    --drop_p ${drop_p}
    --n_epochs ${n_epochs}
    --learning_rate ${lr}
    --weight_decay ${weight_decay}
    --test_every 1
    --save_every -1
    --exp_dir './results/${dataset}/${gnn}/L=${depth}/${dropout}/P=${drop_p}'
```

See `config.py` for the full list of command line arguments.
- `${dataset}` can be one of the classes in `dataset.__init__.py`
- `${gnn}` can be one of the message passing classes in `message_passing.__init__.py`
    - if using GAT, pass the number of attention heads, eg. `--attention_heads 2`
    - if using APPNP, pass the number of power iteration steps and the teleport probability, eg. `--power_iter 10 --teleport_p 0.1`
- the hidden layer sizes can be passed via `--gnn_layer_sizes`, eg. `64 32 16` or even `64*3 32*2 16*1`
- if the task is at the graph level, `--pooler` argument needs to be passed
    - options are `mean`, `add` and `max`
- the readout module is an MLP with hidden layer sizes passed via `--ffn_layer_sizes`
    - empty argument defaults the readout to be a linear layer
- if using a GPU, pass its index, eg. `--device_index 0`
- for the `--test_every` or `--save_every` arguments
    - passing `-1` instructs to test/save only in the last epoch
    - passing nothing instructs to not save/test in any epoch

## Reproducing the Results

**Figure 1**

```bash
# (a)
python -m plots.linear_gcn.asymmetric 
    --dataset ${dataset}
    --device_index ${device_index}
```

- `${dataset}` can be any of the node classification datasets. The implementation can only use these datasets right now because handling (multiple) small networks, eg. in graph-level tasks, will have a different logic compared to handling a (single) large networks. We choose to focus on (single) large networks for our experiments because their larger diameter allows us to more precisely capture the over-squashing effects between nodes at large distances.
- The image files are stored at `./assets/linear-gcn/asymmetric/${dataset}.png`.

```bash
# (b) Logging Jacobian norms
python -m over_squashing.log.single_large 
    --dataset ${dataset}
    --gnn ${gnn}
    --gnn_layer_sizes ${width}*${depth}
    --dropout ${dropout}
    --drop_p ${drop}
    --exp_dir null
    --device_index ${device_index}
```

- 25 target nodes are sampled from `dataset=Cora`.
- For each target node, its 6-hop neighborhood and the corresponding subgraph is computed.
- The Jacobian of the target node's final representation is computed with respect to the source node's features.
- 25 model samples are computed for each target node. In case of dropping methods, the model initialization and random masks are jointly sampled.
- The Jacobian norms are stored at `./jac-norms/${dataset}/i=${i}/L=${depth}/${gnn}/${dropout}/sample=${sample}.pkl`, where `${i}` is the index of the node in the original Cora network and `${sample}` is from 1 to 25.
- The shortest distances from source nodes are stored at `./jac-norms/${dataset}/i=${i}/L=${depth}/shortest_distances.pkl`.

```bash
# Plotting averaged Jacobian norms
python -m over_squashing.plot.sensitvity
    --dataset ${dataset}
    --L ${depth}
    --drop_p ${drop_p}
```

<!-- - Compute the total number of node pairs at each of the distances 0 to 6, as well as the sum of the sensitivity between nodes at these distances (line 48).
- Compute the mean sensitivity between nodes at different distances (line 51).
- Compute the influence distribution (line 53).
- Compute the mean and standard deviation &ndash; over the 25 samples &ndash; of the influence at different distances (line 55). -->
- Compute the sensitivity of each target node's representations to corresponding source nodes' features, and average over the shortest distances.
- Compute the mean and standard deviation &ndash; over the 25 model samples &ndash; of the sensitivity at different distances.

The image file is saved at `./assets/sensitivity/${dataset}.png`.

**Figure 2**

```bash
# Run the experiment
bash experiments/zinc_ct.sh ${gnn} ${device_index}
# Plot the results
python -m plots.metrics.zinc
    --gnn ${gnn}
    --metric ${metric}
    --which ${which}
```

- In Figure 2, `gnn=GCN`, `metric=mae` and `which=best`.
- The image file is stored at `./assets/SyntheticZINC/${gnn}/${which}-${metric}.png`.

**Table 1**

```bash
# Find the best dropping probabilities over 20 independent runs
bash experiments/node.sh ${dataset} ${gnn} ${device_index}
python -m tables.main
    --node
    --best_prob
# Perform another 30 runs with the best performing dropping probability
total_samples=50
bash experiments/node.sh ${dataset} ${gnn} ${device_index}
    ${dropout} ${best_drop_p} ${total_samples}
# Report the p-values of the t-tests comparing performance against NoDrop
python -m tables.main
    --node
    --significance
```

**Table 2**

Replace `node` with `graph` in the above sequence of commands.

**Figure 3**

```bash
bash experiments/drop_sens.sh ${dataset} ${gnn} ${device_index}
python -m plots.metrics.drop_sens
```

The image file is stored at `./assets/DropSens/errors-diff.png`.

**Figure 4**

```bash
python -m plots.linear_gcn.symmetric 
    --dataset ${dataset}
    --device_index ${device_index}
```

The image file is stored at `./assets/linear-gcn/symmetric/${dataset}.png`.

**Figure 5**

```bash
python -m plots.linear_gcn.black_extension 
    --dataset ${dataset}
    --device_index ${device_index}
```

The image files are stored at `./assets/linear-gcn/black-extension/${dataset}.png`.

**Figures 6 and 7**

```bash
python -m plots.metrics.philia
    --gnn ${gnn} 
    --dropout ${dropout}
```

- In Figure 6, `gnn=GCN` and `dropout=DropEdge`.
- In Figure 7, `gnn=GCN` and `dropout=DropNode`.
- The image files are stored at `./assets/${dropout}/${philia}.png`, where `${philia}` is one of `homophilic` and `heterophilic`.

**Figure 8**

```bash
python -m plots.metrics.ablation --gnn ${gnn} --dropout ${dropout}
```

The image files are stored at `./assets/${dropout}/ablation.png`.