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
- `over_squashing` - studying the raw sensitivity between nodes at different distances
    - `log` - log the sensitivity measures (takes a while to run)
    - `plot` - plot the sensitivity between nodes against eg. shortest distance
    - `utils` - utility functions for these experiments, eg. `compute_shortest_distances`, `aggregate`
- `plots` - methods for plotting experimental results
    - `linear_gcn` - empirical results accompanying the theoretical analysis
    - `metrics` - plots of the performance metrics
- `results` - root directory for storing results of training runs
    - directory structure is not fixed, and must be passed using the `exp_dir` command-line argument
    - see `experiments` for the directory structure in use
- `tables` - methods for reporting the final experimental results
- `utils` - utility methods for model training and logging

## Setup

```bash
conda create --name ${env_name} python=3.9
conda activate ${env_name}
pip install -r requirements.txt
```

`PyG 2.5.3` has an error in the file `torch_geometric.io.fs` at line 193 (see [issue](https://github.com/pyg-team/pytorch_geometric/issues/9330)). Change it to
```python
def mv(path1: str, path2: str) -> None:
    fs1 = get_fs(path1)
    fs2 = get_fs(path2)
    assert fs1.protocol == fs2.protocol
    fs1.mv(path1, path2)
```
as was fixed in `PyG 2.6.0` (see [pull request](https://github.com/pyg-team/pytorch_geometric/pull/9436)).

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
    --test_every ${test_every}
    --save_every ${save_every}
    --exp_dir ${exp_dir}
    --device_index ${device_index}
```

See `utils/config.py` for the full list of command line arguments.
- `${dataset}` can be one of the classes in `dataset/__init__.py`
- `${gnn}` can be one of the message passing classes in `model/message_passing/__init__.py`
    - if using GAT, pass the number of attention heads, eg. `--attention_heads 2`
    - if using APPNP, pass the number of power iteration steps and the teleport probability, eg. `--power_iter 10 --teleport_p 0.1`
- the hidden layer sizes can be passed via `--gnn_layer_sizes`, eg. `64 32 16` or even `64*3 32*2 16*1`
- if the task is at the graph level, `--pooler` argument needs to be passed
    - options are `mean`, `add` and `max`
- the readout module is an MLP with hidden layer sizes passed via `--ffn_layer_sizes`
    - empty argument defaults the readout to be a linear layer
- `${dropout}` can be one of the dropout classes in `model/dropout/__init__.py`
- if using a GPU, pass its index, eg. `--device_index 0`, else CPU will be used
- for the `--test_every` or `--save_every` arguments
    - passing `n` instructs to test/save every `n` epochs
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
    --device_index ${device_index}
```

- 25 target nodes are sampled from `dataset=Cora`.
- For each target node, its `depth=6`-hop neighborhood and the corresponding subgraph is extracted.
- The Jacobian of the target node's final representation is computed with respect to the source node's features.
- 25 model samples are computed for each target node. In case of dropping methods, the model initialization and random masks are jointly sampled.
- The Jacobian norms are stored at `./jac-norms/${dataset}/i=${i}/L=${depth}/${gnn}/${dropout}/sample=${sample}.pkl`, where `${i}` is the index of the node in the original Cora network and `${sample}` is from 1 to 25.
- The shortest distances from source nodes are stored at `./jac-norms/${dataset}/i=${i}/L=${depth}/shortest_distances.pkl`.

```bash
# Plotting averaged influence distribution
python -m over_squashing.plot.influence
    --dataset ${dataset}
    --L ${depth}
    --drop_p ${drop_p}
```

- Compute the sensitivity of each target node's representations to corresponding source nodes' features, and average over the shortest distances.
- Compute the influence distribution.
- Compute the mean and standard deviation &ndash; over the 25 model samples &ndash; of influence at different distances.

The image file is saved at `./assets/influence/${dataset}.png`.

**Figure 2**

```bash
# Run the experiment
bash experiments/zinc_ct.sh 
    --gnns ${gnn}
    --dropouts ${dropouts}
    --device_index ${device_index}
# Plot the results
python -m plots.metrics.zinc
    --gnn ${gnn}
    --metric ${metric}
    --which ${which}
```

- In Figure 2, `gnn=GCN`, `metric=mae` and `which=best`.
- The image file is stored at `./assets/SyntheticZINC/${gnn}/${which}-${metric}.png`.

**Table 1**

Find the best dropping probabilities over 20 independent runs:
```bash
bash experiments/dropout.sh 
    --datasets ${datasets}
    --gnns ${gnns}
    --dropouts ${dropouts}
    --device_index ${device_index}
python -m tables.main
    --best_prob
```

Perform another 30 runs with the best performing dropping probability:
```bash
total_samples=50
bash experiments/dropout.sh 
    --datasets ${dataset}
    --gnns ${gnn}
    --dropouts ${dropout} 
    --drop_ps ${best_drop_p}
    --total_samples ${total_samples}
    --device_index ${device_index}
```

Report the p-values of the t-tests comparing performance against NoDrop:
```bash
python -m tables.main
    --p_value
    --node
```

**Table 2**

Execute the first two sets of commands as above, and pass the `--graph` flag in the final command:
```bash
python -m tables.main
    --p_value
    --graph
```

**Table 3** and **Table 4**

Find the best dropping configuration over 20 independent runs:
```bash
bash experiments/drop_sens.sh 
    --datasets ${datasets}
    --gnns ${gnns}
    --device_index ${device_index}
python -m tables.main
    --best_prob
```

Perform another 30 runs with the best performing dropping configuration:
```bash
total_samples=50
bash experiments/dropout.sh 
    --datasets ${dataset}
    --gnns ${gnn} 
    --drop_ps ${best_drop_p}
    --info_save_ratios ${best_info_save_ratio} 
    --total_samples ${total_samples}
    --device_index ${device_index}
```

**Table 5**

Computes edge homophily measure [1], and a new homophily measure proposed in [2].
```bash
python -m utils.homphily
    --dataset ${dataset}
```
`${dataset}` can be one of the implemented node-classification dataset classes. If you want to compute the homophily measures for a new dataset, define it accordingly in `utils/homphily.py`.

[1] Jiong Zhu, Yujun Yan, Lingxiao Zhao, Mark Heimann, Leman Akoglu, and Danai Koutra. Beyond homophily in graph neural networks: Current limitations and effective designs. In Advances in Neural Information Processing Systems, volume 33, pp. 7793–7804.

[2] Derek Lim, Xiuyu Li, Felix Hohne, and Ser-Nam Lim. New benchmarks for learning on non-homophilous graphs. arXiv preprint arXiv:2104.01404, 2021.

**Figure 3**

```bash
python -m plots.linear_gcn.symmetric 
    --dataset ${dataset}
    --device_index ${device_index}
```

The image file is stored at `./assets/linear-gcn/symmetric/${dataset}.png`.

**Figure 4**

```bash
python -m plots.linear_gcn.black_extension 
    --dataset ${dataset}
    --device_index ${device_index}
```

The image file is stored at `./assets/linear-gcn/black-extension/${dataset}.png`.

**Figure 5**

```bash
python -m plots.drop_sense_approx
```

The image file is stored at `./assets/DropSens/approximation.png`.

**Figures 6 and 7**

```bash
python -m plots.metrics.philia
    --gnn ${gnn} 
    --dropout ${dropout}
```

- In Figure 6, `gnn=GCN` and `dropout=DropEdge`.
- In Figure 7, `gnn=GCN` and `dropout=DropNode`.
- The image files are stored at `./assets/philia/${gnn}/${dropout}/Test/heterophilic.png`.

**Figure 8**

```bash
python -m plots.metrics.philia
    --gnn ${gnn} 
    --dropout ${dropout}
    --train
```

- In Figure 8, `dropout=DropEdge`.
- The image file is stored at `./assets/philia/${gnn}/${dropout}/Train/heterophilic.png`.

**Table 8**

```bash
python -m tables.main
    --best_prob
```

**Table 9**

```bash
python -m tables.main
    --effect_size
```