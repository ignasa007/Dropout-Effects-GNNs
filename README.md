This is the official code repository for the paper [**Effects of Random Edge-Dropping on Over-Squashing in Graph Neural Networks**](https://openreview.net/forum?id=ZZwP9zljas). If you use this work, kindly cite it as

```
@misc{
    singh2024effects,
    title={Effects of Random Edge-Dropping on Over-Squashing in Graph Neural Networks},
    author={Jasraj Singh, Keyue Jiang, Brooks Paige, Laura Toni},
    booktitle={Submitted to The Thirteenth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=ZZwP9zljas},
    note={under review}
}
```

## Directory Structure

- `assets` - plots and other images included in the manuscript
- `data` - root directory for saving raw (and transformed) datasets, eg. `./data/Planetoid/Cora/`
- `dataset` - Python classes to handle different datasets, and make them suitable for training, eg. Cora
- `manuscript` - helper methods for reporting the final results
- `metrics` - Python classes for storing and computing performance metrics, eg. classification
- `model` - Python classes to initialize the model architectures
    - `activation` - activation functions, eg. ReLU
    - `dropout` - dropping methods, eg. DropEdge
    - `message_passing` - message passing layers, eg. GCN
    - `readout` - readout layer; task dependent, eg. node level or graph level task
- `plots` - plotting experimental results
    - `beta` - inspecting the scaling factor for commute times under a DropEdge random walk
    - `linear_gcn` - empirical results accompanying the theoretical ones
    - `metrics` - plots of the performance metrics
- `results` - results of the different runs
    - directory structure is not fixed, and can be set in `./utils/logger.py`
- `sensitvity` - studying the raw sensitivity between nodes at different distances
    - `log` - log the sensitivity measures (takes a while to run)
    - `plot` - plot the sensitivity between nodes against eg. shortest distance
    - `utils` - utility functions for these experiments, eg. `compute_shortest_distances`, `bin_jac_norms`
- `utils` - utility files for model training and logging

## Setup

```bash
conda create --name ${env_name} python=3.9
conda activate ${env_name}
pip install -r requirements_${os}.txt
```

## Execution

To train a model, execute
```bash
python -B main.py \
    --dataset ${dataset} \
    --gnn ${gnn} \
    --gnn_layer_sizes ${width}*${depth}
    --ffn_layer_sizes \
    --dropout ${dropout} \
    --drop_p ${drop_p} \
    --n_epochs ${n_epochs} \
    --learning_rate ${lr} \
    --weight_decay ${weight_decay} \
    --test_every 1 \
    --save_every -1
```

See `config.py` for the full list of command line arguments.
- `${dataset}` can be one of Cora, CiteSeer, PubMed, Proteins, MUTAG and PTC.
- `${gnn}` can be one of GCN, ResGCN, GAT and APPNP
    - is using GAT, pass the number of attention heads, eg. `--attention_heads 2`
    - if using APPNP, pass the number of power iteration steps and the teleport probability, eg. `--power_iter 10 --teleport_p 0.1`
- the hidden layer sizes can be passed via `--gnn_layer_sizes`, eg. `64 32 16` or even `64*3 32*2 16*1`
- if the task is at the graph level, `--pooler` argument needs to be passed
    - options are `mean`, `add` and `max`
- the readout module is an MLP with hidden layer sizes passed via `--ffn_layer_sizes`
    - empty argument defaults the readout to be a linear layer
- if using a GPU, pass the device index, eg. `--device_index 0`
- for the `--test_every` or `--save_every` arguments
    - passing `-1` instructs to test/save only in the last epoch
    - passing nothing instructs to not save/test in any epoch

## Reproducing the Results

**Figure 1**

```bash
python -m plots.linear_gcn.asymmetric --dataset ${dataset}   # left and middle plots
python -m plots.linear_gcn.compare_drop --dataset ${dataset} # right plot
```

- `${dataset}` can be one of Proteins and MUTAG. Technically, PTC can work as well, but it only has 188 graphs, and we report results for 200 graphs in the manuscript. The implementations can only use these datasets right now because PyG has a common API for them, but one can edit the line `dataset = TUDataset(...)` to use any other suitable dataset. Importantly, the dataset must have small graphs for fast sensitivity computation.
- The image files are stored at `./assets/linear-gcn/asymmetric/${dataset}.png` and `./assets/linear-gcn/compare-drop/${dataset}.png`, respectively.

**Table 2**

First exectute the experiments as described in the [Execution](#execution) section. Then print the LaTex table (I am lazy like that :')).

```bash
python -m manuscript.results_table
```

**Figure 2**

```bash
python -m plots.metrics.philia --gnn ${gnn} --dropout ${dropout}
```

- In Figure 2, `gnn=GCN` and `dropout=DropEdge`.
- The image files are stored at `./assets/${dropout}/${philia}.png`, where `${philia}` is one of `homophilic` and `heterophilic`.

**Figure 3**

```bash
python -m plots.metrics.ablation --gnn ${gnn} --dropout ${dropout}
```

The image files are stored at `./assets/${dropout}/ablation.png`.

**Figure 4**

This experiment uses Jacobian sampling with Cora dataset.

```bash
python -m sensitivity.log.single_large \
    --dataset ${dataset} \
    --gnn ${gnn} \
    --gnn_layer_sizes ${width}*${depth}
    --ffn_layer_sizes \
    --dropout ${dropout} \
    --drop_p ${drop_p}
```

- 20 target nodes are sampled from `dataset=Cora`.
- For each target node, its 6-hop neighborhood and the corresponding subgraph is computed.
- The Jacobian of the target node's final representation is computed with respect to the source node's features.
- 25 model samples are computed for each target node. In case of dropping methods, the model initialization and random masks are jointly sampled.
- The Jacobian norms are stored at `./jac-norms/i=${i}/${dropout}-${gnn}/sample-${sample}.pkl`, where `${i}` is the index of the node in the original Cora network and `${sample}` is from 1 to 25.
- The shortest distances from source nodes are stored at `./jac-norms/i=${i}/shortest_distances.pkl`.

```bash
python -m sensitivity.plot.jac_norm_vs_sd
```

- Compute the total number of node pairs at each of the distances 0 to 6 (line 42), as well as the sum of the sensitivity between nodes at these distances (line 48).
- Compute the mean sensitivity between nodes at different distances (line 51).
- Compute the influence distribution (line 53).
- Compute the mean and standard deviation &ndash; over the 25 samples &ndash; of the influence at different distances (line 55).

**Figure 5**

```bash
python -m plots.linear_gcn.symmetric --dataset ${dataset}
```

The image files are stored at `./assets/linear-gcn/symmetric/${dataset}.png`.

**Figure 6**

```bash
python -m plots.linear_gcn.black_extension --dataset ${dataset}
```

The image files are stored at `./assets/linear-gcn/black-extension/${dataset}.png`.

**Figure 7**

```bash
python -m plots.metrics.philia --gnn ${gnn} --dropout ${dropout}
```

In Figure 7, `gnn=GCN` and `dropout=DropNode`.
