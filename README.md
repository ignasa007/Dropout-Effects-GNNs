## Directory Structure

- `assets` - plots and other images included in the thesis
- `data` - root directory for saving raw (and transformed) datasets, eg. `./data/Planetoid/Cora/`
- `dataset` - Python classes to handle different datasets, and make them suitable for training, eg. Cora
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
conda create --name ${env_name} python=3.8
conda activate ${env_name}
pip install -r requirements.txt
```

## Execution

To train a model, execute
```bash
python -B main.py \
    --dataset $dataset \
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
- `${gnn}` can be one of GCN, GAT and APPNP
    - is using GAT, pass the number of attention heads, eg. `--attention_heads 2`
    - if using APPNP, pass the number of power iteration steps and the teleport probability, eg. `--power_iter 10 --teleport_p 0.1`
- the hidden layer sizes can be passed via `--gnn_layer_sizes`, eg. `64 32 16` or even `64*3 32*2 16*1`
- if the task is at the graph level, `--pooler` argument needs to be passed
    - options are `mean`, `add` and `max`
- the readout layer is an MLP with hidden layer sizes passed via `--ffn_layer_sizes`
    - empty argument defaults the readout to be a linear layer
- if using a GPU, pass the device index, eg. `--device_index 0`
- for the `--test_every` or `--save_every` arguments
    - passing `-1` instructs to test/save only in the last epoch
    - passing nothing instructs to not save/test in any epoch
