## Directory Structure

- `assets` - root directory for storing images (.png files)
- `data` - root directory for storing raw (and transformed) datasets, e.g. `./data/Planetoid/Cora/`
- `dataset` - classes for handling different datasets, making them suitable for training, e.g. Cora
- `experiments` - driver code (.sh files) for running the experiments
- `jac-norms` - root directory for storing Jacobian norms for empirical sensitivity analysis
- `metrics` - classes for computing performance metrics for regression and classification tasks
- `model` - classes defining model architectures
    - `activation` - activation functions, e.g. ReLU
    - `dropout` - dropping methods, e.g. DropEdge
    - `message_passing` - message passing layers, e.g. GCN
    - `readout` - task-dependent readout layer e.g. for node-level classification
- `over_squashing` - studying the raw sensitivity between nodes at different distances
    - `log` - log the sensitivity measures (takes a while to run)
    - `plot` - plot the sensitivity between nodes against e.g. shortest distance
    - `utils` - utility functions for these experiments, e.g. `compute_shortest_distances`, `aggregate`
- `plots` - methods for plotting experimental results
    - `drop_sens` - DropSens related plots, e.g. runtime plots, comparison with NoDrop and DropEdge
    - `linear_gcn` - empirical results accompanying the theoretical analysis
    - `metrics` - plots of the performance metrics
- `results` - root directory for storing results of training runs
    - directory structure is not fixed, and must be passed using the `exp_dir` command-line argument
    - see `experiments` for the directory structure in use
- `tables` - methods for reporting the final experimental results
- `utils` - utility methods for model training and logging

## Setup

```bash
conda create --name ${env_name} python=3.9.21
conda activate ${env_name}
pip install -r ./requirements.txt
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
    --gnn_layer_sizes ${gnn_width}*${gnn_depth}     # Hidden layer sizes
    --ffn_layer_sizes ${ffn_width}*${ffn_depth}     # Hidden layer sizes
    --dropout ${dropout}
    --drop_p ${drop_p}
    --learning_rate ${lr}
    --weight_decay ${weight_decay}
    --n_epochs ${n_epochs}
    --test_every ${test_every}
    --save_every ${save_every}
    --exp_dir ${exp_dir}
    --device_index ${device_index}
```

See `./utils/config.py` for the full list of command line arguments.
- `${dataset}` can be one of the classes in `./dataset/__init__.py`
- `${gnn}` can be one of the message passing classes in `./model/message_passing/__init__.py`
    - if using GAT, pass the number of attention heads, e.g. `--gat_attention_heads 2`
    - if using APPNP, pass the number of power iteration steps and the teleport probability, e.g. `--appnp_power_iter 10 --appnp_teleport_p 0.1`
- the hidden layer sizes can be passed via `--gnn_layer_sizes`, e.g. `64 32 16` or even `64*3 32*2 16*1`
- if the task is at the graph level, `--graph_pooler` argument needs to be passed
    - options are `mean`, `add` and `max`
- the readout module is an MLP with hidden layer sizes passed via `--ffn_layer_sizes`
    - empty argument defaults the readout to be a linear layer
- `${dropout}` can be one of the dropout classes in `./model/dropout/__init__.py`
- if using a GPU, pass its index, e.g. `--device_index 0`, else CPU will be used
- for the `--test_every` or `--save_every` arguments
    - passing `n` instructs to test/save every `n` epochs
    - passing `-1` instructs to test/save only in the last epoch
    - passing nothing instructs to not save/test in any epoch

## Citation

```
@inproceedings{
    singh2025effects,
    title={Effects of Dropout on Performance in Long-range Graph Learning Tasks},
    author={Jasraj Singh and Keyue Jiang and Brooks Paige and Laura Toni},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=4449zuaaeL}
}
```