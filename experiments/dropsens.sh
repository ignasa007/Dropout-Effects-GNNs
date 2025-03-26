#!/bin/bash

dataset=${1}
gnn=${2}
device_index=${3}
dropout="DropSens"
if [ ! -z "${4+x}" ]; then    # ${4} is set
    drop_ps=("${4}")
else
    drop_ps=(0.2 0.3 0.5 0.8)
fi
if [ ! -z "${5+x}" ]; then
    info_loss_ratios=("${5}")
else 
    info_loss_ratios=(0.5 0.8 0.9 0.95)
fi
total_samples="${6:-20}"

hidden_size=64
depth=4
bias=true
attention_heads=2
learning_rate=1e-3
weight_decay=0
n_epochs=300

for drop_p in "${drop_ps[@]}"; do
    for info_loss_ratio in "${info_loss_ratios[@]}"; do
        config_dir="./results/${dropout}/${dataset}/${gnn}/L=${depth}/P=${drop_p}/C=${info_loss_ratio}"
        num_samples=$(find ${config_dir} -mindepth 1 -type d 2>/dev/null | wc -l)
        while [ ${num_samples} -lt ${total_samples} ]; do
            python -B main.py \
                --dataset ${dataset} \
                --gnn ${gnn} \
                --gnn_layer_sizes ${hidden_size}*${depth} \
                $( [[ "${gnn}" == "GAT" ]] && echo --attention_heads ${attention_heads} ) \
                --bias ${bias} \
                --pooler "mean" \
                --dropout ${dropout} \
                --drop_p ${drop_p} \
                --info_loss_ratio ${info_loss_ratio} \
                --learning_rate ${learning_rate} \
                --weight_decay ${weight_decay} \
                --n_epochs ${n_epochs} \
                --device_index ${device_index} \
                --exp_dir ${config_dir}/$(date "+%Y-%m-%d-%H-%M-%S") \
            && num_samples=$((${num_samples}+1));
        done
    done
done