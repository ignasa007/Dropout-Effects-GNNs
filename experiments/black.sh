#!/bin/bash

total_samples=20
dataset=${1}
gnn=${2}
device_index=${3}

dropouts=("NoDrop" "DropEdge" "Dropout" "DropMessage")
drop_ps=$(seq 0.1 0.1 0.9)

hidden_size=64
depth=4
bias=true
pooler=mean
attention_heads=2
learning_rate=1e-3
weight_decay=0
n_epochs=300


for dropout in "${dropouts[@]}"; do
    for drop_p in $( [[ "$dropout" == "NoDrop" ]] && echo "0.0" || echo "${drop_ps[@]}" ); do
        num_samples=$(find results/${dropout}/${dataset}/${gnn}/L=${depth}/P=${drop_p} -mindepth 1 -type d 2>/dev/null | wc -l)
        while [ ${num_samples} -lt ${total_samples} ]; do
            python -B main.py \
                --dataset ${dataset} \
                --gnn ${gnn} \
                --gnn_layer_sizes ${hidden_size}*${depth} \
                $( [[ "${gnn}" == "GAT" ]] && echo "--attention_heads ${attention_heads}" ) \
                --bias ${bias} \
                --pooler ${pooler} \
                --dropout ${dropout} \
                --drop_p ${drop_p} \
                --learning_rate ${learning_rate} \
                --weight_decay ${weight_decay} \
                --n_epochs ${n_epochs} \
                --device_index ${device_index} \
            && num_samples=$((${num_samples}+1));
        done
    done
done