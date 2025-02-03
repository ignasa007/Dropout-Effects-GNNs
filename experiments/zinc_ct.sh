#!/bin/bash

total_samples=5
dataset=SyntheticZINC_CT
gnn=${1}
device_index=${2}

dropouts=("NoDrop" "DropEdge")  # "Dropout" "DropMessage" "DropNode" "DropAgg" "DropGNN")
drop_ps=(0.1 0.2 0.5)
alphas=$(seq 0.0 0.1 1.0)

hidden_size=64      # 96
depth=11
bias=true           # false
pooler=max
attention_heads=2   # 4
learning_rate=2e-3
weight_decay=5e-4   # 0
n_epochs=250        # 500

for dropout in "${dropouts[@]}"; do
    for drop_p in $( [[ "$dropout" == "NoDrop" ]] && echo "0.0" || echo "${drop_ps[@]}" ); do
        for alpha in ${alphas}; do
            num_samples=$(find results/${dropout}/${dataset}/${gnn}/L=${alpha}/P=${drop_p} -mindepth 1 -type d 2>/dev/null | wc -l)
            while [ ${num_samples} -lt ${total_samples} ]; do
                python -B main.py \
                    --dataset ${dataset} \
                    --distance ${alpha} \
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
done