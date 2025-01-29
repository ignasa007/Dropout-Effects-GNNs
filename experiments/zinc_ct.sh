#!/bin/bash

gnn=${1}
device_index=${2}

dropouts=("NoDrop" "Dropout" "DropMessage" "DropEdge")  # "DropNode" "DropAgg" "DropGNN")
drop_ps=(0.1 0.2 0.5)
alphas=$(seq 0.0 0.05 1.0)
samples=$(seq 1 1 10)

hidden_size=64      # 96
depth=11
bias=true           # false
pooler=max
attention_heads=2   # 4
learning_rate=2e-3
weight_decay=5e-4   # 0
n_epochs=250        # 500

for sample in ${samples}; do
    for alpha in ${alphas}; do
        for dropout in "${dropouts[@]}"; do
            for drop_p in $( [[ "$dropout" == "NoDrop" ]] && echo "0.0" || echo "${drop_ps[@]}" ); do
                python -B main.py \
                    --dataset SyntheticZINC_CT \
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
                    --device_index ${device_index};
            done
        done
    done
done