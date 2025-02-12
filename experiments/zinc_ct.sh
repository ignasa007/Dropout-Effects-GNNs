#!/bin/bash

total_samples=20
dataset=SyntheticZINC_CT
gnn=${1}
device_index=${2}

dropouts=("NoDrop" "DropEdge")  # "Dropout" "DropMessage" "DropNode" "DropAgg" "DropGNN"
drop_ps=(0.2 0.5)
alphas=$(seq 0.0 0.1 1.0)

hidden_size=16
depth=11
bias=true
pooler=max 
attention_heads=2
learning_rate=2e-3
weight_decay=1e-4
n_epochs=500

for dropout in "${dropouts[@]}"; do
    for drop_p in $( [[ "$dropout" == "NoDrop" ]] && echo "0.0" || echo "${drop_ps[@]}" ); do
        for alpha in ${alphas}; do
            for sample in $(seq 1 1 ${total_samples}); do
                exp_dir="./results/${dropout}/${dataset}/${gnn}/P=${drop_p}/alpha=${alpha}/sample-${sample}"
                if [ -d "${exp_dir}" ]; then
                    continue
                fi
                python -B main.py \
                    --dataset ${dataset} \
                    --distance ${alpha} \
                    --gnn ${gnn} \
                    --gnn_layer_sizes ${hidden_size}*${depth} \
                    $( [[ "${gnn}" == "GAT" ]] && echo --attention_heads ${attention_heads} ) \
                    --bias ${bias} \
                    --pooler ${pooler} \
                    --dropout ${dropout} \
                    --drop_p ${drop_p} \
                    --model_sample ${sample} \
                    --learning_rate ${learning_rate} \
                    --weight_decay ${weight_decay} \
                    --n_epochs ${n_epochs} \
                    --device_index ${device_index} \
                    --exp_dir ${exp_dir};
            done
        done
    done
done