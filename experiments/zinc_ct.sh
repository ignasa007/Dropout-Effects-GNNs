#!/bin/bash

total_samples=10
dataset=SyntheticZINC_CT
step=0.1
gnn=${1}
device_index=${2}

dropouts=("NoDrop" "DropEdge" "Dropout" "DropMessage")
drop_ps=(0.2 0.5)
distances=$(seq 0.0 ${step} 1.0)
samples=$(seq 1 1 ${total_samples}) 

hidden_size=16
depth=11
bias=true
pooler=max
attention_heads=2
learning_rate=2e-3
weight_decay=1e-4
schedule_lr=true
n_epochs=200

python -m dataset.synthetic_zinc \
    --step ${step};

python -m utils.gen_model_samples \
    --dataset ${dataset} \
    --gnn ${gnn} \
    --gnn_layer_sizes ${hidden_size}*${depth} \
    --pooler ${pooler} \
    --num_samples ${total_samples} \
    --exp_dir null;

for dropout in "${dropouts[@]}"; do
    for drop_p in $( [[ "$dropout" == "NoDrop" ]] && echo "0.0" || echo "${drop_ps[@]}" ); do
        for distance in ${distances}; do
            for sample in ${samples}; do
                exp_dir="./results/${dataset}/distance=${distance}/${gnn}/L=${depth}/sample=${sample}/${dropout}/P=${drop_p}/"
                if [ -f "${exp_dir}/logs" ]; then
                    continue
                fi
                python -B main.py \
                    --dataset ${dataset} \
                    --distance ${distance} \
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
                    --schedule_lr ${schedule_lr} \
                    --n_epochs ${n_epochs} \
                    --device_index ${device_index} \
                    --exp_dir ${exp_dir};
            done
        done
    done
done