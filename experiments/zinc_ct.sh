#!/bin/bash

source experiments/parse_args.sh
parse_args "$@"

dataset=SyntheticZINC_CT
step=0.1

python -m dataset.synthetic_zinc \
    --step ${step};

if [ ! -v gnns ] || [ ! ${#gnns[@]} -eq 1 ]; then
    echo "Error: exactly one argument needs to passed with --gnns."
    exit 1
fi

gnn=${gnns[0]}
hidden_size="${hidden_size:-16}"
depth="${depth:-11}"
bias="${bias:-true}"
attention_heads="${attention_heads:-2}"
pooler="${pooler:-max}"

python -m utils.gen_model_samples \
    --dataset ${dataset} \
    --gnn ${gnn} \
    --gnn_layer_sizes ${hidden_size}*${depth} \
    --bias ${bias} \
    --pooler ${pooler} \
    --num_samples ${total_samples};

if [ ! -v dropouts ] || [ ${#dropouts[@]} -eq 0 ]; then
    echo "Error: --dropouts cannot be empty."
    exit 1
fi
if [ ! -v drop_ps ] || [ ${#drop_ps[@]} -eq 0 ]; then
    drop_ps=(0.2 0.5)
fi

learning_rate="${learning_rate:-0.002}"
weight_decay="${weight_decay:-0.0001}"
n_epochs="${n_epochs:-200}"
device_index="${device_index:-}"
total_samples="${total_samples:-10}"

dropouts=("NoDrop" "DropEdge" "Dropout" "DropMessage")
drop_ps=(0.2 0.5)
distances=$(seq 0.0 ${step} 1.0)
samples=$(seq 1 1 ${total_samples}) 

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