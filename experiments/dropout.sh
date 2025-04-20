#!/bin/bash

source experiments/parse_args.sh
parse_args "$@"

if [ ! -v datasets ] || [ ${#datasets[@]} -eq 0 ]; then
    echo "Error: --datasets cannot be empty."
    exit 1
fi
if [ ! -v gnns ] || [ ${#gnns[@]} -eq 0 ]; then
    echo "Error: --gnns cannot be empty."
    exit 1
fi

bias="${bias:-true}"
hidden_size="${hidden_size:-64}"
depth="${depth:-4}"
attention_heads="${attention_heads:-2}"
pooler="${pooler:-mean}"

if [ ! -v dropouts ] || [ ${#dropouts[@]} -eq 0 ]; then
    echo "Error: --dropouts cannot be empty."
    exit 1
fi
if [ ! -v drop_ps ] || [ ${#drop_ps[@]} -eq 0 ]; then
    drop_ps=$(seq 0.1 0.1 0.9)
fi

learning_rate="${learning_rate:-0.001}"
weight_decay="${weight_decay:-0}"
n_epochs="${n_epochs:-300}"
device_index="${device_index:-}"
total_samples="${total_samples:-20}"

for dataset in "${datasets[@]}"; do
    for gnn in "${gnns[@]}"; do
        for dropout in "${dropouts[@]}"; do
            for drop_p in $( [[ "$dropout" == "NoDrop" ]] && echo "0.0" || echo "${drop_ps[@]}" ); do
                echo "Running: bash experiments/dropout.sh --datasets ${dataset} --gnns ${gnn} --bias ${bias} --hidden_size ${hidden_size} --depth ${depth} --attention_heads ${attention_heads} --pooler ${pooler} --dropouts ${dropout} --drop_ps ${drop_p} --learning_rate ${learning_rate} --weight_decay ${weight_decay} --n_epochs ${n_epochs} --device_index ${device_index} --total_samples ${total_samples}"
                config_dir="./results/${dataset}/${gnn}/L=${depth}/${dropout}/P=${drop_p}"
                num_samples=$(find "${config_dir}" -mindepth 1 -type d 2>/dev/null | wc -l)
                while [ ${num_samples} -lt ${total_samples} ]; do
                    python -m main \
                        --dataset "${dataset}" \
                        --gnn "${gnn}" \
                        --gnn_layer_sizes "${hidden_size}*${depth}" \
                        $( [[ "${gnn}" == "GAT" ]] && echo --attention_heads ${attention_heads} ) \
                        --bias "${bias}" \
                        --pooler "${pooler}" \
                        --dropout "${dropout}" \
                        --drop_p "${drop_p}" \
                        --learning_rate "${learning_rate}" \
                        --weight_decay "${weight_decay}" \
                        --n_epochs "${n_epochs}" \
                        $( [[ -n "${device_index}" ]] && echo --device_index ${device_index} ) \
                        --exp_dir "${config_dir}/$(date "+%Y-%m-%d-%H-%M-%S")" \
                    && num_samples=$((${num_samples}+1));
                done
            done
        done
    done
done