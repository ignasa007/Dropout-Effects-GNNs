#!/bin/bash

source experiments/parse_args.sh
parse_args "$@"

if [ ! -v datasets ] || [ ${#datasets[@]} -eq 0 ]; then
    datasets=("Cora" "CiteSeer" "PubMed" "Chameleon" "Squirrel" "TwitchDE" "Actor")
fi

if [ ! -v gnns ] || [ ${#gnns[@]} -eq 0 ]; then
    gnns=("GCN" "GIN" "GAT")
fi
bias="${bias:-true}"
hidden_size="${hidden_size:-64}"
depth="${depth:-4}"
attention_heads="${attention_heads:-2}"
pooler="${pooler:-mean}"

if [ ! -v drop_ps ] || [ ${#drop_ps[@]} -eq 0 ]; then
    drop_ps=(0.2 0.3 0.5 0.8)
fi
if [ ! -v info_loss_ratios ] || [ ${#info_loss_ratios[@]} -eq 0 ]; then
    info_loss_ratios=(0.5 0.8 0.9 0.95)
fi

learning_rate="${learning_rate:-0.001}"
weight_decay="${weight_decay:-0}"
n_epochs="${n_epochs:-300}"
device_index="${device_index:-}"
total_samples="${total_samples:-20}"

for dataset in "${datasets[@]}"; do
    for gnn in "${gnns[@]}"; do
        for drop_p in $( [[ "$dropout" == "NoDrop" ]] && echo "0.0" || echo "${drop_ps[@]}" ); do
            for info_loss_ratio in "${info_loss_ratio[@]}"; do
                config_dir="./results/${dataset}/${gnn}/L=${depth}/${dropout}/P=${drop_p}/C=${info_loss_ratio}"
                num_samples=$(find "${config_dir}" -mindepth 1 -type d 2>/dev/null | wc -l)
                while [ ${num_samples} -lt ${total_samples} ]; do
                    python -m main \
                        --dataset "${dataset}" \
                        --gnn "${gnn}" \
                        --gnn_layer_sizes "${hidden_size}*${depth}" \
                        $( [[ "${gnn}" == "GAT" ]] && echo --attention_heads ${attention_heads} ) \
                        --bias "${bias}" \
                        --pooler "${pooler}" \
                        --dropout "DropSens" \
                        --drop_p "${drop_p}" \
                        --info_loss_ratio ${info_loss_ratio} \
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