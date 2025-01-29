#!/bin/bash

for sample in `seq 1 1 10`; do
    for distance in `seq 4 1 8`; do
        for gnn in {'GCN','GAT'}; do
            python -B main.py \
                --dataset SyntheticZINC_SD \
                --distance ${distance} \
                --gnn ${gnn} \
                --gnn_layer_sizes 32*${distance} \
                --attention_heads 2 \
                --pooler 'max' \
                --dropout 'NoDrop' \
                --device_index 0;
            for dropout in {'Dropout','DropMessage','DropEdge','DropNode','DropAgg','DropGNN'}; do
                for drop_p in {0.1,0.2,0.5}; do
                    python -B main.py \
                        --dataset SyntheticZINC_SD \
                        --distance ${distance} \
                        --gnn ${gnn} \
                        --gnn_layer_sizes 32*${distance} \
                        --attention_heads 2 \
                        --pooler 'max' \
                        --dropout ${dropout} \
                        --drop_p ${drop_p} \
                        --device_index 0;
                done
            done
        done
    done
done