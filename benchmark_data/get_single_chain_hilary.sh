#!/bin/bash

mutation_frequencies=("0.01" "0.05" "0.10" "0.20" "0.30")
seeds=("0" "1" "2")


# Loop through the array
for mut in "${mutation_frequencies[@]}"; do
    for seed in "${seeds[@]}"; do
        mkdir -p /home/gabrielathenes/Documents/study/HILARy/benchmark_data/seed-${seed}/scratch-mute-freq-${mut}/hilary/single_chain/
        infer /home/gabrielathenes/Documents/study/benchmark_data/seed-$seed/scratch-mute-freq-$mut/simulations/single_chain/igh.tsv -vv --config /home/gabrielathenes/Documents/study/HILARy/benchmark_data/simu_to_hilary.json --result-folder /home/gabrielathenes/Documents/study/HILARy/benchmark_data/seed-$seed/scratch-mute-freq-$mut/hilary/single_chain/ --nmin 2000
    done
done
