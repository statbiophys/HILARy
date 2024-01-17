#!/bin/bash

mutation_frequencies=("0.01" "0.05" "0.10" "0.20" "0.30")
seeds=("0" "1" "2")

# Loop through the array
for mut in "${mutation_frequencies[@]}"; do
    for seed in "${seeds[@]}"; do
        mkdir -p ./seed-${seed}/scratch-mute-freq-${mut}/scoper/both_chains/
        cp ~/Downloads/for-zenodo-vs-shm-v2/vs-shm/v2/seed-$seed/scratch-mute-freq-$mut/scoper/work/joint/partition.tsv ./seed-${seed}/scratch-mute-freq-${mut}/scoper/both_chains/partition.tsv
    done
done
