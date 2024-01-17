#!/bin/bash

mutation_frequencies=("0.01" "0.05" "0.10" "0.20" "0.30")
seeds=("0" "1" "2")
chains=("h" "k")

# Loop through the array
for mut in "${mutation_frequencies[@]}"; do
    for seed in "${seeds[@]}"; do
        for chain in "${chains[@]}"; do
            mkdir -p ./seed-${seed}/scratch-mute-freq-${mut}/scoper/single_chain/
            cp ~/Downloads/for-zenodo-vs-shm-v2/vs-shm/v2/seed-$seed/scratch-mute-freq-$mut/scoper/work/ig$chain/partition.tsv ./seed-${seed}/scratch-mute-freq-${mut}/scoper/single_chain/partition-ig$chain.tsv
        done
    done
done
