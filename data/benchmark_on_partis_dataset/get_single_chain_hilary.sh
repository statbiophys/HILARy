#!/bin/bash

mutation_frequencies=("0.01" "0.05" "0.10" "0.20" "0.30")
seeds=("0" "1" "2")

current_folder="/home/gathenes/gitlab/HILARy/data/benchmark_on_synthetic_dataset/subsampled_simulations/"

# Loop through the array
for mut in "${mutation_frequencies[@]}"; do
    for seed in "${seeds[@]}"; do
        mut_seed_folder=${current_folder}seed-${seed}/scratch-mute-freq-${mut}/hilary-1-1-7/single_chain/
        echo $mut_seed_folder
        mkdir -p $mut_seed_folder
        infer ${current_folder}seed-$seed/scratch-mute-freq-$mut/simulations/single_chain/igh.tsv -vv --config ${current_folder}simu_to_hilary.json --result-folder $mut_seed_folder --json -t -1 --override > ${mut_seed_folder}logs.json
    done
done
