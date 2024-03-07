#!/bin/bash

mutation_frequencies=("0.01" "0.05" "0.10" "0.20" "0.30")
seeds=("0" "1" "2")

current_folder="/home/gathenes/gitlab/HILARy/data_with_scripts/benchmark_on_partis_dataset/"

# Loop through the array
for mut in "${mutation_frequencies[@]}"; do
    for seed in "${seeds[@]}"; do
        mut_seed_folder=${current_folder}seed-${seed}/scratch-mute-freq-${mut}/hilary-full-1-2-0/both_chains/
        echo $mut_seed_folder
        mkdir -p $mut_seed_folder
        infer-lineages full-method ${current_folder}seed-$seed/scratch-mute-freq-$mut/simulations/both_chains/igh.tsv --kappa-file ${current_folder}seed-$seed/scratch-mute-freq-$mut/simulations/both_chains/igk.tsv -vv --config ${current_folder}simu_to_hilary.json --result-folder $mut_seed_folder --json -t -1 -s 1 > ${mut_seed_folder}logs.json
    done
done
