#!/bin/bash

mutation_frequencies=("0.01" "0.05" "0.10" "0.20" "0.30")
seeds=("0" "1" "2")
thresholds=("0.09" "0.12" "0.15" "0.18" "0.21" "0.24")

current_folder="/home/gathenes/gitlab/HILARy/data/benchmark_on_partis_dataset/"

# Loop through the array
for mut in "${mutation_frequencies[@]}"; do
    for seed in "${seeds[@]}"; do
        for threshold in "${thresholds[@]}"; do
            mut_seed_folder=${current_folder}seed-${seed}/scratch-mute-freq-${mut}/hilary-standard-method-nt$threshold/single_chain/
            echo $mut_seed_folder
            mkdir -p $mut_seed_folder
            infer-lineages crude-method ${current_folder}seed-$seed/scratch-mute-freq-$mut/simulations/single_chain/igh.tsv -vv --config ${current_folder}simu_to_hilary.json --result-folder $mut_seed_folder --json -t -1 --override -nt $threshold > ${mut_seed_folder}logs.json
        done
    done
done
