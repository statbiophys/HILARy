#!/bin/bash

current_folder=/home/gathenes/gitlab/HILARy/data_with_scripts/benchmark_on_synthetic_dataset/subsampled_simulations/

# Loop through the array
for set in {1..5} ; do
    for l in {15..45..3}; do
        echo $set $l
        result_folder=/home/gathenes/gitlab/HILARy/data_with_scripts/benchmark_on_synthetic_dataset/hilary-full-1-2-0/set${set}
        log_folder=${result_folder}/logs
        mkdir -p $result_folder
        mkdir -p $log_folder
        infer-lineages full-method ${current_folder}/families${set}_1e4_ppost326651_mut326713_cdr3l${l}.csv.gz -vv --config ${current_folder}simu_to_hilary.json --result-folder ${result_folder} --json -t -1 --override > ${log_folder}/logs_${l}.json
    done
done
