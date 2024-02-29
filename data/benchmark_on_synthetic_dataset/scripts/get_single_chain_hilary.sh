#!/bin/bash

current_folder=/home/gathenes/gitlab/HILARy/data/benchmark_on_synthetic_dataset/subsampled_simulations/

# Loop through the array
for set in {1..5} ; do
    for l in {18..45..3}; do
        echo $set $l
        result_folder=/home/gathenes/gitlab/HILARy/data/benchmark_on_synthetic_dataset/results_hilary_p99/set${set}
        log_folder=${result_folder}/logs
        mkdir -p $result_folder
        mkdir -p $log_folder
        infer ${current_folder}/families_cdr3l_${l}_1e5_set_${set}.csv.gz -vv --config ${current_folder}simu_to_hilary.json --result-folder ${result_folder} --json -t -1 -p 0.99 -s 0.9 > ${log_folder}/logs_${l}.json
    done
done
