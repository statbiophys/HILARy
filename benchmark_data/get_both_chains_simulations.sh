#!/bin/bash

mutation_frequencies=("0.01" "0.05" "0.10" "0.20" "0.30")
seeds=("0" "1" "2")
chains=("h" "k")


# Loop through the array
for mut in "${mutation_frequencies[@]}"; do
    for seed in "${seeds[@]}"; do
        for chain in "${chains[@]}"; do
            mkdir -p ./seed-${seed}/scratch-mute-freq-${mut}/simulations/both_chains/
            python /home/gabrielathenes/Documents/study/partis/bin/parse-output.py ~/Downloads/for-zenodo-vs-shm-v2/vs-shm/v2/seed-$seed/scratch-mute-freq-$mut/simu/igh+igk/ig$chain.yaml ./seed-${seed}/scratch-mute-freq-${mut}/simulations/both_chains/ig$chain.tsv --airr-output --extra-columns cdr3_length:naive_seq:v_gl_seq:j_gl_seq:v_qr_seqs:j_qr_seqs
        done
    done
done
