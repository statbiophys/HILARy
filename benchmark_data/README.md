# Benchmarking report

We benchmark HILARy with the data used in figure 3 from article [Inference of B cell clonal families using heavy/light chain pairing information](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010723). In order to do that we follow instructions from their [zenodo](https://zenodo.org/records/6998443) web page and download folder `for-zenodo-vs-shm-v2` as deduced from line 14 of this [script](https://github.com/psathyrella/partis/blob/7c7ec8981ca55bdaf8139fb5692a56382f050dca/bin/run-paired-loci.sh).


## Data structure

Let's display part of the data structure using linux `tree` command.

`$ tree for-zenodo-vs-shm-v2/vs-shm/v2/seed* -L 2 -I 'synth* vjc*|mobille|vs*|*0.10|*0.20|*0.30|seed-2'`

```
for-zenodo-vs-shm-v2/vs-shm/v2/seed-0
├── scratch-mute-freq-0.01
│   ├── partis
│   ├── scoper
│   └── simu
└── scratch-mute-freq-0.05
    ├── partis
    ├── scoper
    └── simu
for-zenodo-vs-shm-v2/vs-shm/v2/seed-1
├── scratch-mute-freq-0.01
│   ├── partis
│   ├── scoper
│   └── simu
└── scratch-mute-freq-0.05
    ├── partis
    ├── scoper
    └── simu
for-zenodo-vs-shm-v2/vs-shm/v2/seed-2
├── scratch-mute-freq-0.01
│   ├── partis
│   ├── scoper
│   └── simu
└── scratch-mute-freq-0.05
    ├── partis
    ├── scoper
    └── simu
```

For three seeds, `scratch-mute-freq-<i>/simu` is the folder containing simulations of mutation frequency `<i>`. `scratch-mute-freq-<i>/<model_name>` contains the partitions inferred from model `<model_name>` on these simulations.

## Retrieving partitions

- For scoper, the partitions were already in `.tsv` format. (From the `scoper.log` file we know the 2020 version was used). We therefore copied these files in this folder.
  - Scripts : `get_both_chains_scoper.sh` & `get_single_chain_scoper.sh`
- For partis and simulations, we used the [code](https://github.com/psathyrella/partis/blob/main/bin/parse-output.py) to convert the `.yaml` partitions in `.tsv` format. **Note** This code needs to be run in a python 2 environment.
  - Scripts : `get_both_chains_simulations.sh` `get_single_chain_simulations.sh` `get_both_chains_partis.sh` &`get_single_chain_partis.sh`
- We then ran HILARy on the simulations to compare to partis and scoper. **Note** First install HILARy with `pip install hilary`
  - Script : `get_single_chain_hilary.sh`

## Results

The clonal family of each sequence is found in column `naive_seq` for the simulations, `clone_id` for partis and scoper and `family` for HILARy. We finally compared clonal families inferred from scoper, partis and hilary to the ground truth : the `naive_seq` column of the simulations.
