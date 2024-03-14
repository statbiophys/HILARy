# 1. Benchmark report of the [HILARy paper](https://doi.org/10.1101/2022.12.22.521661)

:warning: All benchmark is done with version 1.2.0. To install it :
`pip install hilary==1.2.0`

There are three uses of HILARy mentionned in the article.

1. On data from [Briney](https://www.nature.com/articles/s41586-019-0879-y). This concerns figure 6 of the article. The philogenies inferred with HILARy are obtained using `HILARy/tutorial.ipynb`. The working directory of this tutorial is `./data_from_briney/`, where data will be downloaded and lineages inferred.

2. On the dataset used in article [Inference of B cell clonal families using heavy/light chain pairing information](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010723). The working directory of this benchmark is `./benchmark_on_partis_dataset/`, where all scripts used for inference can be found.

3. On our synthetic data (section II.C of the article). The working directory of this benchmark is `./benchmark_on_synthetic_dataset/`, where all scripts used for inference can be found.

In order not to put data on github we upload it on zenodo using the same folder architecture than `data_with_scripts`.

Finally,
- `./article_figure_scripts/` contains scripts ot reproduce the figures of the article.
- `./null_distributions/` contains the null distributions used for $P_F$ and computed by Sonnia.
- `./human_genes/` contains the V gene and J gene to process data from Briney for lineage inference.


# 2. Benchmark on the dataset from [partis](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010723).

We benchmark HILARy with the data used in figure 3 from article [Inference of B cell clonal families using heavy/light chain pairing information](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010723). In order to do that we follow instructions from their [zenodo](https://zenodo.org/records/6998443) web page and download folder `for-zenodo-vs-shm-v2` as deduced from line 14 of this [script](https://github.com/psathyrella/partis/blob/7c7ec8981ca55bdaf8139fb5692a56382f050dca/bin/run-paired-loci.sh).


#### 2.1 Data structure

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

#### 2.2 Retrieving partitions

- For scoper, the partitions were already in `.tsv` format. (From the `scoper.log` file we know the 2020 version was used). We therefore copied these files in this folder.
  - Scripts : `get_both_chains_scoper.sh` & `get_single_chain_scoper.sh`
- For partis and simulations, we used the [code](https://github.com/psathyrella/partis/blob/main/bin/parse-output.py) to convert the `.yaml` partitions in `.tsv` format. **Note** This code needs to be run in a python 2 environment.
  - Scripts : `get_both_chains_simulations.sh` `get_single_chain_simulations.sh` `get_both_chains_partis.sh` &`get_single_chain_partis.sh`
- We then ran HILARy on the simulations to compare to partis and scoper. **Note** First install HILARy with `pip install hilary`
  - Script : `get_single_chain_hilary.sh`

#### 2.3 Results

The clonal family of each sequence is found in column `naive_seq` for the simulations, `clone_id` for partis and scoper and `family` for HILARy. We finally compared clonal families inferred from scoper, partis and hilary to the ground truth : the `clone_id` column of the simulations.

# 3. Benchmark on our synthetic dataset

### Synthetic data for families inference benchmark

This is the dataset used to compare different methods for families inference, results presented in Figure 4 "Benchmark of the alternative methods" in

1. *Combining mutation and recombination statistics to infer clonal families in antibody repertoires* by Spisak, Dupic, Mora, and Walczak, 2022, https://doi.org/10.1101/2022.12.22.521661

The data was generated as described therein in section IV E ("Methods: Synthetic data generation"). Briefly, we drew 100000 unmutated sequences from the Ppost distribution (Isacchini et al. 2021) and simulated mutation process mimicking the mutation landscape found in families inferred at high precision in long-CDR3 subpart of a real dataset (IgG repertoire of donor 326651 from Briney et al. 2019).

To analyze the pairwise sensitivity and precision of methods, we used subsamples of this dataset (10000 unique sequences) in order to compare performance of fast and slow methods together. Independent test of inference time was performed using real data (Figure 4A).

2. *Deep generative selection models of T and B cell receptor repertoires with soNNia*
   Isacchini, Walczak, Mora, and Nourmohammad, 2021, https://doi.org/10.1073/pnas.2023141118
3. *Commonality despite exceptional diversity in the baseline human antibody repertoire*,  Briney, Inderbitzin,  Joyce, and Burton, 2019, https://doi.org/10.1038/s41586-019-0879-y
