# 0. Installation

`pip install hilary`

# 1. Usage

### 1.1 Inputs

Inputs needs to be a tsv or excel file in airr format, meaning with the following columns :

| sequence_id | v_call      | j_call   | junction  | v_sequence_alignment | j_sequence_alignment | v_germline_alignment | j_germline_alignment |
| ----------- | ----------- | -------- | --------- | -------------------- | -------------------- | -------------------- | -------------------- |
| 1           | IGHV1-34*01 | IGHJ3*01 | TGTGCAACC | TTAGTACTT            | TTGCTTACT            | AGCACAGCC            | TTGCTTACT            |
| 2           | IGHV1-18*01 | IGHJ4*01 | TGTGCAAGA | TTAATCCTA            | GCTATGGAC            | TTAATCCTA            | GCTATGGAC            |
| 3           | IGHV1-74*01 | IGHJ4*01 | TGTGCAAGA | CATGCAACT            | GCTATGGAC            | CTACAATCA            | GCTATGGAC            |
| 4           | IGHV5-17*01 | IGHJ4*01 | TGTGCAAGA | CCCTGTTCC            | CTATGCTATGG          | GAGGTGTTC            | CTATGCTAT            |

### 1.2 From the command line

`infer --help`

```
Arguments:
  DATA_PATH  Path of the excel file to infer lineages.  [required]

Options:
  -v, --verbose            Set logging verbosity level.  [default: 0]
  -t, --threads INTEGER    Choose number of cpus on which to run code.
                           [default: all available]
  -p, --precision FLOAT    Choose desired precision.  [default: 0.99]
  -s, --sensitivity FLOAT  Choose desired sensitivity.  [default: 0.9]
  --nmin INTEGER           Infer prevalence and mu on classes of size larger than nmin.
                            Mean prevalence is assigned to lower than nmin classes.  [default: 1000]
  -m, --model INTEGER      Model name to infer Null distribution.  [default: 326713]
  --silent                 Do not show progress bars if used.
  --result-folder PATH     Where to save the result files. By default it will be saved in a
                            'result/' folder in input data's parent directory.
  --config PATH            Configuration file for column names. File should be a json with keys as
                            your data's column names and values as hilary's required column names.
  --help                   Show this message and exit.
```

**example :** `infer /home/gabrielathenes/Documents/study/test.xlsx --nmin 10000 -vv --result-folder ../hilary_test`

### 1.3 From Python

See `tutorial.ipynb`

# 2. Functional description of HILARy

## 2.1 CDR3-based inferrence method with adaptive threshold
### Step 1
1. Sequences are first filtered (are removed non productive sequences, null values ect) and then grouped by VJl class (sequences having same V gene, J gene and CDR3 length).
2. For each VJl class, the histogram of pairwise distances is computed.
3. We hypothesize that for a given VJl class, the distribution of pairwise distances $P$ is the $\rho$ weighted average of two distributions, a Poisson distribution $P_\mu \sim Pois(l\mu)$ representing related sequences and a null distribution $P_0$ representing non related sequences and identical for all classes and computed using Sonnia.
$$P(x)=\rho P_\mu + (1-\rho) P_0$$
Please note that even though $P_\mu$ is of parameter $l\mu$, only $\mu$ needs to be inferred as $l$ is known.
We finally estimate $\rho$ and $\mu$ for each class using an expectation-maximization algorithm.

**Summary of step 1**

![Step 1](./doc/CDR3_clustering1.png)

### Step 2

4. For a given class, we can now compute precision and sensitivity just from the inferred distribution $P$ (we know the distribution of related sequences $P_\mu$, the distribution of unrelated sequences $P_0$ and the weight $\rho$.)
5. For a given precision $\pi^{\star}$ we compute a threshold $t^\star$.
6. This threshold used by a single clustering algorithm to build a partition with precision $\pi^{\star}$. The single linkage algorithm adds a sequence $s_1$ in a cluster if a member $s_2$ is such that the hamming distance of the CDR3s of $s_1$ and $s_2$ is smaller than $l t^{\star}$. (Note that since inside a VJl class their CDR3s have same length $l$.)

**Summary of step 2**

![Step 2](./doc/CDR3clustering_2.png)

## 2.2 Incorporating phylogenetic signal

For a wide range of parameters, the method is predicted to achieve both high precision and high sensitivity. However, it is expected to fail when the prevalence and the CDR3 length are both low. HILARy therefore uses the number of shared mutations to upgrade sensitivity for low.

For each class, compute a high sensitivity (>90%) partition exactly like in step 2 but replacing precision with sensitivity. If the partition coincides with a high precision partition, then the partition is precise and sensitive and nothing needs to be done. Otherwise, we make the partition more precise by removing false positives. To do so we compute two variables $x'$ and $y$ coding respectfully for CDR3 divergence and number of mutations. We then classify pairs as related when $y-x'> t$ (resp. unrelated when <) with $t$ chosen to achieve high precision similarly than for the CDR3-based method.

**Summary**
Suppose we represent related sequences left (+ signs) and unrelated sequences right (o signs) on a plane. The rectangle-like shape is the estimation of positive sequences. In case the CDR3-based sensitive partition is not precise enough, the mutation based method upgrades the partition method by removing false positives.
![Summary](./doc/mutations.png)

# TODO
