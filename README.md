# HILARy
Python package for high-precision inference of lineages in antibody repertoires (HILARy). 



### Methods

This package implements the methods described in [Combining mutation and recombination statistics to infer clonal families in antibody repertoires](https://doi.org/10.1101/2022.12.22.521661), including:

1. A priori estimation of prevalence, the fraction of pairs in the dataset linking sequences belonging to the same clonal family (module `apriori.py`).

2. Fast CDR3-based clustering with fixed precision/sensitivity (class `CDR3Clustering` in module `inference.py`).

3. Full method relying on information encoded in the CDR3 as well as phylogenetic signal encoded outside the CDR3 (class `HILARy` in module `inference.py`).

4. Evaluation of inference results (module `aposteriori.py`)

### Prerequisites 

The dependencies can be installed with

``` pip install sonnia ``` (see [soNNia](https://github.com/statbiophys/soNNia))

``` pip install atriegc ``` (see [ATrieGC](https://github.com/statbiophys/ATrieGC))


### Usage

The input are aligned sequences in [AIRR-compatible format](https://docs.airr-community.org/en/stable/datarep/rearrangements.html), a tab-separated file with the following columns 

Name | Example
--- | ---
sequence_id  | 1
v_call | IGHV1-2\*01
j_call | IGHJ1\*01
junction | TGTCATGCGATTAACAGCGCGTGG
v_sequence_alignment | TCTGACGACACGGCCGTATATTACTGT
j_sequence_alignment | TGGGGCCGGGGGACC
v_germline_alignment | TCTGACGACACGGCCGTGTATTACTGT
j_germline_alignment | TGGGGCCAGGGCACC

See `inference.ipynb` for example pipeline.


### Contact

Email [natanael.spisak@gmail.com](mailto:natanael.spisak@gmail.com)
