### Synthetic data for families inference benchmark

This is the dataset used to compare different methods for families inference, results presented in Figure 4 "Benchmark of the alternative methods" in

1. *Combining mutation and recombination statistics to infer clonal families in antibody repertoires* by Spisak, Dupic, Mora, and Walczak, 2022, https://doi.org/10.1101/2022.12.22.521661 

The data was generated as described therein in section IV E ("Methods: Synthetic data generation"). Briefly, we drew 100000 unmutated sequences from the Ppost distribution (Isacchini et al. 2021) and simulated mutation process mimicking the mutation landscape found in families inferred at high precision in long-CDR3 subpart of a real dataset (IgG repertoire of donor 326651 from Briney et al. 2019). 

To analyze the pairwise sensitivity and precision of methods, we used subsamples of this dataset (10000 unique sequences) in order to compare performance of fast and slow methods together. Independent test of inference time was performed using real data (Figure 4A).

2. *Deep generative selection models of T and B cell receptor repertoires with soNNia*
   Isacchini, Walczak, Mora, and Nourmohammad, 2021, https://doi.org/10.1073/pnas.2023141118
3. *Commonality despite exceptional diversity in the baseline human antibody repertoire*,  Briney, Inderbitzin,  Joyce, and Burton, 2019, https://doi.org/10.1038/s41586-019-0879-y

