# Based on code from Natanael Spisak
import os

import pandas as pd

pd.options.mode.chained_assignment = None
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Iterable

import numpy as np
from numpy.random import zipf
from hilary.simulate import Simulator
from tqdm import tqdm
from hilary.utils_simulate import mutate, mutate2,nt2aa
from textdistance import hamming
from hilary.utils import applyParallel


class SimulatorPaired:

    """
    Simulate families given Ppost naive sequences and mutational spectra from known families.
    Attributes:
        mut_directory_heavy: str = 'mutations/'
        mut_directory_light: str = 'mutations/'
        threshold:  int = 10
        model_directory_heavy: str ='human_B_heavy'
        model_directory_light: str ='human_B_kappa'
    Methods:
        smallFamilies(args):
            Simulate small families.
        largeFamilies(rootsLarge):
            Simulate large families.
        simulate():
            Simulate clonal families and store the dataset.
        save():
            Save the simulated dataset to a file.
    """

    def __init__(
        self,
        threshold: int = 10,
        model_directory_heavy: str = "human_B_heavy",
        model_directory_light: str = "human_B_kappa",
    ) -> None:
        self.threshold = threshold
        self.heavy_simulator = Simulator(threshold=threshold, model_directory=model_directory_heavy)
        self.light_simulator = Simulator(threshold=threshold, model_directory=model_directory_light)

    def simulate(
        self,
        nbOfFamilies: int,
        alpha: float | None = None,
        subtype: str = "mixture",
        max_threshold: int | None = 1000,
        mut_directory: str | None = None,
    ) -> pd.DataFrame:
        """This method generates clonal families based on the specified number of families and the Zipf distribution parameter.
        It separates the families into large and small based on a threshold, generates naive sequences for each family,
        and then simulates the families. The results are stored in the `dataset` attribute.

        Parameters:
            nbOfFamilies (int): Number of families to simulate.
            alpha (float, optional): Parameter for the Zipf distribution. Default is 2.3.
            cdr3_selection (optional): Selection criteria for CDR3 sequences. Default is None.

        Returns:
            pd.DataFrame: A DataFrame containing the simulated clonal families.
        """
        file_location = os.path.dirname(__file__)
        if subtype.lower() == "igg":
            ratios = [5 / 7, 2 / 7]
            if alpha is None:
                alpha = 2.3
            if mut_directory is None:
                mut_directory = os.path.join(file_location,"mutations/326713_igg_")
        elif subtype.lower() == "igm":
            ratios = [1 / 12, 11 / 12]
            if alpha is None:
                alpha = 2.2
            if mut_directory is None:
                mut_directory = os.path.join(file_location,"mutations/326713_igm_")
        elif subtype.lower() == "mixture":
            ratios = [1 / 8, 7 / 8]
            if alpha is None:
                alpha = 2.25
            if mut_directory is None:
                mut_directory = os.path.join(file_location,"mutations/326713_")
        else:
            raise ValueError(f"Unsupported subtype: {subtype}")

        # same mutation for both for now
        self.small = pd.read_csv(mut_directory + "small_spectra.csv.gz")
        self.large = pd.read_csv(mut_directory + "large_spectra.csv.gz")

        ns = zipf(alpha, size=int(nbOfFamilies * ratios[0]))
        ns = np.concatenate([ns, np.ones(int(nbOfFamilies * ratios[1]))]).astype(int)
        # clip max family size
        ns = np.clip(ns, 0, max_threshold)

        # define small and large families
        self.smallSizes = ns[ns <= self.threshold]

        self.heavy_simulator.nbLarge = nbOfFamilies - len(self.smallSizes)
        self.light_simulator.nbLarge = nbOfFamilies - len(self.smallSizes)

        # generate roots
        self.generate_naive(nbOfFamilies)

        print(self.heavy_simulator.nbLarge, self.smallSizes)

        print(len(self.rootsLarge_heavy), len(self.rootsSmall_heavy))
        print(len(self.rootsLarge_light), len(self.rootsSmall_light))

        # generate mutations
        self.familiesLarge = self.largeFamilies()
        if self.heavy_simulator.nbLarge > 0:
            self.familiesLarge["family"] = self.familiesLarge["family"] + 1
        self.familiesSmall = self.smallFamilies()
        self.familiesSmall["family"] = self.familiesSmall["family"] + (
            1 + self.heavy_simulator.nbLarge
        )
        self.dataset = pd.concat([self.familiesLarge, self.familiesSmall], ignore_index=True)
        self.dataset = produce_alt_alignments_paired(self.dataset)
        # some filtering
        self.dataset = self.dataset.loc[
            self.dataset.sequence_heavy.apply(lambda x: "*" not in nt2aa(x))
        ]  # get rid of stop codons
        self.dataset = self.dataset.loc[
            self.dataset.mutation_count_heavy.apply(lambda x: x < 70)
        ]  # get rid of too many mutations
        self.dataset = self.dataset.loc[
            self.dataset.sequence_light.apply(lambda x: "*" not in nt2aa(x))
        ]  # get rid of stop codons
        self.dataset = self.dataset.loc[
            self.dataset.mutation_count_light.apply(lambda x: x < 70)
        ]  # get rid of too many mutations

        return self.dataset

    def generate_naive(self, nbOfFamilies):
        """Sample generated sequences (naive)"""

        self.rootsLarge_heavy, self.rootsSmall_heavy = self.heavy_simulator.generate_naive(
            nbOfFamilies
        )
        self.rootsLarge_light, self.rootsSmall_light = self.light_simulator.generate_naive(
            nbOfFamilies
        )

    def largeFamilies(self):
        """
        Simulate large families.
        This method simulates large families based on the provided roots and spectra data.
        It generates mutated sequences for each family and constructs a DataFrame containing
        the family information.

        Returns:
            pd.DataFrame,pd.DataFrame: A DataFrame containing the simulated family information.
        """
        self.rootsLarge_heavy.reset_index(drop=True, inplace=True)
        self.rootsLarge_light.reset_index(drop=True, inplace=True)

        spectra = self.large.sample(
            len(self.rootsLarge_heavy), replace=False
        ).reset_index(drop=True)
        families = pd.DataFrame()
        print("Generate large families.")
        zipped_iter = zip(self.rootsLarge_heavy.iterrows(), self.rootsLarge_light.iterrows())
        for (
            (index, (v_heavy, j_heavy, l_heavy, _, naive_heavy, anchor_heavy)),
            (index_1, (v_light, j_light, l_light, _, naive_light, anchor_light)),
        ) in tqdm(zipped_iter):
            familySize, configurations, nbs = spectra.loc[index]
            which = np.array([list(c) for c in configurations.split(":")]).astype(int)
            howMany = np.array(nbs.split(":")).astype(int)
            order = np.argsort(-which.sum(axis=1))
            family = pd.DataFrame()
            family["sequence_heavy"], family["sequence_light"] = mutateFamilyPaired(
                naive_heavy, naive_light, familySize, which[order], howMany[order]
            )
            family["germline_heavy"], family["germline_light"] = naive_heavy, naive_light
            family["v_gene_heavy"], family["v_gene_light"] = v_heavy, v_light
            family["j_gene_heavy"], family["j_gene_light"] = j_heavy, j_light
            family["cdr3_length_heavy"], family["cdr3_length_light"] = l_heavy, l_light
            family["cdr3_heavy"] = family["sequence_heavy"].str[
                anchor_heavy + 3 : anchor_heavy + 3 + int(l_heavy)
            ]
            family["cdr3_light"] = family["sequence_light"].str[
                anchor_light + 3 : anchor_light + 3 + int(l_light)
            ]
            family["family"] = index
            families = pd.concat([families, family], ignore_index=True)
        return families

    def smallFamilies(self):
        """
        Simulate small families by sampling mutation spectra and applying parallel processing.
        This method performs the following steps:
        1. Assigns family sizes to the `rootsSmall` DataFrame.
        2. Sorts the `rootsSmall` DataFrame by family size.
        3. Samples mutation spectra for each family size.
        4. Concatenates the sampled spectra into a DataFrame.
        5. Assigns the sampled mutation spectra back to the `rootsSmall` DataFrame.
        6. Applies parallel processing to the grouped `rootsSmall` DataFrame.

        Returns:
            DataFrame: The result of applying the `small_family_parallel` function to the grouped `rootsSmall` DataFrame.
        """
        # merge heavy and light samples
        self.rootsSmall = self.rootsSmall_heavy.merge(
            self.rootsSmall_light, left_index=True, right_index=True, suffixes=("_heavy", "_light")
        )
        self.rootsSmall["family_size"] = self.smallSizes
        self.rootsSmall = self.rootsSmall.sort_values("family_size").reset_index(drop=True)
        print("Generate small families.")

        # sample mutation spectra
        spectra_df = pd.DataFrame()
        for _, (familySize, n_samples) in tqdm(
            self.rootsSmall.family_size.value_counts().reset_index().iterrows()
        ):
            spectra = (
                self.small.loc[
                    self.small["family_size"] == familySize
                ]
                .reset_index()["mutation_spectrum"]
                .apply(lambda x: str(x).split(":"))
            )
            sampled_spectra = np.random.choice(
                np.arange(len(spectra)), size=n_samples, replace=True
            )
            out = pd.DataFrame()
            out["family_size"] = [familySize] * len(sampled_spectra)
            out["mutation_spectrum"] = [np.array(s).astype(int) for s in spectra[sampled_spectra]]
            spectra_df = pd.concat([spectra_df, out])
        spectra_df = spectra_df.sort_values("family_size").reset_index(drop=True)
        self.rootsSmall["mutation_spectrum"] = spectra_df["mutation_spectrum"]
        cols = [
            "family_size",
            "cdr3_length_heavy",
            "j_gene_heavy",
            "v_gene_heavy",
            "cdr3_length_light",
            "j_gene_light",
            "v_gene_light",
        ]
        return applyParallel(
            self.rootsSmall.groupby(cols), small_family_parallel_paired, cpuCount=cpu_count()
        )

    def save(self, filename="synthetic_families.csv"):
        """Save the simulated dataset to a file"""
        self.dataset.sample(frac=1).to_csv(filename, index=False)

def mutateFamilyPaired(naive_heavy, naive_light, familySize, which, howMany):
    """
    Simulates the mutation of a clonal family of sequences.

    Parameters:
        naive (str): The naive sequence from which the family is derived.
        familySize (int): The number of sequences in the family.
        which (list of np.ndarray): A list of boolean arrays indicating which sequences to mutate.
        howMany (list of int): A list of integers indicating how many mutations to introduce for each corresponding sequence in 'which'.

    Returns:
        np.ndarray: An array of mutated sequences.
    """
    length_heavy = len(naive_heavy)
    naive = naive_heavy + naive_light
    assert not "*" in nt2aa(naive)
    sequences = np.array([naive] * int(familySize))
    length = len(naive)
    scale = length / 250.0  # Briney IGH templated alignment length is 250
    for w, nb in zip(which, howMany):
        nb_scaled = int(nb * scale)  # np.random.poisson(nb*scale)
        stop_codon, k = True, 0
        while stop_codon and k < 10000:
            seqs = sequences.copy()
            xs = np.random.randint(
                0, length, size=nb_scaled
            )  # here context/position dependence model should enter
            tos = np.random.randint(3, size=nb_scaled)  # substitution model enters here
            for x, to in zip(xs, tos):
                for i in w.nonzero()[0]:
                    seqs[i] = mutate(seqs[i], x, mutate2.get(naive[x])[to])
            stop_codons = ["*" in nt2aa(s) for s in seqs]
            stop_codon = any(stop_codons)
            k += 1
            if k == 1000:
                print("Warning: stop codon not removed")
        sequences = seqs
    seqs_heavy = [s[:length_heavy] for s in sequences]
    seqs_light = [s[length_heavy:] for s in sequences]

    return seqs_heavy, seqs_light


def small_family_parallel_paired(args) -> pd.DataFrame:
    """
    Simulates clonal families in parallel.

    Args:
        args (tuple): A tuple containing:
            - sign (list): A list where the first element is the family size.
            - rootsSmall (pd.DataFrame): A DataFrame containing the root sequences and other related information.
    Returns:
        pd.DataFrame: A DataFrame containing the simulated clonal families
    """
    sign, rootsSmall = args
    familySize = int(sign[0])
    families = pd.DataFrame()
    for index, (
        v_heavy,
        j_heavy,
        cdr3_length_heavy,
        _,
        naive_heavy,
        anchor_heavy,
        v_light,
        j_light,
        cdr3_length_light,
        _,
        naive_light,
        anchor_light,
        _,
        spectrum,
    ) in rootsSmall.iterrows():
        family = pd.DataFrame()
        if sum(spectrum) == 0:
            family["sequence_heavy"] = [naive_heavy] * familySize
            family["sequence_light"] = [naive_light] * familySize
        else:
            mask = spectrum.nonzero()[0]
            which = np.array([list(bin(m + 1)[2:].zfill(familySize)) for m in mask]).astype(int)
            howMany = spectrum[mask]
            order = np.argsort(-which.sum(axis=1))
            family["sequence_heavy"], family["sequence_light"] = mutateFamilyPaired(
                naive_heavy, naive_light, familySize, which[order], howMany[order]
            )
        family["germline_heavy"], family["germline_light"] = naive_heavy, naive_light
        family["v_gene_heavy"], family["v_gene_light"] = v_heavy, v_light
        family["j_gene_heavy"], family["j_gene_light"] = j_heavy, j_light
        family["cdr3_length_heavy"], family["cdr3_length_light"] = (
            cdr3_length_heavy,
            cdr3_length_light,
        )
        family["cdr3_heavy"] = family["sequence_heavy"].str[
            anchor_heavy + 3 : anchor_heavy + 3 + int(cdr3_length_heavy)
        ]
        family["cdr3_light"] = family["sequence_light"].str[
            anchor_light + 3 : anchor_light + 3 + int(cdr3_length_light)
        ]
        family["family"] = index
        families = pd.concat([families, family], ignore_index=True)
    return families


def produce_alt_alignments_paired(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces alternative alignments for sequences and germlines by removing the CDR3 region.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Dataframe with alt sequence alignment for both chains:
    """
    df["cdr3_length_heavy"] = df.cdr3_heavy.apply(len)
    df["cdr3_start_heavy"] = (
        df[["sequence_heavy", "cdr3_heavy"]]
        .apply(lambda x: str(x[0]).find(x[1]), axis=1)
        .astype(int)
    )
    df["cdr3_end_heavy"] = df["cdr3_start_heavy"] + df["cdr3_length_heavy"]
    df["alt_sequence_alignment_heavy"] = df[
        ["sequence_heavy", "cdr3_start_heavy", "cdr3_end_heavy"]
    ].apply(lambda x: (x[0][: x[1]] + x[0][x[2] :]), axis=1)
    df["alt_germline_alignment_heavy"] = df[
        ["germline_heavy", "cdr3_start_heavy", "cdr3_end_heavy"]
    ].apply(lambda x: (x[0][: x[1]] + x[0][x[2] :]), axis=1)
    df["cdr3_length_light"] = df.cdr3_light.apply(len)
    df["cdr3_start_light"] = (
        df[["sequence_light", "cdr3_light"]]
        .apply(lambda x: str(x[0]).find(x[1]), axis=1)
        .astype(int)
    )
    df["cdr3_end_light"] = df["cdr3_start_light"] + df["cdr3_length_light"]
    df["alt_sequence_alignment_light"] = df[
        ["sequence_light", "cdr3_start_light", "cdr3_end_light"]
    ].apply(lambda x: (x[0][: x[1]] + x[0][x[2] :]), axis=1)
    df["alt_germline_alignment_light"] = df[
        ["germline_light", "cdr3_start_light", "cdr3_end_light"]
    ].apply(lambda x: (x[0][: x[1]] + x[0][x[2] :]), axis=1)
   
    df["mutation_count_light"] = df[["alt_sequence_alignment_light", "alt_germline_alignment_light"]].apply(
        lambda x: hamming(*x), axis=1
    )
    df["mutation_count_heavy"] = df[["alt_sequence_alignment_heavy", "alt_germline_alignment_heavy"]].apply(
        lambda x: hamming(*x), axis=1
    )
    return df
