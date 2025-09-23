# Based on code from Natanael Spisak
import os

import pandas as pd

pd.options.mode.chained_assignment = None
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Iterable

import numpy as np
import righor
import sonnia.sonia
from generate_conditional import generate_ppost_seqs
from numpy.random import zipf
from sonnia.utils import gene_to_num_str
from textdistance import hamming
from tqdm import tqdm

from hilary.utils import apply_parallel
from utils_simulate import mutate, mutate2, nt2aa


class Simulator:
    """
    Simulate families given Ppost naive sequences and mutational spectra from known families.
    Attributes:
        l (int): Length of the sequences.
        scale (float): Scale factor based on sequence length.
        mutModel (int): Mutation model identifier.
        spectra (Spectra): Spectra object for mutation spectra.
        genModel (int): Generation model identifier.
        genes (Genes): Genes object for gene assignment.
        generator (Generator): Generator object for sequence generation.
        nbOfFamilies (int): Number of families to simulate.
        smallSizes (array): Sizes of small families.
        nbLarge (int): Number of large families.
        outputMacro (str): Output file pattern for saving results.
    Methods:
        read():
            Load generated sequences (naive) generated with self.genModel.
        mutateFamily(naive, familySize, which, howMany):
            Mutate naive sequence to get family of size familySize with mutation pattern which/howMany.
        smallFamilies(args):
            Simulate small families.
        largeFamilies(rootsLarge):
            Simulate large families.
        simulate():
            Simulate clonal families and store the dataset.
        save():
            Save the simulated dataset to a file.
    """

    def __init__(self, threshold: int = 10, model_directory: str = "human_B_heavy") -> None:
        """Initialize the Simulator with the specified parameters."""
        self.threshold = threshold  # threshold to define small and big families
        default_models = ["human_B_heavy", "mouse_B_heavy", "human_B_kappa", "human_B_lambda"]
        self.sonia_model = sonnia.sonia.Sonia(ppost_model=model_directory)
        if model_directory in default_models:
            model_directory = os.path.join(
                os.path.dirname(sonnia.sonia.__file__), "default_models", model_directory
            )
        self.righor_model = righor.load_model_from_files(
            path_params=os.path.join(model_directory, "model_params.txt"),
            path_marginals=os.path.join(model_directory, "model_marginals.txt"),
            path_anchor_vgene=os.path.join(model_directory, "V_gene_CDR3_anchors.csv"),
            path_anchor_jgene=os.path.join(model_directory, "J_gene_CDR3_anchors.csv"),
        )

    def simulate(
        self,
        nbOfFamilies: int,
        alpha: float | None = None,
        max_threshold: int | None = 1000,
        cdr3_selection: int | None = None,
        available_j: Iterable | None = None,
        available_v: Iterable | None = None,
        subtype: str = "mixture",
        mut_directory: str | None = None,
        roots = None
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

        self.small = pd.read_csv(mut_directory + "small_spectra.csv.gz")
        self.large = pd.read_csv(mut_directory + "large_spectra.csv.gz")
        ns = zipf(alpha, size=int(nbOfFamilies * ratios[0]))
        ns = np.concatenate([ns, np.ones(int(nbOfFamilies * ratios[1]))]).astype(int)
        # clip max family size
        ns = np.clip(ns, 0, max_threshold)
        self.smallSizes = ns[ns <= self.threshold]
        self.nbLarge = nbOfFamilies - len(self.smallSizes)
        if roots is None:
            roots= self.generate_naive(
                nbOfFamilies, cdr3_selection, available_j=available_j, available_v=available_v
            )
        print("roots")
        print(roots)
        self.rootsLarge, self.rootsSmall = roots[: self.nbLarge], roots[self.nbLarge :]
        self.familiesLarge = self.largeFamilies(self.rootsLarge)
        print("large families")
        print(self.familiesLarge)
        if self.nbLarge > 0:
            self.familiesLarge["family"] = self.familiesLarge["family"] + 1
        self.familiesSmall = self.smallFamilies()
        self.familiesSmall["family"] = self.familiesSmall["family"] + (1 + self.nbLarge)
        self.dataset = pd.concat([self.familiesLarge, self.familiesSmall], ignore_index=True)
        self.dataset = produce_alt_alignments(self.dataset)
        self.dataset = self.dataset.loc[
            self.dataset.sequence.apply(lambda x: "*" not in nt2aa(x))
        ]  # get rid of stop codons
        self.dataset = self.dataset.loc[
            self.dataset.mutation_count.apply(lambda x: x < 70)
        ]  # get rid of too many mutations
        return self.dataset

    def generate_naive(
        self,
        n: int = 1,
        cdr3_selection: int | None = None,
        available_j: Iterable | None = None,
        available_v: Iterable | None = None,
    ) -> (pd.DataFrame, pd.DataFrame):
        """Sample generated sequences (naive)"""
        print("Generate naive sequences.")
        df = generate_ppost_seqs(
            self.sonia_model,
            self.righor_model,
            n_seqs=n*2,
            available_j=available_j,
            available_v=available_v,
        )
        df = df[["v_gene", "j_gene", "cdr3_length", "cdr3"]].astype({"cdr3_length": int}).dropna()
        if cdr3_selection is not None:
            df = df.loc[df["cdr3_length"] == cdr3_selection]

        # the assign sequence function might have some issues in matching the right v gene for
        # reconstruction of the sequence. So I am hiding these edge cases by sampling more and select
        # only what it works
        df[["sequence", "cdr3_anchor"]] = (
            df[["v_gene", "j_gene", "cdr3"]]
            .apply(self.assignSequence, axis=1, result_type="expand")
        ) ## hiding a bug here, the dropna shouldn't be necessary
        # here the reconstruction doesn't have to be a multiple of three, so I enforce it
        df=df.dropna()
        df['sequence']=df['sequence'].apply(lambda x: x[:(len(x)//3)*3])
        ## hiding a bug here, they should be all productive
        df=df.loc[df.sequence.apply(lambda x: "*" not in nt2aa(x))].reset_index(drop=True)
        return df[:n]

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
        self.rootsSmall["family_size"] = self.smallSizes
        self.rootsSmall = self.rootsSmall.sort_values("family_size").reset_index(drop=True)
        print("Generate small families.")
        # sample mutation spectra
        spectra_df = pd.DataFrame()
        for _, (familySize, n_samples) in tqdm(
            self.rootsSmall.family_size.value_counts().reset_index().iterrows()
        ):
            spectra = (
                self.small.loc[self.small["family_size"] == familySize]
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
        return apply_parallel(
            self.rootsSmall.groupby(["family_size", "cdr3_length", "j_gene", "v_gene"]),
            small_family_parallel,
            cpu_count=cpu_count(),
        )

    def largeFamilies(self, rootsLarge):
        """
        Simulate large families.
        This method simulates large families based on the provided roots and spectra data.
        It generates mutated sequences for each family and constructs a DataFrame containing
        the family information.
        Parameters:
            rootsLarge (pd.DataFrame): A DataFrame containing the root information for large families.
                Expected columns are:
                - 'v_gene': V gene segment
                - 'j_gene': J gene segment
                - 'cdr3_length': Length of the CDR3 region
                - 'naive': Naive sequence
                - 'anchor': Anchor position
        Returns:
            pd.DataFrame: A DataFrame containing the simulated family information with the following columns:
                - 'sequence': Mutated sequence for each family member
                - 'germline': Naive sequence (germline)
                - 'v_gene': V gene segment
                - 'j_gene': J gene segment
                - 'cdr3_length': Length of the CDR3 region
                - 'cdr3': CDR3 sequence
                - 'family': Family index
        """
        rootsLarge.reset_index(drop=True, inplace=True)
        spectra = self.large.sample(len(rootsLarge), replace=True).reset_index(drop=True)
        families = pd.DataFrame()
        print("Generate large families.")
        for index, (v, j, l, _, naive, anchor) in tqdm(rootsLarge.iterrows()):
            anchor = int(anchor)
            familySize, configurations, nbs = spectra.loc[index]
            which = np.array([list(c) for c in configurations.split(":")]).astype(int)
            howMany = np.array(nbs.split(":")).astype(int)
            order = np.argsort(-which.sum(axis=1))
            family = pd.DataFrame()
            family["sequence"] = mutateFamily(naive, familySize, which[order], howMany[order])
            family["germline"] = naive
            family["v_gene"] = v
            family["j_gene"] = j
            family["cdr3_length"] = l
            family["cdr3"] = family["sequence"].str[anchor + 3 : anchor + 3 + int(l)]
            family["family"] = index
            families = pd.concat([families, family], ignore_index=True)
        return families

    def assignSequence(self, args):
        """Return full sequence and CDR3 anchor position"""
        V, J, ntcdr3 = args
        try:
            V = self.sonia_model.pgen_model.V_mask_mapping[gene_to_num_str(V, "V")][0]
            J = self.sonia_model.pgen_model.J_mask_mapping[gene_to_num_str(J, "J")][0]
        except:
            return np.nan, np.nan
        fullV_gene = self.sonia_model.genomic_data.genV[V][2]
        endV = -len(self.sonia_model.genomic_data.genV[V][1])
        begin = fullV_gene[: endV + 3]
        fullJ_gene = self.sonia_model.genomic_data.genJ[J][2]
        beginJ = len(self.sonia_model.genomic_data.genJ[J][1])
        end = fullJ_gene[beginJ - 3 :]
        ntseq = begin + ntcdr3 + end
        return ntseq, len(begin) - 3

    def save(self, filename="synthetic_families.csv"):
        """Save the simulated dataset to a file"""
        dataset = pd.concat([self.familiesLarge, self.familiesSmall], ignore_index=True)
        dataset.sample(frac=1).to_csv(filename, index=False)


def mutateFamily(naive, familySize, which, howMany):
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
    # creates an array of naive sequences
    sequences = np.array([naive] * int(familySize))
    length = len(naive)
    scale = (
        length / 250.0
    )  # Briney templated alignment length is 250, full sequence is 320+l (on average)
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
    return sequences


def small_family_parallel(args) -> pd.DataFrame:
    """
    Simulates clonal families in parallel.

    Args:
        args (tuple): A tuple containing:
            - sign (list): A list where the first element is the family size.
            - rootsSmall (pd.DataFrame): A DataFrame containing the root sequences and other related information.
    Returns:
        pd.DataFrame: A DataFrame containing the simulated clonal families with the following columns:
            - 'sequence': The mutated sequences of the family.
            - 'germline': The naive sequence.
            - 'v_gene': The V gene segment.
            - 'j_gene': The J gene segment.
            - 'cdr3_length': The length of the CDR3 region.
            - 'cdr3': The CDR3 region of the sequence.
            - 'family': The index of the family.
    """
    sign, rootsSmall = args
    familySize = int(sign[0])
    families = pd.DataFrame()
    for index, (v, j, cdr3_length, _, naive, anchor, _, spectrum) in rootsSmall.iterrows():
        family = pd.DataFrame()
        if sum(spectrum) == 0:
            family["sequence"] = [naive] * familySize
        else:
            mask = spectrum.nonzero()[0]
            which = np.array([list(bin(m + 1)[2:].zfill(familySize)) for m in mask]).astype(int)
            howMany = spectrum[mask]
            order = np.argsort(-which.sum(axis=1))
            family["sequence"] = mutateFamily(naive, familySize, which[order], howMany[order])
        family["germline"] = naive
        family["v_gene"] = v
        family["j_gene"] = j
        family["cdr3_length"] = cdr3_length
        family["cdr3"] = family["sequence"].str[anchor + 3 : anchor + 3 + int(cdr3_length)]
        family["family"] = index
        families = pd.concat([families, family], ignore_index=True)
    return families


def produce_alt_alignments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces alternative alignments for sequences and germlines by removing the CDR3 region.

    Args:
        df (pandas.DataFrame): A DataFrame containing the columns 'sequence', 'cdr3', and 'germline'.

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns:
            - 'cdr3_length': Length of the CDR3 region.
            - 'cdr3_start': Starting position of the CDR3 region in the sequence.
            - 'cdr3_end': Ending position of the CDR3 region in the sequence.
            - 'alt_sequence_alignment': Sequence with the CDR3 region removed.
            - 'alt_germline_alignment': Germline with the CDR3 region removed.
    """

    df["cdr3_length"] = df.cdr3.apply(len)
    df["cdr3_start"] = (
        df[["sequence", "cdr3"]].apply(lambda x: str(x[0]).find(x[1]), axis=1).astype(int)
    )
    df["cdr3_end"] = df["cdr3_start"] + df["cdr3_length"]
    df["alt_sequence_alignment"] = df[["sequence", "cdr3_start", "cdr3_end"]].apply(
        lambda x: (x[0][: x[1]] + x[0][x[2] :]), axis=1
    )
    df["alt_germline_alignment"] = df[["germline", "cdr3_start", "cdr3_end"]].apply(
        lambda x: (x[0][: x[1]] + x[0][x[2] :]), axis=1
    )
    df["mutation_count"] = df[["alt_sequence_alignment", "alt_germline_alignment"]].apply(
        lambda x: hamming(*x), axis=1
    )
    return df
