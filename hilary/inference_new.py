"""Infer clonal families."""

from __future__ import annotations

from itertools import combinations
from multiprocessing import Pool
from typing import TYPE_CHECKING
from numba import njit, prange

import numpy as np
import pandas as pd
import structlog
from atriegc import TrieNucl as Trie
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm
from textdistance import hamming

from hilary.utils import apply_chunked_parallel, apply_parallel, p_required

if TYPE_CHECKING:
    from gitlab.HILARy.hilary.apriori import Apriori

log = structlog.get_logger()

NUM_RELIABLE_SEQ=100
SUFFICIENT_MUT_NUM=3


@njit
def hamming_njit(s1, s2):
    count = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            count += 1
    return count


@njit
def metric(cdr31, s1, n1, cdr32, s2, n2, l_L, l_L_L, L):
    if not n1 * n2:
        return L

    n = hamming_njit(cdr31, cdr32)
    nl = hamming_njit(s1, s2)
    n0 = (n1 + n2 - nl) / 2

    exp_n = l_L * (nl + 1)
    std_n = np.sqrt(exp_n * l_L_L)

    exp_n0 = n1 * n2 / L
    std_n0 = np.sqrt(exp_n0)

    x = (n - exp_n) / std_n
    y = (n0 - exp_n0) / std_n0
    return x - y


@njit(parallel=True)
def compute_chunk(cdr3s, vjs, muts, start, end, n, l_L, l_L_L, L):
    dist = np.empty(end - start)
    for idx in prange(end - start):
        k = start + idx

        i = int(n - 2 - np.floor(np.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5))
        j = int(k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2)

        d = metric(
            cdr3s[i], vjs[i], muts[i],
            cdr3s[j], vjs[j], muts[j],
            l_L, l_L_L, L
        )
        dist[idx] = d
    return dist

def group_mutations(args:tuple[int,pd.DataFrame])->pd.DataFrame:
    """Get list of mutations for a given VJL class.

    Args:
        args (tuple[int,pd.DataFrame]): (class_id, dataframe for that class).

    Returns
    -------
        pd.DataFrame: CLass dataframe with mutation count list.
    """
    _, df = args
    v_gene, j_gene, cdr3_length, _, class_id, alignment_length = df.iloc[0]
    mutations = df["mutation_count"].values
    return pd.DataFrame(
        [
            v_gene,
            j_gene,
            cdr3_length,
            mutations,
            alignment_length,
            class_id,
        ]
    ).T

class CDR3Clustering:
    """
    A class to infer families using CDR3 length and thresholds computed by the Apriori class.

    Attributes
    ----------
    thresholds : pd.DataFrame
        Dataframe containing thresholds for each (V, J, l) class.
    threads : int
        Number of CPUs on which to run the code, defaults to 1.

    Methods
    -------
    cluster(args: tuple[tuple[str, str, int], pd.DataFrame]) -> pd.Series
        Returns cluster labels depending on thresholds in self.thresholds.
    infer(df: pd.DataFrame, group: list[str] | None = None, silent: bool = False) -> pd.Series
        Returns cluster labels depending on thresholds in self.thresholds.
        Runs self.cluster in parallel on dataframe grouped by 'group' argument.
    """

    def __init__(self, thresholds: pd.DataFrame, threads: int = 1) -> None:
        """Initialize thresholds.

        Args:
            thresholds pd.DataFrame: Dataframe containing thresholds for each (V,J,l) class.
            threads (int, optional): Number of cpus on which to run code, defaults to 1.
        """
        self.thresholds = thresholds
        self.threads = threads

    def cluster(self, args: tuple[tuple[str, str, int], pd.DataFrame]) -> pd.Series:
        """Return cluster labels depending of thresholds in self.thresholds.

        Args:
            args (Tuple[Tuple[str, str, int], pd.DataFrame]): (Vgene,Jgene,l), Dataframe
            of sequences grouped by V,J,l class.

        Returns
        -------
            pd.Series: Cluster labels for this V,J,l class.
        """
        (v, j, cdr3_l), df = args
        trie = Trie()
        for cdr3 in df["cdr3"]:
            trie.insert(cdr3)
        t = self.thresholds.loc[
            (self.thresholds.v_gene == v)
            & (self.thresholds.j_gene == j)
            & (self.thresholds.cdr3_length == cdr3_l)
        ].values[0][-1]
        if t >= 0:
            dct = trie.clusters(t)
            return df["cdr3"].map(dct)
        return pd.Series(df.index, index=df.index)

    def infer(
        self,
        df: pd.DataFrame,
        group: list[str] | None = None,
        *,
        silent: bool = False,
    ) -> pd.Series:
        """Return cluster labels depending of thresholds in self.thresholds.

        Runs self.cluster parallely on dataframe grouped by 'group' argument.

        Args:
            df (pd.DataFrame): Dataframe of sequences.
            group (list[str], optional): Groups on which to do parallel inferring of clusters.
            Defaults to ["v_gene", "j_gene", "cdr3_length"].
            silent (bool,optional) : Do not show progress bar if True.

        Returns
        -------
            pd.Series: Series with cluster labels.
        """
        if group is None:
            group = ["v_gene", "j_gene", "cdr3_length"]
        use = [*group, "cdr3"]
        log.debug("Inferring clusters.", group=group)

        df["cluster"] = apply_chunked_parallel(
            df[use].groupby(group),
            self.cluster,
            silent=silent,
            cpu_count=self.threads,
        )
        group = [*group, "cluster"]
        return df.groupby(group).ngroup() + 1


class DistanceMatrix:
    def __init__(self, cdr3_l, alignment_length, df: pd.DataFrame, threads: int = 1):
        self.threads = threads
        self.l = cdr3_l
        self.L = alignment_length
        self.l_L = cdr3_l / alignment_length
        self.l_L_L = (cdr3_l + alignment_length) / alignment_length

        # Max byte length — adjust based on real data if needed
        max_cdr3_len = df["cdr3"].str.len().max()
        max_vj_len = df["alt_sequence_alignment"].str.len().max()

        self.cdr3s = df["cdr3"].astype(f"S{max_cdr3_len}").values
        self.vjs = df["alt_sequence_alignment"].astype(f"S{max_vj_len}").values
        self.muts = df["mutation_count"].astype(np.int32).values

        self.n = len(self.cdr3s)
        self.k_max = self.n * (self.n - 1) // 2
        self.k_step = max(self.n**2 // 2 // 500, 3)

    def proc(self, start):
        k1 = start
        k2 = min(start + self.k_step, self.k_max)
        chunk = compute_chunk(
            self.cdr3s, self.vjs, self.muts,
            k1, k2, self.n,
            self.l_L, self.l_L_L, self.L
        )
        return k1, k2, chunk

    def compute(self):
        dist = np.zeros(self.k_max)
        with Pool(self.threads) as pool:
            for k1, k2, chunk in pool.imap_unordered(self.proc, range(0, self.k_max, self.k_step)):
                dist[k1:k2] = chunk
        return dist + self.L

class HILARy:
    """Infer families using CDR3 and mutation information.

    Methods
    -------
    __init__(apriori: Apriori, df, crude: bool = False) -> None
        Initialize Hilary attributes using Apriori object.
    simulate_xs_ys(args) -> tuple
        Simulate xs and ys values for given arguments.
    get_xy_thresholds(df: pd.DataFrame) -> None
        Compute xy_thresholds for each (v_gene, j_gene, cdr3_length) class.
    single_linkage(indices: np.ndarray, dist: np.ndarray, threshold: float) -> dict[int, int]
        Map precise clusters to new precise AND sensitive clusters by merging clusters together.
    class2pairs(args: tuple[tuple[str, str, int, int], pd.DataFrame]) -> pd.Series
        Group precise clusters together.
    compute_prec_sens_clusters(df: pd.DataFrame) -> pd.DataFrame
        Infer precise and sensitive clusters.
    compute_crude_method_clusters(df: pd.DataFrame, normalized_threshold: float = 0.2,
        fixed_threshold: int = -1) -> pd.DataFrame
        Infer precise and sensitive clusters using crude method.
    mark_class(df: pd.DataFrame) -> pd.Series
        Flag all indices of a sensitive cluster not reaching desired sensitivity.
    to_do(df: pd.DataFrame, size_threshold: int = 500)
        -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Classify sensitive clusters not reaching desired sensitivity into big or small cluster.
    infer(df: pd.DataFrame) -> pd.DataFrame
        Infer family clusters.
    """

    def __init__(self, apriori: Apriori, df, *, crude: bool = False) -> None:
        """Initialize Hilary attributes using Apriori object.

        Args:
            apriori (Apriori): Apriori object containing histograms and thresholds.
            xy_threshold (int): Threshold to use for the xy method.
        """
        self.group = ["v_gene", "j_gene", "cdr3_length"]
        self.classes = apriori.classes
        self.use = [
            "cdr3",
            "alt_sequence_alignment",
            "mutation_count",
            "index",
        ]
        self.alignment_length = len(df["alt_sequence_alignment"].values[0])
        if not crude:
            self.remaining = (
                self.classes.query(
                    "v_gene != 'None' and precise_threshold<sensitive_threshold and pair_count > 0",
                )
                .groupby(self.group)
                .first()
                .index
            )
        self.silent = apriori.silent
        self.threads = apriori.threads
        self.paired = apriori.paired

    def simulate_xs_ys(
        self,
        args,
    ):
        """
        Simulate xs and ys values based on the given arguments.

        Args:
            args (tuple): A tuple containing the following elements:
                - _: Unused argument.
                - _: Unused argument.
                - l (int): Length parameter.
                - prevalence (float): Prevalence parameter.
                - mutations (list): List of mutation counts.
                - alignment_length (int): Length of the alignment.
                - class_id (int): Identifier for the class.

        Returns
        -------
            tuple: A tuple containing:
                - float: The sorted zs value at the required prevalence percentile.
                - int: The class identifier.
        """
        rng = np.random.default_rng(seed=42)
        size = int(1e6)
        (_, _, _, mutations, alignment_length, class_id) = args
        classes_temp = self.classes.loc[self.classes.class_id == class_id]
        cdr3_length = classes_temp.cdr3_length_value.values[0]
        if (len(mutations) < NUM_RELIABLE_SEQ):
            return (0, class_id)
        bins = np.arange(np.max(mutations) + 1)
        pni, nis = np.histogram(mutations, bins=bins)
        p = pni[1:] / sum(pni[1:])
        if len(nis) < SUFFICIENT_MUT_NUM:
            return (0, class_id)
        n1s = rng.choice(nis[1:-1], size=size, replace=True, p=p)
        n2s = rng.choice(nis[1:-1], size=size, replace=True, p=p)
        exp_n0 = n1s * n2s / alignment_length
        n0s = rng.poisson(lam=exp_n0, size=size)
        std_n0 = np.sqrt(exp_n0)
        ys = (n0s - exp_n0) / std_n0
        ns = rng.choice(np.arange(cdr3_length + 1), size=size, replace=True)
        nls = np.maximum(n1s + n2s - 2 * n0s, 0)
        exp_n = (cdr3_length / alignment_length) * (nls + 1)
        std_n = np.sqrt(exp_n * (cdr3_length + alignment_length) / alignment_length)
        xs = (ns - exp_n) / std_n
        zs = xs - ys
        return (
            np.sort(zs)[min(int(size * p_required(0.2)), size - 1)],
            class_id,
        )

    def get_xy_thresholds(self, df: pd.DataFrame) -> None:
        """Compute xy_thresholds for each (v_gene,j_gene,cdr3_length) class.

        Args:
            df(pd.DataFrame):Dataframe of sequences.

        Returns
        -------
            None
        """
        alignment_length = len(df["alt_sequence_alignment"].values[0])
        log.debug(
            "Group mutations by (v_gene,j_gene,cdr3_length) and compute xy_thresholds.",
        )
        self.classes["alignment_length"] = alignment_length
        merged = df[["v_gene", "j_gene", "cdr3_length", "mutation_count"]].merge(
            self.classes[
                [
                    "v_gene",
                    "j_gene",
                    "cdr3_length",
                    "class_id",
                    "alignment_length",
                ]
            ]
        )
        mutations_grouped = apply_chunked_parallel(
            merged.groupby("class_id"),
            group_mutations,
            cpu_count=self.threads,
            silent=self.silent,
        )

        log.debug(
            "Compute xy_thresholds for each (v_gene,j_gene,cdr3_length) class.",
        )
        result = apply_parallel(
            mutations_grouped.values,
            self.simulate_xs_ys,
            cpu_count=self.threads,
            silent=self.silent,
            isint=True,
        )
        thresholds_data = pd.DataFrame(result, columns=["xy_threshold", "class_id"]).set_index(
            "class_id"
        )
        self.classes["xy_threshold"] = thresholds_data["xy_threshold"]

    def single_linkage(
        self,
        indices: np.ndarray,
        dist: np.ndarray,
        threshold: float,
    ) -> dict[int, int]:
        """Map precise clusters to new precise AND sensitive clusters by merging clusters together.

        Args:
            indices (np.array): Indices of precise clusters.
            dist (np.ndarray): Distances between precise clusters.
            threshold (float): Threshold to merge two precise clusters if the distance is smaller.

        Returns
        -------
            dict: Dictionary mapping precise clusters to their new clusters.
        """
        clusters = fcluster(
            linkage(dist, method="single"),
            criterion="distance",
            t=threshold,
        )
        return dict(zip(indices, clusters))

    def class2pairs(
        self,
        args: tuple[tuple[str, str, str], pd.DataFrame],
    ) -> pd.Series:
        """Group precise clusters together.

        Args:
            args (Tuple[Tuple[str,str,int],pd.DataFrame]): (Vgene,Jgene,cdr3length),dataframe
            representing sensitive cluster.

        Returns
        -------
            pd.Series: New clusters made of grouped precise clusters.
        """
        df = args[1]  # (vgene, jgene, cdr3length, sensitive cluster), df
        v_gene, j_gene, cdr3_length = args[0]

        if self.paired:
            cdr3_length_value = int(cdr3_length.split(",")[0]) + int(cdr3_length.split(",")[1])
        else:
            cdr3_length_value = int(cdr3_length)

        xy_threshold = self.classes.query(
            "v_gene==@v_gene and j_gene==@j_gene and cdr3_length==@cdr3_length"
        )["xy_threshold"].values[0]
        indices = np.arange(len(df))
        df["index"] = indices
        if len(indices) <= 1:
            return df["index"]
        dim = len(indices)
        distance_matrix = np.ones((dim, dim), dtype=float) * (2 * self.alignment_length)
        for i in range(dim):
            distance_matrix[i, i] = 0
        for (cdr31, s1, n1, i1), (cdr32, s2, n2, i2) in combinations(df[self.use].values, 2):
            if i1 != i2:
                n1n2 = n1 * n2
                if n1n2 > 0:
                    n = hamming(cdr31, cdr32)
                    nl = hamming(s1, s2)
                    n0 = (n1 + n2 - nl) / 2

                    exp_n = cdr3_length_value / self.alignment_length * (nl + 1)
                    std_n = np.sqrt(
                        exp_n * (cdr3_length_value + self.alignment_length) / self.alignment_length
                    )
                    exp_n0 = n1n2 / self.alignment_length
                    std_n0 = np.sqrt(exp_n0)
                    x = (n - exp_n) / std_n
                    y = (n0 - exp_n0) / std_n0
                    distance = x - y + self.alignment_length
                    distance_matrix[i1, i2] = min(distance, distance_matrix[i1, i2])
                    distance_matrix[i2, i1] = min(distance, distance_matrix[i2, i1])
        sl = self.single_linkage(
            indices,
            squareform(distance_matrix),
            threshold=self.alignment_length + xy_threshold,
        )
        print("done2")
        return df["index"].map(sl)

    def compute_prec_sens_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer precise and sensitive clusters.

        Args:
            df(pd.DataFrame):Dataframe of sequences.

        Returns
        -------
            pd.DataFrame with sensitive and precise clusters.
        """
        prec = CDR3Clustering(
            self.classes[[*self.group, "precise_threshold"]], threads=self.threads
        )
        sens = CDR3Clustering(
            self.classes[[*self.group, "sensitive_threshold"]],
            threads=self.threads,
        )
        df["precise_cluster"] = prec.infer(df, silent=self.silent)
        df["sensitive_cluster"] = sens.infer(df, silent=self.silent)
        return df

    def compute_crude_method_clusters(
        self,
        df: pd.DataFrame,
        normalized_threshold: float = 0.2,
        fixed_threshold: int = -1,
    ) -> pd.DataFrame:
        """Infer precise and sensitive clusters.

        Args:
            df(pd.DataFrame):Dataframe of sequences.

        Returns
        -------
            pd.DataFrame with sensitive and precise clusters.
        """
        if fixed_threshold >= 0:
            log.info("Using crude method with a fixed threshold.", threshold=fixed_threshold)
            self.classes["threshold"] = fixed_threshold
        else:
            log.info(
                "Using crude method with a normalized threshold.", threshold=normalized_threshold
            )
            self.classes["threshold"] = (
                self.classes["cdr3_length_value"] * normalized_threshold
            ).astype(int)
        prec = CDR3Clustering(self.classes[[*self.group, "threshold"]], threads=self.threads)
        df["crude_method_family"] = prec.infer(df, silent=self.silent)
        return df

    def infer(self, df: pd.DataFrame, size_threshold: int = 1000) -> pd.DataFrame:
        """Infer family clusters.

        First, for each sensitive cluster that does not reach desired sensitivity, group precise
        clusters together with a single linkage algorithm. This grouping is done differently
        depending on whether the sensitive cluster is large or not to use parallelization in the
        most efficient way possible.

        Args:
            df(pd.DataFrame):Dataframe of sequences.

        Returns
        -------
            df(pd.DataFrame): Dataframe with inferred clonal families in 'clone_id'.
        """
        df_grouped = df.groupby(
            [*self.group],
        )
        sizes = df_grouped.size()
        mask = sizes > size_threshold
        large_to_do = sizes[mask].index
        small_to_do = sizes[~mask].index
        self.alignment_length = len(df["alt_sequence_alignment"].values[0])
        log.debug("Checking alignment length.", alignment_length=self.alignment_length)
        log.debug("Inferring family clusters for small groups.")
        small_to_do_df = pd.DataFrame(list(small_to_do), columns=small_to_do.names)
        df["index"] = df.index.values
        small_df = small_to_do_df.merge(df)
        small_df.index = small_df["index"].values
        small_df = small_df.drop(columns=["index"])

        df["family_cluster"] = apply_chunked_parallel(
            small_df.groupby([*self.group]),
            self.class2pairs,
            silent=self.silent,
            cpu_count=self.threads,
        )
        print("done")
        log.debug("Inferring family clusters for large groups.")
        large_to_do_df = pd.DataFrame(list(large_to_do), columns=large_to_do.names)
        df["index"] = df.index.values
        large_df = large_to_do_df.merge(df)
        large_df.index = large_df["index"].values
        large_df = large_df.drop(columns=["index"])
        if self.paired:
            large_df["cdr3_length_value"] = large_df.cdr3_length.apply(
                lambda x: int(x.split(",")[0]) + int(x.split(",")[1])
            )
        else:
            large_df["cdr3_length_value"] = large_df.cdr3_length.astype(int)
        grouped_list = list(
            large_df.groupby(
                ["v_gene", "j_gene", "cdr3_length", "cdr3_length_value"]
            )
        )
        large_dict = {}
        for g, grouped_df in tqdm(grouped_list):
            v_gene, j_gene, cdr3_length, cdr3_length_value = g
            xy_threshold = self.classes.query(
                "v_gene==@v_gene and j_gene==@j_gene and cdr3_length==@cdr3_length"
            )["xy_threshold"].values[0]
            dm = DistanceMatrix(
                cdr3_l=cdr3_length_value,
                alignment_length=self.alignment_length,
                df=grouped_df[
                    ["cdr3", "alt_sequence_alignment", "mutation_count"]
                ],
                threads=self.threads,
            )
            d = dm.compute()
            dct = self.single_linkage(
                indices=grouped_df.index,
                dist=d,
                threshold=self.alignment_length + xy_threshold,
            )
            large_dict.update(dct)

        df["new_index"] = df.index
        df["family_cluster"] = df["family_cluster"].fillna(df["new_index"].map(large_dict))
        df.fillna(value={"family_cluster": 0}, inplace=True)
        df["clone_id"] = (
            df.groupby(
                [*self.group, "family_cluster"],
            ).ngroup()
            + 1
        )
        return df.drop(
            columns=[
                "family_cluster",
            ],
        )
