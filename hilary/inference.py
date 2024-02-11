from __future__ import annotations

from itertools import combinations
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import structlog
from atriegc import TrieNucl as Trie
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from textdistance import hamming

from hilary.apriori import Apriori
from hilary.utils import applyParallel

# from atriegc import Trie

log = structlog.get_logger()


class CDR3Clustering:
    """Infer families using CDR3 length and thresholds computed by Apriori class."""

    def __init__(self, thresholds: pd.DataFrame) -> None:
        """Initialize thresholds.

        Args:
            thresholds pd.DataFrame: Dataframe containing thresholds for each (V,J,l) class.
        """
        self.thresholds = thresholds

    def cluster(self, args: tuple[tuple[str, str, int], pd.DataFrame]) -> pd.Series:
        """Returns cluster labels depending of thresholds in self.thresholds

        Args:
            args (Tuple[Tuple[str, str, int], pd.DataFrame]): (Vgene,Jgene,l), Dataframe
            of sequences grouped by V,J,l class.

        Returns:
            pd.Series: Cluster labels for this V,J,l class.
        """
        (v, j, l), df = args
        trie = Trie()
        for cdr3 in df["cdr3"]:
            trie.insert(cdr3)
        t = self.thresholds.loc[
            (self.thresholds.v_gene == v)
            & (self.thresholds.j_gene == j)
            & (self.thresholds.cdr3_length == l)
        ].values[0][-1]
        if t >= 0:  # ? question t>0 better ?
            dct = trie.clusters(t)
            return df["cdr3"].map(dct)
        return df.index.to_frame()

    def infer(
        self,
        df: pd.DataFrame,
        group: list[str] = ["v_gene", "j_gene", "cdr3_length"],
        silent: bool = False,
    ) -> pd.Series:
        """Returns cluster labels depending of thresholds in self.thresholds.
        Runs self.cluster parallely on dataframe grouped by 'group' argument.

        Args:
            df (pd.DataFrame): Dataframe of sequences.
            group (list[str], optional): Groups on which to do parallel inferring of clusters.
            Defaults to ["v_gene", "j_gene", "cdr3_length"].
            silent (bool,optional) : Do not show progress bar if True.

        Returns:
            pd.Series: Series with cluster labels.
        """
        use = group + ["cdr3"]
        df["cluster"] = applyParallel(
            df[use].groupby(group),
            self.cluster,
            silent=silent,
        )
        group = group + ["cluster"]
        return df.groupby(group).ngroup() + 1


class DistanceMatrix:
    """Object enabling parallel computing of the distance martrix of a large cluster."""

    def __init__(
        self,
        l: int,
        alignment_length: int,
        df: pd.DataFrame,
        threads: int = cpu_count() - 1,
        xy_threshold: int = 0,
    ) -> None:
        """Initialize attributes.

        Args:
            l (int): CDR3 length
            df (pd.DataFrame): Dataframe of sequences grouped by (v,j,l,sensitive cluster)
            threads (int, optional): Number of cpus on which to run code. Defaults to cpu_count()-1.
        """
        self.threads = threads
        self.l = l
        self.L = alignment_length
        self.l_L = l / self.L
        self.l_L_L = l / (l + self.L)

        self.data = df.values
        self.n = self.data.shape[0]
        # maximum elements in 1D dist array
        self.k_max = self.n * (self.n - 1) // 2
        self.k_step = max(self.n**2 // 2 // (500), 3)  # ~500 bulks

    def metric(
        self,
        arg1: tuple[int, str, int, int],
        arg2: tuple[int, str, int, int],
    ) -> float:
        """Compute difference btween normalized cdr3 divergence & shared mutations of two sequences.

        Args:
            arg1 (Tuple[int,str,int,int]): (CDR3 length, V+J sequence alignment, number of mutations
            from germline, index) for sequence 1
            arg2 (Tuple[int,str,int,int]): Same for sequence 2

        Returns:
            float: Difference between two quantities.
        """
        cdr31, s1, n1, i1 = arg1
        cdr32, s2, n2, i2 = arg2
        if i1 == i2:
            return -self.L
        if n1 * n2 == 0:
            return self.L
        n = hamming(cdr31, cdr32)
        nL = hamming(s1, s2)
        n0 = (n1 + n2 - nL) / 2

        exp_n = self.l_L * (nL + 1)
        std_n = np.sqrt(exp_n * self.l_L_L)

        exp_n0 = n1 * n2 / self.L
        std_n0 = np.sqrt(exp_n0)

        x = (n - exp_n) / std_n
        y = (n0 - exp_n0) / std_n0
        return x - y

    def proc(self, start: int) -> tuple[int, int, list[float]]:
        """Compute 1D distance matrix between start and start+self.k_step

        Args:
            start (int): Index from which to compute distances.

        Returns:
            Tuple[int, int, list[float]]: Start index, End index, distance for indices inbetween
        """
        dist = []
        k1 = start
        k2 = min(start + self.k_step, self.k_max)
        for k in range(k1, k2):
            # get (i, j) for 2D distance matrix knowing (k) for 1D distance matrix
            i = int(
                self.n
                - 2
                - int(np.sqrt(-8 * k + 4 * self.n * (self.n - 1) - 7) / 2.0 - 0.5),
            )
            j = int(
                k
                + i
                + 1
                - self.n * (self.n - 1) / 2
                + (self.n - i) * ((self.n - i) - 1) / 2,
            )
            # store distance
            a = self.data[i, :]
            b = self.data[j, :]
            d = self.metric(a, b)
            dist.append(d)
        return k1, k2, dist

    def compute(self) -> np.ndarray:
        """Run self.proc parallely to compute 1D distance matrix.

        Returns:
            np.array: 1D distance matrix
        """
        dist = np.zeros(self.k_max)
        with Pool(self.threads) as pool:
            for k1, k2, res in pool.imap_unordered(
                self.proc,
                range(0, self.k_max, self.k_step),
            ):
                dist[k1:k2] = res
        return dist + self.L


class HILARy:
    """Infer families using CDR3 and mutation information"""

    def __init__(self, apriori: Apriori, xy_threshold: int = 0):
        """Initialize Hilary attributes using Apriori object.

        Args:
            apriori (Apriori): Apriori object containing histograms and thresholds.
        """

        self.group = ["v_gene", "j_gene", "cdr3_length"]
        self.classes = apriori.classes
        self.use = [
            "cdr3",
            "alt_sequence_alignment",
            "mutation_count",
            "index",
        ]
        self.alignment_length = None
        self.xy_threshold = xy_threshold

        self.remaining = (
            self.classes.query(
                "v_gene != 'None' and precise_threshold < sensitive_threshold and pair_count > 0",
            )
            .groupby(self.group)
            .first()
            .index
        )
        self.silent = apriori.silent
        self.cdfs = apriori.cdfs
        self.lengths = apriori.lengths

    def singleLinkage(
        self,
        indices: np.ndarray,
        dist: np.ndarray,
        threshold: float,
    ) -> dict[int, int]:
        """Maps precise clusters to new precise AND sensitive clusters by merging clusters together.


        Args:
            indices (np.array): Indices of precise clusters.
            dist (np.ndarray): Distances between precise clusters.
            threshold (float): Threshold to merge two precise clusters if the distance is smaller.

        Returns:
            dict: _description_
        """
        clusters = fcluster(
            linkage(dist, method="single"),
            criterion="distance",
            t=threshold,
        )
        return {i: c for i, c in zip(indices, clusters)}

    def class2pairs(
        self,
        args: tuple[tuple[str, str, int, int], pd.DataFrame],
    ) -> pd.Series:
        """Group precise clusters together.

        Args:
            args (Tuple[Tuple[str,str,int,int],pd.DataFrame]): (Vgene,Jgene,cdr3length),dataframe
            representing sensitive cluster.

        Returns:
            pd.Series: New clusters made of grouped precise clusters.
        """
        df = args[1]  # (vgene, jgene, cdr3length, sensitive cluster), df
        l = args[0][2]
        indices = np.unique(df["precise_cluster"])
        if len(indices) <= 1:
            return df["precise_cluster"]

        translateIndices = dict(zip(indices, range(len(indices))))
        df["index"] = df["precise_cluster"].map(translateIndices)
        dim = len(indices)
        distanceMatrix = np.ones((dim, dim), dtype=float) * (2 * self.alignment_length)
        for i in range(dim):
            distanceMatrix[i, i] = 0

        for (cdr31, s1, n1, i1), (cdr32, s2, n2, i2) in combinations(
            df[self.use].values,
            2,
        ):
            if i1 == i2 or n1 * n2 == 0:
                continue
            n = hamming(cdr31, cdr32)
            nL = hamming(s1, s2)
            n0 = (n1 + n2 - nL) / 2

            exp_n = l / self.alignment_length * (nL + 1)
            std_n = np.sqrt(
                exp_n * (l + self.alignment_length) / self.alignment_length,
            )

            exp_n0 = n1 * n2 / self.alignment_length
            std_n0 = np.sqrt(exp_n0)

            x = (n - exp_n) / std_n
            y = (n0 - exp_n0) / std_n0
            distance = x - y + self.alignment_length
            distanceMatrix[i1, i2] = distance
            distanceMatrix[i2, i1] = distance

        sl = self.singleLinkage(
            indices,
            squareform(distanceMatrix),
            threshold=self.alignment_length + self.xy_threshold,
        )
        return df["precise_cluster"].map(sl)

    def compute_prec_sens_clusters(self, df) -> None:
        """Infer precise and sensitive clusters."""
        prec = CDR3Clustering(self.classes[self.group + ["precise_threshold"]])
        sens = CDR3Clustering(
            self.classes[self.group + ["sensitive_threshold"]],
        )
        df["precise_cluster"] = prec.infer(df, silent=self.silent)
        df["sensitive_cluster"] = sens.infer(df, silent=self.silent)
        return df

    def mark_class(self, df: pd.DataFrame) -> pd.Series:
        """Flag all indices of a sensitive cluster not reaching desired sensitivity.

        Args:
            df (pd.DataFrame): Dataframe grouped representing given cluster.

        Returns:
            pd.Series: Series flagging all indices of given cluster.
        """
        df["to_resolve"] = True
        return df["to_resolve"]

    def to_do(
        self,
        df,
        size_threshold: float = 5,
        xy_complete: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Classify sensitive clusters not reaching desired sensitivity into big or small cluster.

        Args:
            size_threshold (float, optional): Threshold to separate big and small clusters.
            Defaults to 1e3.

        Returns:
            Tuple[pd.DataFrame,pd.DataFrame]: Returns indices of small and big sensitive clusters.
        """
        df["to_resolve"] = False
        if not self.remaining.empty:
            df["to_resolve"] = applyParallel(
                [df.groupby(self.group).get_group(g) for g in self.remaining],
                self.mark_class,
                silent=True,
            )
            df.fillna(value={"to_resolve": False}, inplace=True)
        if self.remaining.empty and not xy_complete:
            log.info("No classes passed to xy method.")

        if xy_complete:
            df["to_resolve"] = True
            dfGrouped = df.query("to_resolve == True").groupby(
                self.group,
            )
        if not xy_complete:
            dfGrouped = df.query("to_resolve == True").groupby(
                self.group + ["sensitive_cluster"],
            )
        sizes = dfGrouped.size()
        mask = sizes > size_threshold
        large_to_do = sizes[mask].index
        small_to_do = sizes[~mask].index
        return df, small_to_do, large_to_do

    def infer(self, df, xy_complete: bool = False) -> None:
        """Infer family clusters.
        First, for each sensitive cluster that does not reach desired sensitivity, group precise
        clusters together with a single linkage algorithm. This grouping is done differently
        depending on whether the sensitive cluster is large or not.
        """
        df, small_to_do, large_to_do = self.to_do(df, xy_complete=xy_complete)
        self.alignment_length = len(df["alt_sequence_alignment"].values[0])

        log.info("Alignment length", alignment_length=self.alignment_length)

        if sum(df["to_resolve"]) == 0:
            log.info("Returning cdr3 method precise clusters.")
            df["family"] = df["precise_cluster"]
            df = df.drop(
                columns=[
                    "cluster",
                ],
            )
            return
        if xy_complete:
            log.info("Running xy method on all VJL classes.")
            dfGrouped = df.groupby(self.group)
        else:
            dfGrouped = df.groupby(self.group + ["sensitive_cluster"])
        log.debug(
            "Grouping precise clusters together to reach desired sensitivity.",
        )
        log.debug("Inferring family clusters for small groups.")
        df["family_cluster"] = applyParallel(
            [(g, dfGrouped.get_group(g)) for g in small_to_do],
            self.class2pairs,
            silent=self.silent,
        )
        log.debug("Inferring family clusters for large groups.")
        for g in large_to_do:
            v_gene, j_gene, l, sensitive_cluster = g
            xy_threshold_2 = self.classes.query(
                "v_gene==@v_gene and j_gene==@j_gene and cdr3_length==@l"
            )["xy_threshold"].values[0]
            if xy_threshold_2 == "None":
                xy_threshold_2 = 0
            dm = DistanceMatrix(
                l=l,
                alignment_length=self.alignment_length,
                df=dfGrouped.get_group(g)[
                    [
                        "cdr3",
                        "alt_sequence_alignment",
                        "mutation_count",
                        "precise_cluster",
                    ]
                ],
            )

            d = dm.compute()
            print(dm)
            dct = self.singleLinkage(
                dfGrouped.get_group(g).index,
                d,
                self.alignment_length - xy_threshold_2 + 150,
            )
            df["family_cluster"] = df.index.map(dct)

        df.fillna(value={"family_cluster": 0}, inplace=True)
        if xy_complete:
            df["family"] = (
                df.groupby(
                    self.group + ["family_cluster"],
                ).ngroup()
                + 1
            )
        else:
            df["family"] = (
                df.groupby(
                    self.group + ["sensitive_cluster", "family_cluster"],
                ).ngroup()
                + 1
            )
        df = df.drop(
            columns=[
                "family_cluster",
            ],
        )
        return df
