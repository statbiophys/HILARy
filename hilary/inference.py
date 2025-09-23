"""Optimized clonal family inference implementation."""

from __future__ import annotations

from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import structlog
from fastcluster import linkage as fast_linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from tqdm import tqdm

from hilary.distance_matrix import DistanceMatrix
from hilary.utils import (
    apply_chunked_parallel,
    apply_parallel,
    create_classes,
    group_mutations,
    p_required,
)

pd.options.mode.chained_assignment = None  # default='warn'
log = structlog.get_logger()

NUM_RELIABLE_SEQ = 100
SUFFICIENT_MUT_NUM = 3


class HILARy:
    """Infer families using CDR3 and mutation information."""

    def __init__(
        self,
        df: pd.DataFrame,
        threads: int = 1,
        *,
        silent: bool = False,
        paired: bool = False,
        chunk_size: int = 10000,
    ) -> None:
        self.group = ["v_gene", "j_gene", "cdr3_length", "cdr3_length_value", "split_up_cluster"]
        self.use = ["cdr3", "alt_sequence_alignment", "mutation_count", "index"]
        self.threads = threads if threads > 0 else cpu_count()
        self.silent = silent
        self.paired = paired
        self.classes = create_classes(df)
        self.chunk_size = chunk_size

        self.classes.index = self.classes.class_id

    def simulate_xs_ys(self, args:tuple) -> tuple[float, int]:
        """Get thresholds for single linkage clustering.

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
        size = int(1e5)
        (_, _, _, mutations, alignment_length, class_id) = args

        classes_temp = self.classes.loc[self.classes.class_id == class_id]
        cdr3_length = classes_temp.cdr3_length_value.values[0]

        if len(mutations) < NUM_RELIABLE_SEQ:
            return (0, class_id)

        max_mut = np.max(mutations)
        bins = np.arange(max_mut + 1)
        pni, nis = np.histogram(mutations, bins=bins)

        if len(nis) < SUFFICIENT_MUT_NUM:
            return (0, class_id)

        p = pni[1:] / np.sum(pni[1:])
        valid_indices = np.arange(1, len(nis) - 1)

        n1s = rng.choice(valid_indices, size=size, replace=True, p=p)
        n2s = rng.choice(valid_indices, size=size, replace=True, p=p)

        exp_n0 = n1s * n2s / alignment_length
        n0s = rng.poisson(lam=exp_n0)
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
        log.info("⏳ COMPUTING XY THRESHOLDS ⏳.")
        alignment_length = len(df["alt_sequence_alignment"].values[0])
        self.classes["alignment_length"] = alignment_length

        merge_cols = ["v_gene", "j_gene", "cdr3_length", "mutation_count"]
        class_cols = ["v_gene", "j_gene", "cdr3_length", "class_id", "alignment_length"]

        merged = df[merge_cols].merge(self.classes[class_cols], how="left")

        mutations_grouped = apply_chunked_parallel(
            merged.groupby("class_id"),
            group_mutations,
            cpu_count=self.threads,
            silent=True,
        )

        # Parallel threshold computation
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
        self, indices: np.ndarray, dist: np.ndarray, threshold: float
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
        if len(indices) <= 1:
            return dict(zip(indices, indices))
        try:
            clusters = fcluster(
                fast_linkage(dist, method="single", preserve_input=False),
                criterion="distance",
                t=threshold,
            )
        except ValueError:
            clusters = fcluster(
                scipy_linkage(dist, method="single"),
                criterion="distance",
                t=threshold,
            )
        return dict(zip(indices, clusters))

    def class2pairs(self, args: tuple[tuple[str, str, str, int,int], pd.DataFrame]) -> pd.Series:
        """Infer clonal families for one small group.

        Args:
            args (Tuple[Tuple[str,str,int,int, int],pd.DataFrame]):
            (Vgene,Jgene,cdr3length string, cdr3_length int, sensitive cluster),dataframe.

        Returns
        -------
            pd.Series: Clonal family.
        """
        df = args[1]
        v_gene, j_gene, cdr3_length, cdr3_length_value, _ = args[0]
        xy_threshold = self.classes.query(
            "v_gene==@v_gene and j_gene==@j_gene and cdr3_length==@cdr3_length"
        )["xy_threshold"].values[0]

        indices = np.arange(len(df))
        df["index"] = indices
        if len(indices) <= 1:
            return df["index"]
        alignment_length=len(df["alt_sequence_alignment"].values[0])
        dm = DistanceMatrix(
            cdr3_l=cdr3_length_value,
            alignment_length=alignment_length,
            df=df[["cdr3", "alt_sequence_alignment", "mutation_count"]],
            threads=1,  # Use single thread for small groups
        )
        distances = dm.compute()

        sl = self.single_linkage(indices, distances, threshold=alignment_length + xy_threshold)
        return df["index"].map(sl)

    def infer(self, df: pd.DataFrame, size_threshold: int = 500) -> pd.DataFrame:
        """Infer family clusters.

        Clustering is done differently depending on whether the sensitive cluster is large or not
        to use parallelization in the most efficient way possible.

        Args:
            df(pd.DataFrame):Dataframe of sequences.

        Returns
        -------
            df(pd.DataFrame): Dataframe with inferred clonal families in 'clone_id'.
        """
        log.info("⏳ INFERRING FAMILIES WITH FULL XY METHOD ⏳.")
        df_grouped = df.groupby(self.group)
        sizes = df_grouped.size()

        mask = sizes > size_threshold
        large_groups = sizes[mask].index
        small_groups = sizes[~mask].index
        log.debug("Created small and large groups",
                number_small_groups=len(small_groups),
                number_large_groups=len(large_groups)
                )
        df["family_cluster"] = np.nan
        df["index"] = df.index.values

        if len(small_groups) > 0:
            log.debug("Processing small groups.", small_groups=len(small_groups))
            small_df = df[df[self.group].apply(tuple, axis=1).isin(small_groups)]

            family_clusters = apply_chunked_parallel(
                small_df.groupby(self.group),
                self.class2pairs,
                silent=self.silent,
                cpu_count=self.threads,
            )

            df.loc[small_df.index, "family_cluster"] = family_clusters

        if len(large_groups) > 0:
            log.debug("Processing large groups.", number=len(large_groups))
            large_df = df[df[self.group].apply(tuple, axis=1).isin(large_groups)]
            large_clusters = self._process_large_groups(large_df)
            for idx, cluster in large_clusters.items():
                df.loc[idx, "family_cluster"] = cluster

        df["family_cluster"] = df["family_cluster"].fillna(0)
        df["clone_id"] = df.groupby([*self.group, "family_cluster"]).ngroup() + 1

        return df.drop(columns=["family_cluster", "index"])

    def _process_large_groups(self, large_df: pd.DataFrame) -> dict[int, int]:
        """Infer large groups with optimized distance computation.

        Args:
            large_df (pd.DataFrame): Dataframe containinng large groups only.

        Returns
        -------
            dict[int, int]: Dictionary mapping id to cluster.
        """
        large_clusters = {}

        grouped_list = list(large_df.groupby(self.group))

        for g, grouped_df in tqdm(grouped_list, disable=self.silent):
            v_gene, j_gene, cdr3_length, cdr3_length_value, _ = g

            xy_threshold = self.classes.query(
                "v_gene==@v_gene and j_gene==@j_gene and cdr3_length==@cdr3_length"
            )["xy_threshold"].values[0]
            alignment_length = len(large_df["alt_sequence_alignment"].values[0])
            dm = DistanceMatrix(
                cdr3_l=cdr3_length_value,
                alignment_length=alignment_length,
                df=grouped_df[["cdr3", "alt_sequence_alignment", "mutation_count"]],
                threads=self.threads,
            )
            distances = dm.compute()

            dct = self.single_linkage(
                indices=grouped_df.index,
                dist=distances,
                threshold=alignment_length + xy_threshold,
            )

            large_clusters.update(dct)

        return large_clusters
