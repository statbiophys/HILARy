"""Infer clonal families."""

from __future__ import annotations

import pandas as pd
import structlog
from atriegc import TrieNucl as Trie

from hilary.utils import apply_chunked_parallel

log = structlog.get_logger()

NUM_RELIABLE_SEQ = 100
SUFFICIENT_MUT_NUM = 3


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
        log.info("⏳ COMPUTING CDR3 CLUSTERS ⏳.")
        if group is None:
            group = ["v_gene", "j_gene", "cdr3_length"]
        use = [*group, "cdr3"]
        df["cluster"] = apply_chunked_parallel(
            df[use].groupby(group),
            self.cluster,
            silent=silent,
            cpu_count=self.threads,
        )
        group = [*group, "cluster"]
        return df.groupby(group).ngroup() + 1
