from __future__ import annotations

import os
from itertools import combinations
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from scipy.special import factorial
from textdistance import hamming

from hilary.expectmax import EM
from hilary.utils import applyParallel, create_classes, preprocess, pRequired

pd.set_option("mode.chained_assignment", None)

log = structlog.get_logger(__name__)


class Apriori:
    """Computes statistics of pairwise distances"""

    def __init__(
        self,
        lengths: np.ndarray = np.arange(15, 81 + 3, 3).astype(int),
        nmax: int = int(1e5),
        precision: float = 1.0,
        sensitivity: float = 1.0,
        threads: int = cpu_count() - 1,
        model: int = 326713,
        silent: bool = False,
        paired: bool = False,
    ):
        """Initialize attributes to later run class methods.

        Args:
            df (pd.DataFrame): Dataframe of sequences
            lengths (_type_, optional): CDR3 lengths used to filter non productive sequences. \
                Defaults to np.arange(15, 81 + 3, 3).
            nmax (int, optional): For parameter inference, sample and use nmax sequences for \
                classes larger than nmax. Defaults to 100000.
            precision (float, optional): Desired precision, defaults to 1.
            sensitivity (float, optional): Desired sensitivity, defaults to 1.
            threads (int, optional): Number of cpus on which to run code, defaults to cpu_count().
            model (int, optional): Model name to infer Null distribution, defaults to 326713.
            silent (bool) : If true do not to show progress bars.
        """
        self.lengths = lengths
        self.nmax = nmax
        self.threads = threads
        self.threads = threads
        self.precision = precision - 1e-4
        self.sensitivity = sensitivity
        self.silent = silent
        self.histograms = None
        self.mean_prevalence = None
        self.mean_mean_distance = None
        self.paired = paired
        if not paired:
            self.cdfs = pd.read_csv(
                Path(os.path.dirname(__file__)) / Path(f"cdfs_{model}.csv"),
            )
        else:
            self.cdfs = pd.read_csv(
                Path(os.path.dirname(__file__)) / Path("cdfs_paired.csv"),
            )
        self.classes = pd.DataFrame()

    def preprocess(self, df, df_kappa) -> pd.DataFrame:
        """Remove non productive sequences from dataframe.

        Returns:
            pd.Dataframe: Dataframe self.df containing all sequences.
        """
        df = preprocess(
            df,
            silent=self.silent,
        )
        if self.paired:
            df_kappa = preprocess(df_kappa, silent=self.silent)
            for column in df.columns:
                if column == "sequence_id":
                    continue
                df[column + "_h"] = df[column]
                df[column + "_k"] = df_kappa[column]
                df[column] = df[column + "_h"] + df[column + "_k"]
        return df

    def vjls2x(self, args: tuple[int, pd.DataFrame]) -> pd.DataFrame:
        """Compute histogram for a given VJl class

        Returns:
            pd.DataFrame: Histogram of distances for a given large VJl class."""
        i, df = args
        xs = []
        for s1, s2 in combinations(df["cdr3"].values, 2):
            xs.append(hamming(s1, s2))
        return pd.DataFrame(
            np.histogram(
                xs,
                bins=range(
                    self.lengths[-1] + 2,
                ),
                density=False,
            )[0],
            columns=[i],
        ).transpose()

    def compute_allvjl(
        self,
        df,
    ) -> pd.DataFrame:
        """Compute histograms for all large VJl classes.

        Returns:
            pd.DataFrame: Histogram of distances for large VJl classes."""
        query = "v_gene != 'None' and pair_count >0 and cdr3_length in @self.lengths"
        groups = df.groupby(["v_gene", "j_gene", "cdr3_length"])
        log.debug(
            "Computing CDR3 hamming distances within all large VJl classes.",
        )
        results = applyParallel(
            [
                (
                    row.class_id,
                    groups.get_group((row.v_gene, row.j_gene, row.cdr3_length)).sample(
                        frac=min(np.sqrt(self.nmax / row.pair_count), 1),
                    ),
                )
                for _, row in self.classes.query(query).iterrows()
            ],
            self.vjls2x,
            cpuCount=self.threads,
            silent=self.silent,
        )
        results["class_id"] = results.index
        return results

    def get_histograms(self, df) -> pd.DataFrame:
        """Compute histograms for all large classes

        Returns:
            pd.DataFrame: Histogram of distances for all large classes."""
        hs_vjl = self.compute_allvjl(df)
        self.histograms = hs_vjl.sort_values(
            "class_id",
        )[["class_id"] + [*range(self.lengths[-1] + 1)]]
        return self.histograms

    def estimate(self, args: tuple[tuple[int], pd.DataFrame]) -> pd.DataFrame:
        """Fit prevalence and mu using the histogram which is the distribution of distances.

        Args:
            args tuple[int, pd.DataFrame]: class_id, histogram for that class

        Returns:
            pd.DataFrame: dataframe with parameters for each class id.
        """
        class_id, h = args
        l = self.classes.loc[self.classes.class_id == class_id].cdr3_length.values[0]
        em = EM(l, h.values[0, 1:], positives="geometric", cdfs=self.cdfs)
        rho_geo, mu_geo = em.discreteEM()
        error_geo = em.error([rho_geo, mu_geo])
        em = EM(l, h.values[0, 1:], positives="poisson", cdfs=self.cdfs)
        rho_poisson, mu_poisson = em.discreteEM()
        error_poisson = em.error([rho_poisson, mu_poisson])
        result = pd.DataFrame(
            columns=[
                "class_id",
                "rho_geo",
                "mu_geo",
                "error_geo",
                "rho_poisson",
                "mu_poisson",
                "error_poisson",
            ],
        )
        result.class_id = [class_id[0]]
        result.rho_geo = [rho_geo]
        result.mu_geo = [mu_geo]
        result.error_geo = [error_geo]
        result.rho_poisson = [rho_poisson]
        result.mu_poisson = [mu_poisson]
        result.error_poisson = [error_poisson]
        return result

    def get_parameters(self) -> None:
        """Computes prevalence and mean distance for all classes."""
        if self.histograms is None:
            raise ValueError(
                "Histogram attribute is None. Please run get_histograms method.",
            )
        parameters = applyParallel(
            self.histograms.groupby(["class_id"]),
            self.estimate,
            cpuCount=self.threads,
            silent=self.silent,
        ).reset_index(drop=True)

        self.classes.index = self.classes.class_id
        parameters.index = parameters.class_id

        self.classes["prevalence"] = parameters[
            [
                "rho_poisson",
                "rho_geo",
            ]
        ].min(axis=1)
        self.classes["mean_distance"] = (
            parameters["mu_poisson"] / self.classes["cdr3_length"]
        )
        self.classes["effective_prevalence"] = self.classes["prevalence"].fillna(0.2)
        self.classes["effective_mean_distance"] = self.classes["mean_distance"].fillna(
            0.2,
        )

    def simulate_xs_ys(
        self,
        args,
    ):
        size = int(1e4)
        (v_gene, j_gene, l, prevalence, mutations, alignment_length) = args
        if l not in self.lengths:
            return
        if v_gene == "None":
            return 0
        bins = np.arange(np.max(mutations) + 1)
        pni, nis = np.histogram(mutations, bins=bins)
        p = pni[1:] / sum(pni[1:])
        n1s = np.random.choice(nis[1:-1], size=size, replace=True, p=p)
        n2s = np.random.choice(nis[1:-1], size=size, replace=True, p=p)
        exp_n0 = n1s * n2s / alignment_length
        n0s = np.random.poisson(lam=exp_n0, size=size)
        std_n0 = np.sqrt(exp_n0)
        ys = (n0s - exp_n0) / std_n0

        cdf = self.cdfs.loc[self.cdfs["l"] == l].values[0, 1 : l + 1]
        pn = np.diff(cdf, prepend=[0], append=[1])
        ns = np.random.choice(
            np.arange(l + 1), size=size, replace=True, p=pn / pn.sum()
        )
        nLs = n1s + n2s - 2 * n0s
        exp_n = l * (nLs + 1)
        std_n = np.sqrt(exp_n * (l + alignment_length) / alignment_length)
        xs = (ns - exp_n) / std_n
        zs = xs - ys

        return np.sort(zs)[min(int(size * pRequired(prevalence)), int(size) - 1)]

    def get_xy_thresholds(self, df):
        alignment_length = len(df["alt_sequence_alignment"].values[0])
        """self.classes["xy_threshold"] = self.classes.apply(
            lambda x: self.simulate_xs_ys(
                x.v_gene,
                x.j_gene,
                x.cdr3_length,
                x.effective_prevalence,
                df,
                alignment_length,
            ),
            axis=1,
        )"""
        mutations_grouped = []
        for (v_gene, j_gene, cdr3_length, prevalence), _ in self.classes.groupby(
            ["v_gene", "j_gene", "cdr3_length", "effective_prevalence"]
        ):
            mutations_grouped.append(
                (
                    v_gene,
                    j_gene,
                    cdr3_length,
                    prevalence,
                    df.query(
                        "v_gene==@v_gene and j_gene==@j_gene and cdr3_length==@cdr3_length"
                    )["mutation_count"],
                    alignment_length,
                )
            )

        result = applyParallel(
            mutations_grouped,
            self.simulate_xs_ys,
            cpuCount=self.threads,
            silent=self.silent,
            isint=True,
        )
        self.classes["xy_threshold"] = pd.Series(result)

    def assign_precise_sensitive_thresholds(
        self,
        args: tuple[tuple[int], pd.DataFrame],
    ) -> pd.DataFrame:
        """Assign precise and sensitive thresholds to dataframe grouped by length.

        Args:
            args tuple[tuple[int],pd.DataFrame]: length, dataframe grouped by length.

        Returns:
            pd.DataFrame: Precise and sensitive thresholds.
        """

        tuple_l, ldf = args
        l = tuple_l[0]
        if l > self.lengths[-1] or l < self.lengths[0] or l % 3 != 0:
            ldf[["precise_threshold", "sensitive_threshold"]] = (
                np.ones((len(ldf), 2), dtype=int) * l // 5
            )
            return ldf[["precise_threshold", "sensitive_threshold"]]

        rhos = ldf["effective_prevalence"]
        mus = ldf["effective_mean_distance"] * l
        bins = np.arange(l + 1)
        cdf0 = self.cdfs.loc[self.cdfs["l"] == l].values[0, 1 : l + 2]
        cdf1 = (
            np.array([mu**bins * np.exp(-mu) for mu in mus]) / factorial(bins)
        ).cumsum(axis=1)
        ps = cdf0 / cdf1
        t_sens = (cdf1 < self.sensitivity).sum(axis=1)
        t_prec = (
            np.array(
                [
                    p < rho / (1 + 1e-5 - rho) * (1 - self.precision) / self.precision
                    for p, rho in zip(ps, rhos)
                ],
            ).sum(axis=1)
            - 1
        )
        t_prec = np.min([t_prec, t_sens], axis=0)
        ldf["precise_threshold"] = t_prec
        ldf["sensitive_threshold"] = t_sens
        return ldf[["precise_threshold", "sensitive_threshold"]]

    def get_thresholds(self) -> None:
        """Assigns thresholds using null distribution from model.

        Args:
            model int: _description_. Defaults to 326713.
        """
        self.classes[["precise_threshold", "sensitive_threshold"]] = applyParallel(
            self.classes.groupby(["cdr3_length"]),
            self.assign_precise_sensitive_thresholds,
            cpuCount=self.threads,
            silent=self.silent,
        )
