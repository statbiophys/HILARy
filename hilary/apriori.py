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
from hilary.utils import applyParallel, create_classes, preprocess

pd.set_option("mode.chained_assignment", None)

log = structlog.get_logger(__name__)


class Apriori:
    """Computes statistics of pairwise distances"""

    def __init__(
        self,
        df: pd.DataFrame,
        dataframe_kappa: pd.DataFrame = None,
        lengths: np.ndarray = np.arange(15, 81 + 3, 3).astype(int),
        nmax: int = int(1e5),
        precision: float = 1.0,
        sensitivity: float = 1.0,
        threads: int = cpu_count() - 1,
        model: int = 326713,
        silent: bool = False,
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
        self.df = df
        self.dataframe_kappa = dataframe_kappa
        self.threads = threads
        self.precision = precision - 1e-4
        self.sensitivity = sensitivity
        self.silent = silent
        self.histograms = None
        self.mean_prevalence = None
        self.mean_mean_distance = None
        if dataframe_kappa is None:
            self.cdfs = pd.read_csv(
                Path(os.path.dirname(__file__)) / Path(f"cdfs_{model}.csv"),
            )
        else:
            self.cdfs = pd.read_csv(
                Path(os.path.dirname(__file__)) / Path("cdfs_paired.csv"),
            )
        self.preprocess()
        self.classes = self.create_classes()

    def preprocess(self) -> pd.DataFrame:
        """Remove non productive sequences from dataframe.

        Returns:
            pd.Dataframe: Dataframe self.df containing all sequences.
        """
        self.df = preprocess(
            self.df,
            silent=self.silent,
        )
        if self.dataframe_kappa is not None:
            self.dataframe_kappa = preprocess(self.dataframe_kappa, silent=self.silent)
            for column in self.df.columns:
                if column == "sequence_id":
                    continue
                self.df[column + "_h"] = self.df[column]
                self.df[column + "_k"] = self.dataframe_kappa[column]
                self.df[column] = self.df[column + "_h"] + self.df[column + "_k"]
        return self.df

    def create_classes(self) -> pd.DataFrame:
        """Create VJl and l classes from self.df."""
        self.classes = create_classes(self.df)
        return self.classes

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
    ) -> pd.DataFrame:
        """Compute histograms for all large VJl classes.

        Returns:
            pd.DataFrame: Histogram of distances for large VJl classes."""
        query = "v_gene != 'None' and pair_count >0 and cdr3_length in @self.lengths"
        groups = self.df.groupby(["v_gene", "j_gene", "cdr3_length"])
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
        if results.empty:
            results = pd.DataFrame(columns=[*range(81 + 1)])
        results["class_id"] = results.index
        return results

    def get_histograms(self) -> pd.DataFrame:
        """Compute histograms for all large classes

        Returns:
            pd.DataFrame: Histogram of distances for all large classes."""
        hs_vjl = self.compute_allvjl()
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
