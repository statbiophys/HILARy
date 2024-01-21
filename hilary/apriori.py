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
        lengths=np.arange(15, 81 + 3, 3).astype(int),
        nmax: int = int(1e5),
        precision: float = 0.99,
        sensitivity: float = 0.9,
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
            precision (float, optional): Desired precision, defaults to 0.99.
            sensitivity (float, optional): Desired sensitivity, defaults to 0.9.
            threads (int, optional): Number of cpus on which to run code, defaults to cpu_count().
            model (int, optional): Model name to infer Null distribution, defaults to 326713.
            silent (bool) : If true do not to show progress bars.
        """
        self.lengths = lengths
        self.nmax = nmax
        self.threads = threads
        self.df = df
        self.threads = threads
        self.precision = precision
        self.sensitivity = sensitivity
        self.silent = silent
        self.histograms = None
        self.mean_prevalence = None
        self.mean_mean_distance = None
        self.cdfs = pd.read_csv(
            Path(os.path.dirname(__file__)) / Path(f"cdfs_{model}.csv")
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
        return self.df

    def create_classes(self) -> pd.DataFrame:
        """Create VJl and l classes from self.df."""
        self.classes = create_classes(self.df)
        return self.classes

    def vjl2x(self, args: tuple[int, pd.DataFrame]) -> pd.DataFrame:
        """Compute CDR3 hamming distances within VJl class

        Args:
            args _, pd.DataFrame : _ , Dataframe filtered on v_gene, j_gene and length

        Returns:
            pd.DataFrame: Distance distribution for vjl class.
        """
        _, df = args
        xs = []
        for s1, s2 in combinations(df["cdr3"].values, 2):
            xs.append(hamming(s1, s2))
        x = pd.DataFrame()
        x["x"] = xs
        return x

    def l2x(self, df: pd.DataFrame) -> np.ndarray:
        """Compute CDR3 hamming distances within l class

        Args:
            df pd.DataFrame : Dataframe filtered on cdr3 length

        Returns:
            np.array: Distance distribution for l class.
        """
        x = applyParallel(
            df.groupby(["v_gene", "j_gene", "cdr3_length"]),
            self.vjl2x,
            cpuCount=self.threads,
            silent=self.silent,
        )
        return np.histogram(x.x, bins=range(self.lengths[-1] + 2), density=False)[0]

    def compute_alll(self) -> pd.DataFrame:
        """Compute histograms for all large l classes.

        Returns:
            pd.DataFrame: Histogram of distances for large l classes."""
        histograms = []
        class_ids = []
        rows = self.classes.query(
            "v_gene == 'None' and pair_count >0",
        ).iterrows()
        for _, row in rows:
            frac = min(np.sqrt(self.nmax / row.pair_count), 1)
            log.debug(
                "Computing CDR3 hamming distances within l class.",
                CDR3_length=row.cdr3_length,
            )
            h = self.l2x(
                self.df.query(
                    "cdr3_length == @row.cdr3_length",
                ).sample(frac=frac),
            )
            histograms.append(h)
            class_ids.append(row.class_id)
        results = pd.DataFrame(np.array(histograms), index=class_ids)
        results["class_id"] = results.index
        return results

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

    def compute_allvjl(self) -> pd.DataFrame:
        """Compute histograms for all large VJl classes.

        Returns:
            pd.DataFrame: Histogram of distances for large VJl classes."""
        query = "v_gene != 'None' and pair_count >0"
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
        hs_l = self.compute_alll()
        hs_vjl = self.compute_allvjl()
        self.histograms = pd.concat([hs_l, hs_vjl], ignore_index=True).sort_values(
            "class_id",
        )[["class_id"] + [*range(81 + 1)]]
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
        em = EM(l, h.values[0, 1:], positives="geometric")
        rho_geo, mu_geo = em.discreteEM()
        error_geo = em.error([rho_geo, mu_geo])
        em = EM(l, h.values[0, 1:], positives="poisson")
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

    def assign_prevalence(self, args: tuple[int, pd.DataFrame]) -> pd.DataFrame:
        """Updates prevalence of classes not large enough using the mean prevalence.

        Args:
            args tuple[int,pd.DataFrame]: length, dataframe grouped by length l.

        Returns:
            pd.DataFrame: Dataframe containing prevalence for all classes.
        """
        l, ldf = args
        p = ldf.loc[ldf["v_gene"] == "None"].prevalence.values[0]
        if np.isnan(p):
            p = self.mean_prevalence
        return ldf[["prevalence"]].fillna(p)

    def assign_mean_distance(self, args: tuple[int, pd.DataFrame]) -> pd.DataFrame:
        """Updates mean_distance of classes not large enough using the mean mean_distance.

        Args:
            args tuple[int,pd.DataFrame]: length, dataframe grouped by length l.

        Returns:
            pd.DataFrame: Dataframe containing mean_distance for all classes.
        """
        l, ldf = args
        m = ldf.loc[ldf["v_gene"] == "None"].mean_distance.values[0]
        if np.isnan(m):
            m = self.mean_mean_distance
        return ldf[["mean_distance"]].fillna(m)

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

        loc = self.classes["v_gene"] == "None"
        loc = loc & (~self.classes["prevalence"].isna())
        loc = loc & (~self.classes["mean_distance"].isna())
        ns = self.classes.loc[loc]["pair_count"]
        rhos = self.classes.loc[loc]["prevalence"]
        mus = self.classes.loc[loc]["mean_distance"]
        ws = ns * (ns - 1)
        ws = ws / sum(ws)
        self.mean_prevalence = sum(ws * rhos)
        self.mean_mean_distance = sum(ws * mus)
        log.debug("Assigning effective prevalence.")
        self.classes["effective_prevalence"] = applyParallel(
            self.classes.groupby(["cdr3_length"]),
            self.assign_prevalence,
            cpuCount=self.threads,
            silent=self.silent,
        )
        log.debug("Assigning effective mean distance")
        self.classes["effective_mean_distance"] = applyParallel(
            self.classes.groupby(["cdr3_length"]),
            self.assign_mean_distance,
            cpuCount=self.threads,
            silent=self.silent,
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
