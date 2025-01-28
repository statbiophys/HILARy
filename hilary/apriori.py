"""Code to compute prevalence and thresholds."""

from __future__ import annotations

import os
from itertools import combinations
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from scipy.special import factorial
from scipy.stats import poisson
from textdistance import hamming

from hilary.expectmax import EM
from hilary.utils import applyParallel, cdf_to_pmf, preprocess, return_cdf, select_df

pd.set_option("mode.chained_assignment", None)

log = structlog.get_logger(__name__)


class Apriori:
    """Computes statistics of pairwise distances."""

    def __init__(
        self,
        lengths: np.ndarray = np.arange(15, 81 + 3, 3).astype(int),
        nmax: int = int(1e5),
        precision: float = 1.0,
        sensitivity: float = 1.0,
        threads: int = 1,
        model: str = "human_jl",
        silent: bool = False,
        paired: bool = False,
        selection_cdfs: float = 0.02,
    ) -> None:
        """Initialize attributes to later run class methods.

        Args:
            lengths (_type_, optional): CDR3 lengths used to filter non productive sequences.
                Defaults to np.arange(15, 81 + 3, 3).
            nmax (int, optional): For parameter inference, sample and use nmax sequences for
                classes larger than nmax. Defaults to 100000.
            precision (float, optional): Desired precision, defaults to 1.
            sensitivity (float, optional): Desired sensitivity, defaults to 1.
            threads (int, optional): Number of cpus on which to run code, defaults to 1. -1 to use
            all available cpus.
            model (int, optional): Model name to infer Null distribution, defaults to 326713.
            silent (bool) : If true do not to show progress bars.
            paired (bool) : If true use null distributions over paired chain sequences.
        """
        self.lengths = lengths
        self.nmax = nmax
        self.threads = threads if threads > 0 else cpu_count()
        self.precision = precision - 1e-4
        self.sensitivity = sensitivity
        self.silent = silent
        self.paired = paired
        self.histograms = None
        self.mean_prevalence = None
        self.mean_mean_distance = None
        self.selection_cdfs = selection_cdfs
        self.check_translation = False
        if "mouse" in model:
            self.check_translation = True
        if not paired:
            self.cdfs = pd.read_csv(
                Path(os.path.dirname(__file__)) / Path(f"cdfs/cdfs_{model}.csv"),
            )
        else:
            self.cdfs = pd.read_csv(
                Path(os.path.dirname(__file__)) / Path("cdfs/cdfs_paired.csv"),
            )
        self.classes = pd.DataFrame()

    def preprocess(self, df: pd.DataFrame, df_kappa: pd.DataFrame | None = None) -> pd.DataFrame:
        """Remove non productive sequences from dataframe.

        If df_kappa is not null then group VH, JH, VK and JK genes together and concatenate heavy
        and light cdr3s.

        Args:
            df (pd.DataFrame): dataframe of heavy chain sequences.
            df_kappa (pd.DataFrame): dataframe of light chain sequences.

        Returns
        -------
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

    def vjls2x(self, args: tuple[int, pd.DataFrame]):
        """Compute histogram for a given VJl class."""
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
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute histograms for all large VJl classes.

        Args:
            df(pd.DataFrame): Dataframe of sequences.

        Returns
        -------
            pd.DataFrame: Histogram of distances for large VJl classes.
        """
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

    def select_cdfs(self) -> None:
        """Selects VJL only CDFs with large differences with respect to JL CDFs and
        that are present in the data.

        Returns
        -------
            None
        """
        log.debug(
            "Select CDFs for the analysis.",
        )
        selected_dfs = applyParallel(
            self.cdfs.groupby(["j_gene", "cdr3_length"]),
            func=select_df,
            cpuCount=self.threads,
            silent=True,
        )
        v_gene_nans = self.cdfs.isna().v_gene
        j_gene_nans = self.cdfs.isna().j_gene
        jdf = self.cdfs.loc[np.logical_and(v_gene_nans, ~j_gene_nans)]
        ldf = self.cdfs.loc[np.logical_and(v_gene_nans, j_gene_nans)]
        self.cdfs = pd.concat([selected_dfs, jdf, ldf]).reset_index(drop=True)
        m = self.cdfs.fillna("None")[["v_gene", "j_gene", "cdr3_length"]]
        m["index_values"] = m.index.values
        index = m.merge(self.classes[["v_gene", "j_gene", "cdr3_length"]]).index_values
        self.cdfs = self.cdfs.iloc[index].reset_index(
            drop=True
        )  # This is a hack to avoid memory issues

    def get_histograms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute histograms for all large classes.

        Args:
            df(pd.DataFrame) : Dataframe of sequences.

        Returns
        -------
            pd.DataFrame: Histogram of distances for all large classes.
        """
        hs_vjl = self.compute_allvjl(df)
        self.histograms = hs_vjl.sort_values(
            "class_id",
        )[["class_id"] + [*range(self.lengths[-1] + 1)]]
        return self.histograms

    def estimate(self, args: tuple[tuple[int], pd.DataFrame]) -> pd.DataFrame:
        """Fit prevalence and mu using the histogram which is the distribution of distances.

        Args:
            args tuple[int, pd.DataFrame]: class_id, histogram for that class

        Returns
        -------
            pd.DataFrame: dataframe with parameters for each class id.
        """
        class_id, h = args
        l = self.classes.loc[self.classes.class_id == class_id].cdr3_length.values[0]
        cdf = return_cdf(self.classes, self.cdfs, class_id)
        em = EM(cdf=cdf, l=l, h=h.values[0, 1:], positives="geometric")
        rho_geo, mu_geo = em.discreteEM()
        error_geo = em.error([rho_geo, mu_geo])
        em = EM(cdf=cdf, l=l, h=h.values[0, 1:], positives="poisson")
        rho_poisson, mu_poisson = em.discreteEM()
        error_poisson = em.error([rho_geo, mu_geo])
        result = pd.DataFrame(
            columns=[
                "class_id",
                "rho_geo",
                "mu_geo",
                "rho_poisson",
                "mu_poisson",
                "error_poisson",
                "error_geo",
            ],
        )
        result.class_id = [class_id[0]]
        result.rho_geo = [rho_geo]
        result.mu_geo = [mu_geo]
        result.rho_poisson = [rho_poisson]
        result.mu_poisson = [mu_poisson]
        result.error_poisson = [error_poisson]
        result.error_geo = [error_geo]
        return result

    def get_parameters(self) -> None:
        """Compute prevalence and mean distance for all classes."""
        if self.histograms is None:
            msg = "Histogram attribute is None. Please run get_histograms method."
            raise ValueError(
                msg,
            )
        log.debug(
            "Computing prevalence and mean distance for all classes",
        )
        if self.check_translation:
            if not "IGHJ0-7IA7" in np.unique(self.classes.j_gene):  # mouse translation to imgt
                translation_df = pd.read_csv("~/mouse/victor_mouse/mouse_ogrdb2imgt.csv")
                translation_dict = dict(
                    zip(translation_df.values[:, 0], translation_df.values[:, 1])
                )
                translation_dict[np.nan] = np.nan
                self.cdfs.j_gene = self.cdfs.j_gene.apply(lambda x: translation_dict[x])
                if "v_gene" in self.cdfs.columns:
                    self.cdfs.v_gene = self.cdfs.v_gene.apply(lambda x: translation_dict[x])

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
        self.classes["error_geo"] = parameters["error_geo"]
        self.classes["error_poisson"] = parameters["error_poisson"]
        self.classes["mean_distance"] = parameters["mu_poisson"] / self.classes["cdr3_length"]
        self.classes["effective_prevalence"] = self.classes["prevalence"].fillna(0.2)
        self.classes["effective_mean_distance"] = self.classes["mean_distance"].fillna(
            0.04,
        )

    def assign_precise_sensitive_thresholds(
        self,
        args: tuple[tuple[int], pd.DataFrame],
    ) -> pd.DataFrame:
        """Assign precise and sensitive thresholds to dataframe grouped by length.

        Args:
            args tuple[tuple[int],pd.DataFrame]: length, dataframe grouped by length.

        Returns
        -------
            pd.DataFrame: Precise and sensitive thresholds.
        """
        _, ldf = args
        class_id = ldf.class_id.values[0]
        cdr3_length = ldf.cdr3_length.values[0]
        rho = ldf.effective_prevalence.values[0]
        mu = ldf.effective_mean_distance.values[0] * cdr3_length
        err_poisson = ldf.error_poisson.values[0]
        err_geo = ldf.error_geo.values[0]
        min_err = np.min([err_poisson, err_geo])
        threshold_90 = cdr3_length // 10
        threshold_80 = cdr3_length // 5

        # if outside the regime where we have cdfs
        if "v_gene" in self.cdfs.columns and "j_gene" in self.cdfs.columns:
            cdr3_dfs = self.cdfs.loc[
                np.logical_and(self.cdfs["j_gene"].isna(), self.cdfs["v_gene"].isna())
            ]
        elif "j_gene" in self.cdfs.columns:
            cdr3_dfs = self.cdfs.loc[self.cdfs["j_gene"].isna()]
        else:
            cdr3_dfs = self.cdfs
        if (
            cdr3_length > cdr3_dfs.cdr3_length.max()
            or cdr3_length < cdr3_dfs.cdr3_length.min()
            or cdr3_length % 3
        ):
            ldf["precise_threshold"] = threshold_90
            ldf["sensitive_threshold"] = threshold_80
            return ldf[["precise_threshold", "sensitive_threshold"]]
        bins = np.arange(cdr3_length + 1)
        cdf0 = return_cdf(self.classes, self.cdfs, class_id, extend=1)
        cdf1 = ((mu**bins * np.exp(-mu)) / factorial(bins)).cumsum()
        p = cdf0 / cdf1
        t_sens = (cdf1 < self.sensitivity).sum()
        t_prec = (p < rho / (1 + 1e-5 - rho) * (1 - self.precision) / self.precision).sum() - 1
        t_prec = np.min([t_prec, t_sens], axis=0)
        ldf["precise_threshold"] = t_prec
        ldf["sensitive_threshold"] = t_sens

        if min_err > 0.1:
            # at high error threhsold use default thresholds
            ldf["precise_threshold"] = np.min([t_prec, threshold_90])
            ldf["sensitive_threshold"] = np.min([t_sens, threshold_80])
        return ldf[["precise_threshold", "sensitive_threshold"]]

    def get_thresholds(self) -> None:
        """
        Assign thresholds using null distribution from model.

        This method calculates and assigns precise and sensitive thresholds for each class
        based on the null distribution from the model. It uses parallel processing to
        speed up the computation by applying the `assign_precise_sensitive_thresholds`
        function to groups of classes defined by their 'v_gene', 'j_gene', and 'cdr3_length'
        attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        log.debug(
            "Assign thresholds using null distribution from model.",
        )
        self.classes[["precise_threshold", "sensitive_threshold"]] = applyParallel(
            self.classes.groupby(["v_gene", "j_gene", "cdr3_length"]),
            self.assign_precise_sensitive_thresholds,
            cpuCount=self.threads,
            silent=self.silent,
        )

    def return_fit(self, class_id):
        """
        Return fits of the distribution to the histogram data for a given class ID.

        Parameters
        ----------
        class_id (int): The ID of the class for which the fit is to be returned.

        Returns
        -------
        tuple: A tuple containing the following elements:
            - bins (numpy.ndarray): The bin edges for the histogram.
            - cdf0 (numpy.ndarray): The cumulative distribution function (CDF) for the negative distribution.
            - cdf1 (numpy.ndarray): The Poisson CDF for the positive distribution.
            - prevalence (float): The prevalence of the class.
            - fitted_distribution (numpy.ndarray): The fitted distribution for the class.
            - hist_data_normalized (numpy.ndarray): The normalized histogram data for the class.
        """
        v = self.classes.loc[self.classes.class_id == class_id]
        cdr3_length = v.cdr3_length.values[0]
        bins = np.arange(cdr3_length + 1)
        hist_data = self.histograms.loc[self.histograms.class_id == class_id].values[
            0, 1 : cdr3_length + 2
        ]
        mu = v.effective_mean_distance.values[0]
        prevalence = v.prevalence.values[0]
        cdf0 = return_cdf(self.classes, self.cdfs, v.class_id.values[0], extend=1)
        cdf1 = ((mu**bins * np.exp(-mu)) / factorial(bins)).cumsum()
        fitted_distribution = prevalence * poisson.pmf(bins, mu * cdr3_length) + (
            1 - prevalence
        ) * cdf_to_pmf(cdf0)
        return (
            bins,
            cdf_to_pmf(cdf0),
            cdf_to_pmf(cdf1),
            prevalence,
            fitted_distribution,
            hist_data / sum(hist_data),
        )
