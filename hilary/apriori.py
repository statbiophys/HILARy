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
from hilary.utils import applyParallel, cdf_to_pmf, preprocess, return_cdf

pd.set_option("mode.chained_assignment", None)

log = structlog.get_logger(__name__)


class Apriori:
    """Computes statistics of pairwise distances."""

    def __init__(
        self,
        nmax: int = int(1e5),
        precision: float = 1.0,
        sensitivity: float = 1.0,
        threads: int = 1,
        species:  str = "human",
        silent: bool = False,
        paired: bool = False,
        null_model:str = "vjl",
        recenter_mean:bool=False,
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
            species  (str) : species  of the repertoire.
            silent (bool) : If true do not to show progress bars.
            paired (bool) : If true use null distributions over paired chain sequences.
        """
        self.nmax = nmax
        self.threads = threads if threads > 0 else cpu_count()
        self.precision = precision - 1e-4
        self.sensitivity = sensitivity
        self.silent = silent
        self.paired = paired
        self.histograms = None
        self.mean_prevalence = None
        self.mean_mean_distance = None
        self.check_translation = False
        self.species = species
        self.null_model = null_model
        self.recenter_mean=recenter_mean
        if not paired:
            if species =="human":
                self.lengths=np.arange(15, 81 + 3, 3).astype(int)
            elif species =="mouse":
                self.lengths=np.arange(12, 66 + 3, 3).astype(int)
            else:
                msg = f"Unknown species: {species}"
                raise ValueError(msg)
            self.cdf_path=Path(os.path.dirname(__file__)) / Path(f"cdfs/cdfs_{species}_vjl.parquet")
        else:
            self.null_model="l"
            if species =="human":
                self.lengths = np.arange(57, 144 + 3, 3).astype(int)
            elif species =="mouse":
                msg = "Paired method for mouse not implemented yet."
                raise ValueError(msg)
            else:
                msg = f"Unknown species      : {species     }"
                raise ValueError(msg)

            self.cdf_path=Path(os.path.dirname(__file__)) / Path("cdfs/cdfs_paired.parquet")

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
        if self.species == "mouse" and self.paired: # to remove when implemented
            msg="Paired method not working for mouse species  yet"
            raise ValueError(msg)
        df = preprocess(
            df,
            silent=self.silent,
        )
        if self.species == "mouse" and "IGHJ0-7IA7" not in np.unique(df.j_gene):  # mouse translation to imgt
                translation_df = pd.read_csv(Path(os.path.dirname(__file__)) / Path("cdfs/mouse_ogrdb2imgt.csv"))
                translation_dict = dict(
                    zip(translation_df.values[:, 0], translation_df.values[:, 1])
                )
                translation_dict[np.nan] = np.nan
                df.j_gene = df.j_gene.apply(lambda x: translation_dict[x])
                df.v_gene = df.v_gene.apply(lambda x: translation_dict[x])
        if self.paired:
            df_kappa = preprocess(df_kappa, silent=self.silent)
            for column in df.columns:
                if column == "sequence_id":
                    continue
                df[column + "_h"] = df[column]
                df[column + "_k"] = df_kappa[column]
                df[column] = df[column + "_h"] + df[column + "_k"]
        return df

    def vjls2x(self, args: tuple[int, pd.DataFrame])->pd.DataFrame:
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

    def estimate(self, args: tuple[int, pd.DataFrame]) -> pd.DataFrame:
        """Fit prevalence and mu using the histogram which is the distribution of distances.

        Args:
            args tuple[int, pd.DataFrame]: class_id, histogram for that class

        Returns
        -------
            pd.DataFrame: dataframe with parameters for each class id.
        """
        class_id, h = args
        if not isinstance(class_id, int):
            class_id = class_id[0]
        classes_temp = self.classes.loc[self.classes.class_id == class_id]
        l = classes_temp.cdr3_length.values[0]
        v_gene = classes_temp.v_gene.values[0]
        j_gene = classes_temp.j_gene.values[0]
        histo=h.values[0, 1:].astype(int)[: l + 1]
        cdf_list=[]
        names=[]
        if self.null_model=="vjl": # probably a better way to code that
            cdf_df_vjl = return_cdf(self.cdf_path, v_gene=v_gene, j_gene=j_gene, cdr3_length=l)
            cdf_df_jl = return_cdf(self.cdf_path, v_gene="None", j_gene=j_gene, cdr3_length=l)
            cdf_df_l = return_cdf(self.cdf_path, v_gene="None", j_gene="None", cdr3_length=l)
            cdf_list.extend([cdf_df_vjl, cdf_df_jl, cdf_df_l])
            names.extend(["VJL","JL","L"])
        elif self.null_model=="jl":
            cdf_df_jl = return_cdf(self.cdf_path, v_gene="None", j_gene=j_gene, cdr3_length=l)
            cdf_df_l = return_cdf(self.cdf_path, v_gene="None", j_gene="None", cdr3_length=l)
            cdf_list.extend([cdf_df_jl, cdf_df_l])
            names.extend(["JL","L"])
        elif self.null_model=="l":
            cdf_df_l = return_cdf(self.cdf_path, v_gene="None", j_gene="None", cdr3_length=l)
            cdf_list.extend([cdf_df_l])
            names.extend(["L"])
        else:
            msg=f"Unknown CDF null model : {self.null_model}"
            raise ValueError(msg)

        min_error=np.inf
        best_cdf0 = cdf_list[-1].values[0,3:3+l+1]
        best_rho=0
        best_mu=0
        null_model="None"
        for i,cdf in enumerate(cdf_list):
            if cdf.empty: # did not find null model for vjl or jl
                continue
            cdf0=cdf.values[0,3:3+l+1]
            if self.recenter_mean: # change with truncated mean
                histo_pmf = histo/histo.sum()
                pmf0=cdf_to_pmf(cdf0)
                shift = int(np.round(np.mean(histo_pmf[l//5:])-np.mean(pmf0[l//5:])))
                new_pmf0 = np.empty_like(pmf0)
                if shift>0:
                    new_pmf0[:shift]=0
                    new_pmf0[shift:]=pmf0[:-shift]
                if shift<0:
                    new_pmf0[shift:]=0
                    new_pmf0[:shift]=pmf0[-shift:]

                if shift!=0:
                    cdf0 = np.cumsum(new_pmf0)

            em = EM(cdf=cdf0, l=l, h=histo, positives="poisson")
            rho_poisson, mu_poisson = em.discreteEM()
            error = em.error([rho_poisson, mu_poisson])
            if error<=min_error:
                best_cdf0 = cdf.values[0,3:3+l+1]
                min_error=error
                best_rho = rho_poisson
                best_mu = mu_poisson
                null_model = names[i] # what null model is actually being used

        prevalence=best_rho
        bins = np.arange(l + 1)
        cdf1 = ((best_mu**bins * np.exp(-best_mu)) / factorial(bins)).cumsum()
        p = best_cdf0 / cdf1
        t_sens = (cdf1 < self.sensitivity).sum()
        t_prec = (p < prevalence / (1 + 1e-5 - prevalence) * (1 - self.precision) / self.precision).sum() - 1
        t_prec = np.min([t_prec, t_sens], axis=0)

        result = pd.DataFrame(
            columns=[
                "class_id",
                "prevalence",
                "mu",
                "error",
                "t_prec",
                "t_sens",
                "null_model",
            ],
        )
        result.class_id = [class_id]
        result.null_model = [null_model]
        result.t_prec = [t_prec]
        result.t_sens = [t_sens]
        result.prevalence = [prevalence]
        result.mu = [best_mu]
        result.error = [min_error]
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
        parameters = applyParallel(
            self.histograms.groupby(["class_id"]),
            self.estimate,
            cpuCount=self.threads,
            silent=self.silent,
        ).reset_index(drop=True)
        self.classes.index = self.classes.class_id
        parameters.index = parameters.class_id
        self.classes["prevalence"] = parameters["prevalence"]
        self.classes["null_model"] = parameters["null_model"]

        self.classes["error"] = parameters["error"]
        self.classes["mean_distance"] = parameters["mu"] / self.classes["cdr3_length"]
        self.classes["effective_prevalence"] = self.classes["prevalence"].fillna(0.2)
        self.classes["effective_mean_distance"] = self.classes["mean_distance"].fillna(
            0.04,
        )
        self.classes["precise_threshold"]=parameters["t_prec"]
        self.classes["sensitive_threshold"]=parameters["t_sens"]
        self.classes["precise_threshold"] = self.classes["precise_threshold"].fillna(self.classes["cdr3_length"] // 5).astype(int)
        self.classes["sensitive_threshold"] = self.classes["sensitive_threshold"].fillna(self.classes["cdr3_length"] // 5).astype(int)

    def return_fit(self, class_id:int):
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
        v_gene = v.v_gene.values[0]
        j_gene=v.j_gene.values[0]
        null_model = v.null_model.values[0]
        cdr3_length = v.cdr3_length.values[0]
        bins = np.arange(cdr3_length + 1)
        hist_data = self.histograms.loc[self.histograms.class_id == class_id].values[
            0, 1 : cdr3_length + 2
        ]
        mu = v.effective_mean_distance.values[0]
        prevalence = v.prevalence.values[0]

        cdf_df_vjl = return_cdf(self.cdf_path, v_gene=v_gene, j_gene=j_gene, cdr3_length=cdr3_length)
        cdf_df_jl = return_cdf(self.cdf_path, v_gene="None", j_gene=j_gene, cdr3_length=cdr3_length)
        cdf_df_l = return_cdf(self.cdf_path, v_gene="None", j_gene="None", cdr3_length=cdr3_length)

        mode=""
        if self.null_model =="vjl" and not cdf_df_vjl.empty:
            cdf_df = cdf_df_vjl
            mode="VJL"
        elif not cdf_df_jl.empty:
            cdf_df = cdf_df_jl
            mode="JL"
        elif not cdf_df_l.empty:
            mode="L"
            cdf_df = cdf_df_l
        else:
            msg=f"CDR3 length {cdr3_length} not available in CDFs."
            raise ValueError(msg)

        cdf0=cdf_df.values[0,3:3+cdr3_length+1]
        cdf1 = ((mu**bins * np.exp(-mu)) / factorial(bins)).cumsum()
        fitted_distribution = prevalence * poisson.pmf(bins, mu * cdr3_length) + (
            1 - prevalence
        ) * cdf_to_pmf(cdf0)
        return (
            null_model,
            mode,
            bins,
            cdf_to_pmf(cdf0),
            cdf_to_pmf(cdf1),
            prevalence,
            fitted_distribution,
            hist_data / sum(hist_data),
        )
