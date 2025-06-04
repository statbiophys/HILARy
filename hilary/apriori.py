"""Code to compute prevalence and thresholds."""

from __future__ import annotations

from itertools import combinations
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from scipy.special import factorial
from textdistance import hamming

from hilary.expectmax import EM
from hilary.utils import apply_chunked_parallel, cdf_to_pmf, preprocess, return_cdf

pd.set_option("mode.chained_assignment", None)

log = structlog.get_logger(__name__)

DEFAULT_PREVALENCE = 0.2
DEFAULT_MEAN_DISTANCE = 0.04


class Apriori:
    """Computes statistics of pairwise distances."""

    def __init__(  # noqa: PLR0913
        self,
        precision: float = 1.0,
        sensitivity: float = 1.0,
        threads: int = 1,
        null_model: str = "vjl",
        model: str = "human_B_heavy",
        *,
        silent: bool = False,
        paired: bool = False,
    ) -> None:
        """Initialize attributes to later run class methods.

        Args:
            precision (float, optional): Desired precision, defaults to 1.
            sensitivity (float, optional): Desired sensitivity, defaults to 1.
            threads (int, optional): Number of cpus on which to run code, defaults to 1. -1 to use
            all available cpus.
            silent (bool) : If true do not to show progress bars.
            paired (bool) : If true use null distributions over paired chain sequences.
            model (str) : Model to use among 'human_B_heavy','human_B_kappa','human_B_lambda',\
                'human_paired', 'mouse_B_heavy','mouse_B_kappa','mouse_B_lambda','mouse_B_paired'.\
                Defaul to 'human_B_heavy'.
            null_model(str) : Whether to use null model on vjl, jl or l class. Default to vjl.
        """
        self.threads = threads if threads > 0 else cpu_count()
        self.precision = precision - 1e-4
        self.sensitivity = sensitivity
        self.silent = silent
        self.paired = paired
        self.histograms = pd.DataFrame
        self.mean_prevalence = None
        self.mean_mean_distance = None
        self.check_translation = False
        self.null_model = null_model
        self.model = model
        # Fill default values for prevalence and mean_distance

        if not paired:
            if self.model=="human_B_heavy":
                self.lengths = np.arange(15, 81 + 3, 3).astype(int)
            elif "human" in self.model:
                self.lengths = np.arange(15, 63 + 3, 3).astype(int)
            elif "mouse" in self.model:
                self.lengths = np.arange(15, 66 + 3, 3).astype(int)
            else:
                msg = f"Unknown model: {self.model}"
                raise ValueError(msg)
            self.cdf_path = Path(__file__).parent / f"cdfs/{model}.parquet"
        else:
            self.null_model = "jl"
            if "human" in self.model:
                self.lengths = np.arange(30, 141 + 3, 3).astype(int)
                self.cdf_path = Path(__file__).parent / f"cdfs/{self.model}.parquet"
            elif "mouse" in self.model:
                self.lengths = np.arange(21, 102 + 3, 3).astype(int)
                self.cdf_path = Path(__file__).parent / f"cdfs/{self.model}.parquet"
            else:
                msg = f"Unknown model: {self.model}"
                raise ValueError(msg)
        self.classes = pd.DataFrame()

    def preprocess(self, df: pd.DataFrame, df_light: pd.DataFrame | None = None) -> pd.DataFrame:
        """Remove non productive sequences from dataframe.

        If df_light is not null then group VH, JH, VK and JK genes together and concatenate heavy
        and light cdr3s.

        Args:
            df (pd.DataFrame): dataframe of heavy chain sequences.
            df_light (pd.DataFrame): dataframe of light chain sequences.

        Returns
        -------
            pd.Dataframe: Dataframe self.df containing all sequences.
        """
        df = preprocess(
            df,
            silent=self.silent,
        )
        if "mouse" in self.model and "IGHJ0-7IA7" not in np.unique(
            df.j_gene
        ):  # mouse translation to imgt
            translation_df = pd.read_csv(Path(__file__).parent / "cdfs/mouse_ogrdb2imgt.csv")
            translation_dict = dict(zip(translation_df.values[:, 0], translation_df.values[:, 1]))
            translation_dict[np.nan] = np.nan
            df.j_gene = df.j_gene.apply(lambda x: translation_dict[x])
            df.v_gene = df.v_gene.apply(lambda x: translation_dict[x])
        if self.paired:
            df_light = preprocess(df_light, silent=self.silent)
            for column in df.columns:
                if column == "sequence_id":
                    continue
                df[column + "_h"] = df[column]
                df[column + "_k"] = df_light[column]
                if column=="mutation_count":
                    df[column] = df[column + "_h"] + df[column + "_k"]
                else:
                    df[column] = df[column + "_h"].astype(str) + "," + df[column + "_k"].astype(str)
        return df

    def vjls2x(self, args: tuple[int, pd.DataFrame]) -> pd.DataFrame:
        """Compute histogram for a given VJl class."""
        i, df = args
        xs = [hamming(s1, s2) for s1, s2 in combinations(df["cdr3"].values, 2)]
        return pd.DataFrame(
            np.histogram(
                xs,
                bins=range(
                    self.lengths[-1] + 2,
                ),
                density=False,
            )[0],
            columns=i,
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
        # query to select only the classes with v_gene != None and pair_count > 0
        df.cdr3_length = df.cdr3_length.astype(str)
        df = df.merge(
            self.classes.query('v_gene!="None" and pair_count>0')[
                ["class_id", "v_gene", "j_gene", "cdr3_length"]
            ],
            on=["v_gene", "j_gene", "cdr3_length"],
            how="inner",
        )
        log.debug(
            "Computing CDR3 hamming distances within all large VJl classes.",
        )

        results = apply_chunked_parallel(
            df.groupby(["class_id"]),
            self.vjls2x,
            cpu_count=self.threads,
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
        # add cdr3_length_value to classes for computation
        if self.paired:
            self.classes["cdr3_length_value"] = self.classes.cdr3_length.apply(
                lambda x: int(x.split(",")[0]) + int(x.split(",")[1])
            )
        else:
            self.classes["cdr3_length_value"] = self.classes.cdr3_length.astype(int)
        hs_vjl = self.compute_allvjl(df)
        self.histograms = hs_vjl.sort_values(
            "class_id",
        )[["class_id", *range(self.lengths[-1] + 1)]]
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
        cdr3_length_old = classes_temp.cdr3_length_value.values[0]
        cdr3_length = np.clip(cdr3_length_old, np.min(self.lengths), np.max(self.lengths))
        v_gene, j_gene = classes_temp.v_gene.values[0], classes_temp.j_gene.values[0]
        histo = h.values[0, 1:].astype(int)[: cdr3_length + 1]

        cdf_df_vjl = return_cdf(
            self.cdf_path, v_gene=v_gene, j_gene=j_gene, cdr3_length=cdr3_length
        )
        cdf_df_jl = return_cdf(self.cdf_path, v_gene="None", j_gene=j_gene, cdr3_length=cdr3_length)
        cdf_df_l = return_cdf(self.cdf_path, v_gene="None", j_gene="None", cdr3_length=cdr3_length)
        if (self.null_model in ["vjl"]) and (not cdf_df_vjl.empty):
            null_model_used = "vjl"
            cdf = cdf_df_vjl
        elif (self.null_model in ["vjl", "jl"]) and (not cdf_df_jl.empty):
            null_model_used = "jl"
            cdf = cdf_df_jl
        elif (self.null_model in ["vjl", "jl", "l"]) and (not cdf_df_l.empty):
            null_model_used = "l"
            cdf = cdf_df_l
        else:
            msg = f"Unknown {self.null_model} null model or cdf not found"
            raise ValueError(msg)
        cdf0 = cdf.values[0, 3 : 3 + cdr3_length + 1]
        em = EM(cdf=cdf0, h=histo)
        prevalence, mu = em.discrete_em()
        error = em.error([prevalence, mu])
        bins = np.arange(cdr3_length + 1)
        cdf1 = ((mu**bins * np.exp(-mu)) / factorial(bins)).cumsum()
        p = cdf0 / cdf1
        t_sens = (cdf1 < self.sensitivity).sum()
        t_prec = (
            p < prevalence / (1 + 1e-5 - prevalence) * (1 - self.precision) / self.precision
        ).sum() - 1
        t_prec = np.min([t_prec, t_sens], axis=0)

        pdf0, pdf1 = cdf_to_pmf(cdf0), cdf_to_pmf(cdf1)
        tp_p, tp_s = (prevalence * pdf1[: t_prec + 1]).sum(), (
            prevalence * pdf1[: t_sens + 1]
        ).sum()
        fp_p, fp_s = ((1 - prevalence) * pdf0[: t_prec + 1]).sum(), (1 - prevalence) * pdf0[
            : t_sens + 1
        ].sum()
        fn_p, fn_s = (prevalence * pdf1[t_prec + 1 :]).sum(), (
            prevalence * pdf1[t_sens + 1 :]
        ).sum()

        result = pd.DataFrame(
            columns=[
                "class_id",
                "prevalence",
                "mu",
                "error",
                "t_prec",
                "t_sens",
                "null_model_used",
                "est_precision_tprec",
                "est_sensitivity_tprec",
                "est_precision_tsens",
                "est_sensitivity_tsens",
            ],
        )
        result.class_id = [class_id]
        result.t_prec = [t_prec]
        result.t_sens = [t_sens]
        result.prevalence = [prevalence]
        result.mu = [mu]
        result.error = [error]
        result.est_precision_tprec = [tp_p / (tp_p + fp_p + 1e-6)]
        result.est_sensitivity_tprec = [tp_p / (tp_p + fn_p + 1e-6)]
        result.est_precision_tsens = [tp_s / (tp_s + fp_s + 1e-6)]
        result.est_sensitivity_tsens = [tp_s / (tp_s + fn_s + 1e-6)]
        result.null_model_used = [null_model_used]
        return result

    def get_parameters(self) -> None:
        """Compute prevalence and mean distance for all classes."""
        if self.histograms.empty:
            msg = "Histogram is empty. Please run get_histograms method."
            raise ValueError(msg)
        log.debug("Computing prevalence and mean distance for all classes")
        parameters = apply_chunked_parallel(
            self.histograms.groupby(["class_id"]),
            self.estimate,
            cpu_count=self.threads,
            silent=self.silent,
        ).reset_index(drop=True)
        self.classes.index = self.classes.class_id
        parameters.index = parameters.class_id
        assign_cols = [
            "prevalence",
            "null_model_used",
            "error",
            "mu",
            "t_prec",
            "t_sens",
            "est_precision_tprec",
            "est_sensitivity_tprec",
            "est_precision_tsens",
            "est_sensitivity_tsens",
        ]
        for col in assign_cols:
            if col in parameters.columns:
                self.classes[col] = parameters[col]
            else:
                log.warning("Column missing in parameters, skipping assignment", column=col)

        self.classes["mean_distance"] = self.classes["mu"] / self.classes["cdr3_length_value"]
        self.classes["effective_prevalence"] = self.classes["prevalence"].fillna(DEFAULT_PREVALENCE)
        self.classes["effective_mean_distance"] = self.classes["mean_distance"].fillna(
            DEFAULT_MEAN_DISTANCE
        )
        self.classes["precise_threshold"] = (
            self.classes["t_prec"].fillna(self.classes["cdr3_length_value"] // 20).astype(int)
        )

        self.classes["sensitive_threshold"] = (
            self.classes["t_sens"].fillna(self.classes["cdr3_length_value"] // 10).astype(int)
        )
