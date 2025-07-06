"""Code to compute prevalence and thresholds."""

from __future__ import annotations

from multiprocessing import cpu_count

import pandas as pd
import structlog

from hilary.utils import preprocess

pd.set_option("mode.chained_assignment", None)

log = structlog.get_logger(__name__)

DEFAULT_PREVALENCE = 0.2
DEFAULT_MEAN_DISTANCE = 0.04


class Apriori:
    """Computes statistics of pairwise distances."""

    def __init__(
        self,
        threads: int = 1,
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
        self.silent = silent
        self.paired = paired
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
