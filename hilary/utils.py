"Code to process data and fit model parameters."
from __future__ import annotations

from multiprocessing import Pool, cpu_count
from typing import Callable

import numpy as np
import pandas as pd
import structlog
from scipy.special import binom
from textdistance import hamming
from tqdm import tqdm

log = structlog.get_logger(__name__)


def applyParallel(
    dfGrouped: list,
    func: Callable,
    cpuCount: int = cpu_count(),
    silent=False,
) -> pd.DataFrame:
    """Parallely runs func on each group of dfGrouped.

    Args:
        dfGrouped (list): Func runs parallely on each element of the list dfGrouped
        func (Callable): Function to run on dfGrouped
        cpuCount (int, optional): Number of cpus to use. Defaults cpu_count() the maximum possible.
        silent (bool): if true do not show progress bars.

    Returns:
        pd.Dataframe: Dataframe concatenating output of func on each group.
    """
    with Pool(cpuCount) as p:
        ret_list = list(
            tqdm(p.imap(func, dfGrouped), total=len(dfGrouped), disable=silent),
        )
    if len(ret_list) == 0:
        return pd.DataFrame()
    return pd.concat(ret_list)


def count_mutations(args: tuple[int, pd.DataFrame]):
    """Compute & return Return mutation counts column for a given dataframe.

    Args:
        args (tuple[int, pd.DataFrame]): _,dataframe of sequences

    Returns:
        pd.Dataframe: Dataframe with mutation counts.
    """
    _, df = args
    return df[["alt_sequence_alignment", "alt_germline_alignment"]].apply(
        lambda x: hamming(*x),
        axis=1,
    )


def preprocess(
    dataframe: pd.DataFrame,
    max_mutation_count: int = 60,
    lengths: np.ndarray = np.arange(15, 81 + 3, 3).astype(int),
    silent: bool = False,
) -> pd.DataFrame:
    """Processes input dataframe.
    Drop nulls, filters for cdr3 length & mutation counts.

    Args:
        dataframe (pd.DataFrame): Input dataframe of sequences.
        max_mutation_count (int, optional): Remove sequences with higher mutation count. \
            Defaults to 60.
        lengths (np.ndarray, optional): Remove sequences with CDR3 length not in lengths. \
            Defaults to np.arange(15, 81 + 3, 3).astype(int).
        silent (bool, optional): Do not show progress bar if true. Defaults to False.

    Returns:
        pd.Dataframe: processed dataframe.
    """
    df = dataframe.copy()[
        [
            "sequence_id",
            "v_call",
            "j_call",
            "junction",
            "v_sequence_alignment",
            "j_sequence_alignment",
            "v_germline_alignment",
            "j_germline_alignment",
        ]
    ]
    df = df.dropna()
    usecols = [
        "sequence_id",
        "v_gene",
        "j_gene",
        "cdr3_length",
        "cdr3",
        "alt_sequence_alignment",
        "alt_germline_alignment",
        "mutation_count",
    ]
    if "v_gene" not in df.columns:
        df[["v_gene", "_"]] = df["v_call"].str.split("*", expand=True)
    if "j_gene" not in df.columns:
        df[["j_gene", "_"]] = df["j_call"].str.split("*", expand=True)
    if "cdr3" not in df.columns:
        df["cdr3"] = df["junction"].str[3:-3]
    df["cdr3_length"] = df["cdr3"].str.len()

    df["alt_sequence_alignment"] = (
        df["v_sequence_alignment"] + df["j_sequence_alignment"]
    )
    df["alt_germline_alignment"] = (
        df["v_germline_alignment"] + df["j_germline_alignment"]
    )

    df["mutation_count"] = applyParallel(
        df.groupby(["v_gene", "j_gene", "cdr3_length"]),
        count_mutations,
        silent=silent,
    )
    log.debug(
        "Filtering sequences",
        criteria_one="CDR3 length not multiple of three.",
        criteria_two="CDR3 length not in [15,81].",
        criteria_three="More than 60 sequence mutations from germline.",
        criteria_four="With a null column value.",
    )
    return df.query(
        "cdr3_length in @lengths and mutation_count <= @max_mutation_count",
    )[usecols].astype({"mutation_count": int, "cdr3_length": int})


def create_classes(df: pd.DataFrame) -> pd.Dataframe:
    """Create VJl classes.

    Args:
        df (pd.DataFrame): Processed dataframe of sequences.

    Returns:
        pd.DataFrame: Dataframe with classes.
    """
    classes = (
        df.groupby(["v_gene", "j_gene", "cdr3_length"])
        .size()
        .to_frame("sequence_count")
    ).reset_index()
    classes["pair_count"] = (
        classes["sequence_count"].apply(lambda x: binom(x, 2)).astype(int)
    )
    l_classes = (
        classes.groupby("cdr3_length")[["sequence_count", "pair_count"]]
        .sum()
        .reset_index()
    )
    l_classes["v_gene"] = "None"
    l_classes["j_gene"] = "None"
    classes = pd.concat([classes, l_classes], ignore_index=True).sort_values(
        "sequence_count",
        ascending=False,
    )
    classes["class_id"] = range(1, len(classes) + 1)
    classes.reset_index(drop=True, inplace=True)
    return classes