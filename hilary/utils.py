"""Code to process data and fit model parameters."""

from __future__ import annotations

import json
import logging
from functools import partial
from itertools import combinations
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
import structlog
from scipy.special import binom
from textdistance import hamming
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

log = structlog.get_logger(__name__)

VERBOSE_DEBUG = 2
VERBOSE_INFO = 1

def group_mutations(args:tuple[int,pd.DataFrame])->pd.DataFrame:
    """Get list of mutations for a given VJL class.

    Args:
        args (tuple[int,pd.DataFrame]): (class_id, dataframe for that class).

    Returns
    -------
        pd.DataFrame: CLass dataframe with mutation count list.
    """
    _, df = args
    v_gene, j_gene, cdr3_length, _, prevalence, class_id, null_model, alignment_length = df.iloc[0]
    mutations = df["mutation_count"].values
    return pd.DataFrame(
        [
            v_gene,
            j_gene,
            cdr3_length,
            prevalence,
            mutations,
            alignment_length,
            class_id,
            null_model,
        ]
    ).T


def cdf_to_pmf(cdf_values):
    """
    Convert a cumulative distribution function (CDF) to a probability mass function (PMF).

    Parameters
    ----------
    cdf_values (array-like): A list or array of CDF values. The CDF must start at 0 (implicitly)
                            and end at 1. The values must be non-decreasing.

    Returns
    -------
    array-like: An array of PMF values, which are the differences between consecutive CDF values.
    """
    if not np.all(np.diff(cdf_values) >= 0):
        msg = "CDF values must be non-decreasing."
        raise ValueError(msg)

    # PMF is the difference between consecutive CDF values
    return np.diff(cdf_values, prepend=[0])  # Prepend 0 for the first element


def return_cdf(cdf_path: Path, v_gene: str, j_gene: str, cdr3_length: int) -> pd.DataFrame:
    """Return cdf distribution given VJl class.

    Args:
        cdf_path (Path): Where to get the distributions.
        v_gene (str): V gene
        j_gene (str): J gene
        cdr3_length (int): CDR3 length

    Returns
    -------
        pd.DataFrame: _description_
    """
    if cdr3_length % 3 != 0:
        cdr3_length = round(cdr3_length / 3) * 3 # round to the nearest multiple of 3
    return pd.read_parquet(
        cdf_path,
        filters=[
            ("v_gene", "==", v_gene),
            ("j_gene", "==", j_gene),
            ("cdr3_length", "==", cdr3_length),
        ],
    )

def chunked_func(df_list:list[pd.DataFrame], func: Callable) -> pd.DataFrame:
    """Apply a function to each element in a list and concatenates the results.

    Applies a given function `func` to each element `g` in the iterable `x` (usually a list
    of grouped DataFrames), and then concatenates the results into a single DataFrame.

    Args:
        df_list (list[pd.DataFrame]): An list of DataFrames.
        func (Callable): A function that takes one element of `df_list` and returns a DataFrame.

    Returns
    -------
        pd.DataFrame: A concatenated DataFrame.
    """
    return pd.concat([func(df) for df in df_list])

def apply_chunked_parallel(
    df_grouped: Iterable,
    func: Callable,
    cpu_count: int = 1,
    *,
    silent=False,
    isint=False,
) -> pd.DataFrame:
    """Parallely runs func on each group of df_grouped.

    Args:
        df_grouped (Iterable): Func runs parallely on each element of the list df_grouped
        func (Callable): Function to run on df_grouped
        cpu_count (int, optional): Number of cpus to use. Defaults to 1.
        silent (bool): if true do not show progress bars.
        isint (bool): if true return list of pd.Dataframes instead of concatenated pd.Dataframe.

    Returns
    -------
        pd.Dataframe: Dataframe concatenating output of func on each group.
    """
    if not isinstance(df_grouped, list):
        df_grouped = list(df_grouped)
    num_chunks = cpu_count * 10
    chunk_size = np.ceil(len(df_grouped) / num_chunks).astype(int)
    df_grouped_chunks = [
        df_grouped[i : i + chunk_size] for i in range(0, len(df_grouped), chunk_size)
    ]
    chunked_func_with_func = partial(chunked_func, func=func)
    return apply_parallel(
            df_grouped_chunks,
            chunked_func_with_func,
            cpu_count=cpu_count,
            silent=silent,
            isint=isint,
        )

def apply_parallel(
    df_grouped: list,
    func: Callable,
    cpu_count: int = 1,
    *,
    silent=False,
    isint=False,
) -> pd.DataFrame:
    """Parallely runs func on each group of df_grouped.

    Args:
        df_grouped (list): Func runs parallely on each element of the list df_grouped
        func (Callable): Function to run on df_grouped
        cpu_count (int, optional): Number of cpus to use. Defaults to 1.
        silent (bool): if true do not show progress bars.

    Returns
    -------
        pd.Dataframe: Dataframe concatenating output of func on each group.
    """
    with Pool(cpu_count) as p:
        ret_list = list(
            tqdm(p.imap(func, df_grouped), total=len(df_grouped), disable=silent),
        )
    if isint:
        return ret_list
    if not ret_list:
        return pd.DataFrame()
    return pd.concat(ret_list)


def count_mutations(args: tuple[int, pd.DataFrame]):
    """Compute & return Return mutation counts column for a given dataframe.

    Args:
        args (tuple[int, pd.DataFrame]): _,dataframe of sequences

    Returns
    -------
        pd.Dataframe: Dataframe with mutation counts.
    """
    _, df = args
    return df[["alt_sequence_alignment", "alt_germline_alignment"]].apply(
        lambda x: hamming(*x),
        axis=1,
    )


def preprocess(
    dataframe: pd.DataFrame,
    *,
    silent: bool = False,
    threads: int = 1,
) -> pd.DataFrame:
    """Process input dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe of sequences.
        silent (bool, optional): Do not show progress bar if true. Defaults to False.
        threads (int, optional): Number of cpus on which to run code, defaults to 1.

    Returns
    -------
        pd.Dataframe: processed dataframe.
    """
    df = dataframe.copy()
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
        df.dropna(subset=["v_call"], inplace=True)
        df[["v_gene", "_"]] = df["v_call"].str.split("*", expand=True, n=1)
    if "j_gene" not in df.columns:
        df.dropna(subset=["j_call"], inplace=True)
        df[["j_gene", "_"]] = df["j_call"].str.split("*", expand=True, n=1)
    if "cdr3" not in df.columns:
        df.dropna(subset=["junction"], inplace=True)
        df["cdr3"] = df["junction"].str[3:-3]
    if "cdr3_length" not in df.columns:
        df["cdr3_length"] = df["cdr3"].str.len()
    if "alt_sequence_alignment" not in df.columns:
        df.dropna(subset=["v_sequence_alignment", "j_sequence_alignment"], inplace=True)
        df["alt_sequence_alignment"] = df["v_sequence_alignment"] + df["j_sequence_alignment"]
    if "alt_germline_alignment" not in df.columns:
        df.dropna(subset=["v_germline_alignment", "j_germline_alignment"], inplace=True)
        df["alt_germline_alignment"] = df["v_germline_alignment"] + df["j_germline_alignment"]
    if "mutation_count" not in df.columns:
        df["mutation_count"] = apply_parallel(
            df.groupby(["v_gene", "j_gene", "cdr3_length"]),
            count_mutations,
            silent=silent,
            cpu_count=threads,
        )
    return df[usecols].dropna().astype({"cdr3_length": int})

def create_classes(df: pd.DataFrame) -> pd.Dataframe:
    """Create VJl classes.

    Args:
        df (pd.DataFrame): Processed dataframe of sequences.

    Returns
    -------
        pd.DataFrame: Dataframe with classes.
    """
    df["cdr3_length"] = df.cdr3_length.astype(str)
    classes = (
        df.groupby(["v_gene", "j_gene", "cdr3_length"]).size().to_frame("sequence_count")
    ).reset_index()
    classes["pair_count"] = classes["sequence_count"].apply(lambda x: binom(x, 2)).astype(int)
    l_classes = classes.groupby("cdr3_length")[["sequence_count", "pair_count"]].sum().reset_index()
    l_classes["v_gene"] = "None"
    l_classes["j_gene"] = "None"
    jl_classes = (
        classes.groupby(["j_gene", "cdr3_length"])[["sequence_count", "pair_count"]]
        .sum()
        .reset_index()
    )
    jl_classes["v_gene"] = "None"
    classes = pd.concat([classes, l_classes, jl_classes], ignore_index=True).sort_values(
        "sequence_count",
        ascending=False,
    )
    classes["class_id"] = range(1, len(classes) + 1)
    classes.reset_index(drop=True, inplace=True)
    return classes


def save_dataframe(dataframe: pd.DataFrame, save_path: Path) -> None:
    """Save dataframe depending on suffix.

    Args:
        dataframe (pd.DataFrame): Dataframe to save.
        save_path (Path): Where to save the dataframe.

    Raises
    ------
        ValueError: save_path suffix not supported.
    """
    suffix = save_path.suffix
    if suffix == ".xlsx":
        dataframe.to_excel(save_path)
    elif suffix == ".tsv":
        dataframe.to_csv(save_path, sep="\t")
    elif suffix == ".csv":
        dataframe.to_csv(save_path)
    elif suffix == ".gz":
        if ".csv" in save_path.suffixes:
            dataframe.to_csv(
                save_path.with_suffix(""),
            )
    else:
        msg = f"Format {suffix} not supported."
        raise ValueError(msg)


def read_input(input_path: Path, config: Path | None = None) -> pd.DataFrame:
    """Read input file.

    Args:
        input_path (Path):Path of input file.
        config (Path): Json configuration file to change column names of your custom sequence file.

    Raises
    ------
        ValueError: Format of input file is not supported.

    Returns
    -------
        pd.DataFrame: Pandas dataframe.
    """
    suffix = input_path.suffix
    dataframe: pd.DataFrame
    if suffix == ".xlsx":
        dataframe = pd.read_excel(input_path)
    elif suffix == ".tsv":
        dataframe = pd.read_csv(
            input_path,
            sep="\t",
        )
    elif suffix == ".csv":
        dataframe = pd.read_csv(
            input_path,
        )
    elif suffix == ".gz":
        if ".csv" in input_path.suffixes:
            dataframe = pd.read_csv(
                input_path,
            )
    else:
        msg = f"Format {suffix} not supported. Extensions supported are tsv, xlsx, csv, csv.gz"
        raise ValueError(
            msg,
        )
    if config:
        with config.open(encoding="utf-8") as user_file:
            column_dict = json.load(user_file)
            for key in column_dict:
                dataframe[column_dict[key]] = dataframe[key]
    return dataframe


def pairwise_evaluation(
    df: pd.DataFrame, partition: str, truth: str = "ground_truth"
) -> tuple[float, float]:
    """Evaluate performance if ground truth present in dataframe.

    Args:
        df (pd.DataFrame): dataframe to evaluate
        partition (str): name of column corresponding to inferred partition.

    Returns
    -------
        (precision,sensitivity)
    """
    tp = 0
    pos = binom(df.groupby([truth]).size(), 2).sum()
    tp_fp = binom(df.groupby([partition]).size(), 2).sum()
    for _, family in tqdm(df.groupby([truth]), disable=True):
        for r1, r2 in combinations(family[partition], 2):
            if r1 == r2:
                tp += 1
    if (not tp_fp and pos > 0) or not pos:
        return np.nan, np.nan
    precision = tp / tp_fp
    sensitivity = tp / pos
    return precision, sensitivity


def p_required(prevalence:float, pi:float=0.99)->float:
    """Get the fallout from prevalence and desired precision.

    Args:
        prevalence (float): Prevalence.
        pi (float, optional): Precision. Defaults to 0.99.

    Returns
    -------
        Fallout (p): Fallout fp/(fp+tn).
    """
    return prevalence / (1 + 1e-5 - prevalence) * (1 - pi) / pi


def get_logger(verbose:int, *, use_json:bool)->Any:
    """Return logger.

    Args:
        verbose (int): Level of verbosity. 2 is DEBUG, 1 is INFO, 0 is WARNING.
        use_json (bool): Return the logs as json (useful for processing logs in the future).

    Returns
    -------
        Any: Logger.
    """
    if verbose >= VERBOSE_DEBUG:
        logging_level = logging.DEBUG
    elif verbose == VERBOSE_INFO:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARNING
    if use_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(sort_keys=False)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging_level),
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
    )
    return structlog.get_logger()
