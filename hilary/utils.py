"""Code to process data and fit model parameters."""

from __future__ import annotations

import json
import logging
from itertools import combinations
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
import structlog
from scipy.special import binom
from textdistance import hamming
from tqdm import tqdm

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger(__name__)

# pylint: disable=invalid-name

def select_df(args):
    """
    Selects rows from a DataFrame based on a given selection strength.
    Args:
        args (tuple): A tuple containing:
            - (j_gene, cdr3_length): A tuple of the gene and CDR3 length to filter on.
            - sel1 (DataFrame): The DataFrame to filter.

    Returns
    -------
        DataFrame: A filtered DataFrame where the mean absolute difference between
        the values in sel1 and the corresponding values in sel2 is greater than
        the selection strength.
    """
    selection_strength = 0.01 # find out a way to pass this as an argument
    (_,_),cdfs = args
    v_gene_nans=cdfs.isna().v_gene
    j_gene_nans=cdfs.isna().j_gene
    sel1=cdfs.loc[np.logical_and(~v_gene_nans,~j_gene_nans)]
    sel2=cdfs.loc[np.logical_and(v_gene_nans,~j_gene_nans)]
    select=np.abs(sel1.values[:,3:]-sel2.values[:,3:]).mean(axis=1) > selection_strength
    return sel1[select]

def return_cdf(classes: pd.DataFrame, cdfs: pd.DataFrame, class_id: int, extend: int = 0) -> np.array:
    """Class to extract the right cdf.

    Args:
        classes (pd.DataFrame): classes df (see apriori)
        cdfs (pd.DataFrame): cdfs object (see apriori)
        class_id (int): id of the class to return

    Returns
    -------
        np.array: array encoding the cdf
    """
    cdr3_length = classes.loc[classes.class_id == class_id].cdr3_length.values[0]
    if "v_gene" in cdfs.columns and "j_gene" in cdfs.columns:  ## use vjl cdf
        j = classes.loc[classes.class_id == class_id].j_gene.values[0]
        v = classes.loc[classes.class_id == class_id].v_gene.values[0]
        sel_jl = np.logical_and(cdfs["cdr3_length"] == cdr3_length, cdfs["j_gene"] == j)
        sel = np.logical_and(sel_jl,cdfs["v_gene"] == v)
        sel2 = np.logical_and(sel_jl,cdfs["v_gene"].isna())
        if sum(sel) > 0:
            cdf = cdfs.loc[sel].values[0, 3 : cdr3_length + 3 + extend]  ## found vjl cdf
        elif sum(sel2) > 0:
            cdf = cdfs.loc[sel2].values[0, 3 : cdr3_length + 3 + extend]  ## found jl cdf
        else:
            sel_na=np.logical_and(cdfs["v_gene"].isna(),cdfs["j_gene"].isna())
            cdf = cdfs.loc[np.logical_and(sel_na,cdfs["cdr3_length"] == cdr3_length)].values[0, 3 : cdr3_length + 3 + extend]
    elif "j_gene" in cdfs.columns: ## use jl cdf
        j = classes.loc[classes.class_id == class_id].j_gene.values[0]
        sel = np.logical_and(cdfs["cdr3_length"] == cdr3_length, cdfs["j_gene"] == j)
        if sum(sel) > 0:
            cdf = cdfs.loc[sel].values[0, 2 : cdr3_length + 2 + extend]  ## found jl cdf
        else:
            sel= np.logical_and(cdfs["cdr3_length"] == cdr3_length, cdfs["j_gene"].isna())
            cdf = cdfs.loc[sel].values[0, 2 : cdr3_length + 2 + extend]
    else: ## use l cdf
        cdf = cdfs.loc[cdfs["cdr3_length"] == cdr3_length].values[0, 1 : cdr3_length + 1 + extend]
    return cdf


def applyParallel(
    dfGrouped: list,
    func: Callable,
    cpuCount: int = 1,
    silent=False,
    isint=False,
) -> pd.DataFrame:
    """Parallely runs func on each group of dfGrouped.

    Args:
        dfGrouped (list): Func runs parallely on each element of the list dfGrouped
        func (Callable): Function to run on dfGrouped
        cpuCount (int, optional): Number of cpus to use. Defaults to 1.
        silent (bool): if true do not show progress bars.

    Returns
    -------
        pd.Dataframe: Dataframe concatenating output of func on each group.
    """
    with Pool(cpuCount) as p:
        ret_list = list(
            tqdm(p.imap(func, dfGrouped), total=len(dfGrouped), disable=silent),
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
    df["cdr3_length"] = df["cdr3"].str.len()
    if "alt_sequence_alignment" not in df.columns:
        df.dropna(subset=["v_sequence_alignment", "j_sequence_alignment"], inplace=True)
        df["alt_sequence_alignment"] = df["v_sequence_alignment"] + df["j_sequence_alignment"]

    if "alt_germline_alignment" not in df.columns:
        df.dropna(subset=["v_germline_alignment", "j_germline_alignment"], inplace=True)
        df["alt_germline_alignment"] = df["v_germline_alignment"] + df["j_germline_alignment"]

    df["mutation_count"] = applyParallel(
        df.groupby(["v_gene", "j_gene", "cdr3_length"]),
        count_mutations,
        silent=silent,
        cpuCount=threads,
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
    classes = (
        df.groupby(["v_gene", "j_gene", "cdr3_length"]).size().to_frame("sequence_count")
    ).reset_index()
    classes["pair_count"] = classes["sequence_count"].apply(lambda x: binom(x, 2)).astype(int)
    l_classes = classes.groupby("cdr3_length")[["sequence_count", "pair_count"]].sum().reset_index()
    l_classes["v_gene"] = "None"
    l_classes["j_gene"] = "None"
    jl_classes = classes.groupby(["j_gene", "cdr3_length"])[["sequence_count", "pair_count"]].sum().reset_index()
    jl_classes["v_gene"] = "None"
    classes = pd.concat([classes, l_classes,jl_classes], ignore_index=True).sort_values(
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
        with open(config, encoding="utf-8") as user_file:
            column_dict = json.load(user_file)
            for key in column_dict:
                dataframe[column_dict[key]] = dataframe[key]
    return dataframe


def pairwise_evaluation(df: pd.DataFrame, partition: str):
    """Evaluate performance if ground truth present in dataframe.

    Args:
        df (pd.DataFrame): dataframe to evaluate
        partition (str): name of column corresponding to inferred partition.

    Returns
    -------
        (precision,sensitivity)
    """
    if "ground_truth" not in df.columns:
        log.debug("ground_truth not a column value.")
        return None, None
    TP = 0
    P = binom(df.groupby(["ground_truth"]).size(), 2).sum()
    TP_FP = binom(df.groupby([partition]).size(), 2).sum()
    for _, family in tqdm(df.groupby(["ground_truth"]), disable=True):
        for r1, r2 in combinations(family[partition], 2):
            if r1 == r2:
                TP += 1

    if not TP_FP and P > 0:
        return 0, 1.0
    if not P:
        return None, None
    return TP / TP_FP, TP / P  # precision, sensitivity


def pRequired(rho, pi=0.99):
    return rho / (1 + 1e-5 - rho) * (1 - pi) / pi


def get_logger(verbose, use_json):
    if verbose >= 2:
        logging_level = logging.DEBUG
    elif verbose == 1:
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
    log = structlog.get_logger()
    return log
