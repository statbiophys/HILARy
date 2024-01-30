"Code to process data and fit model parameters."
from __future__ import annotations

import json
from itertools import combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable

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
    silent: bool = False,
) -> pd.DataFrame:
    """Processes input dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe of sequences.
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
    return df[usecols].astype({"cdr3_length": int})


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


def save_dataframe(dataframe: pd.DataFrame, save_path: Path):
    """Save dataframe depending on suffix.

    Args:
        dataframe (pd.DataFrame): Dataframe to save.
        save_path (Path): Where to save the dataframe.

    Raises:
        ValueError: save_path suffix not supported.
    """
    suffix = save_path.suffix
    if suffix == ".xlsx":
        dataframe.to_excel(save_path)
    elif suffix == ".tsv":
        dataframe.to_csv(save_path, sep="\t")
    elif suffix == ".csv":
        dataframe.to_csv(save_path)
    else:
        raise ValueError(f"Format {suffix} not supported.")


def read_input(input_path: Path, config: Path | None = None) -> pd.DataFrame:
    """Read input file.

    Args:
        input_path (Path):Path of input file.

    Raises:
        ValueError: Format of input file is not supported.

    Returns:
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
    else:
        raise ValueError(
            f"Format {suffix} not supported. Extensions supported are tsv, xlsx.",
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

    Returns:
        (precision,sensitivity)
    """
    if "clone_id" not in df.columns:
        log.debug("Clone id not a column value.")
        return None, None
    TP = 0
    P = binom(df.groupby(["clone_id"]).size(), 2).sum()
    TP_FP = binom(df.groupby([partition]).size(), 2).sum()
    for i, family in tqdm(df.groupby(["clone_id"]), disable=True):
        for r1, r2 in combinations(family[partition], 2):
            if r1 == r2:
                TP += 1

    if TP_FP == 0 and P > 0:
        return 0, 1.0
    elif P == 0:
        return None, None
    return TP / TP_FP, TP / P  # precision, sensitivity
