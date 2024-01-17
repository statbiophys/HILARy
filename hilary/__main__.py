from __future__ import annotations

import json
import logging
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd
import structlog
import typer

from hilary.apriori import Apriori
from hilary.inference import HILARy

app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_path: Path = typer.Argument(
        ...,
        help="Path of the excel file to infer lineages.",
        show_default=False,
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Set logging verbosity level.",
    ),
    threads: int = typer.Option(
        cpu_count(),
        "--threads",
        "-t",
        help="Choose number of cpus on which to run code.",
    ),
    precision: float = typer.Option(
        0.99,
        "--precision",
        "-p",
        help="Choose desired precision.",
    ),
    sensitivity: float = typer.Option(
        0.9,
        "--sensitivity",
        "-s",
        help="Choose desired sensitivity.",
    ),
    nmin: int = typer.Option(
        1000,
        "--nmin",
        help="""Infer prevalence and mu on classes of size larger than nmin. \
            Mean prevalence is assigned to lower than nmin classes.""",
    ),
    model: int = typer.Option(
        326713,
        "--model",
        "-m",
        help="Model name to infer Null distribution.",
    ),
    silent: bool = typer.Option(
        False,
        "--silent",
        help="Do not show progress bars if used.",
    ),
    result_folder: Path = typer.Option(
        None,
        "--result-folder",
        help="""Where to save the result files. By default it will be saved in a 'result/'
            folder in input data's parent directory.""",
    ),
    config: Path = typer.Option(
        None,
        "--config",
        help="""Configuration file for column names. File should be a json with keys as your data's
            column names and values as hilary's required column names.""",
    ),
) -> None:
    """Infer lineages from data_path excel file.

    Args:
        data_path (Path): Path of the excel file to infer lineages.
        verbose (int, optional): Set logging verbosity level, defaults to 0.
        threads (int, optional): Number of cpus on which to run code, defaults to cpu_count().
        precision (float, optional): Desired precision, defaults to 0.99.
        sensitivity (float, optional): Desired sensitivity, defaults to 0.9.
        nmin (int, optional): Infer prevalence and mu on classes of size > nmin, defaults to 100000.
        model (int, optional): Model name to infer Null distribution, defaults to 326713.
        silent (bool,optional): Do not show progress bars if used.
        result_folder (Path): Where to save the result files. By default it will be saved in a
            'result/' folder in input data's parent directory.
        config (Path): Configuration file for column names. File should be a json with keys
            as your data's column names and values as hilary's required column names.
    """
    if result_folder is None:
        result_folder = data_path.parents[0] / Path("hilary_results/")
    result_folder.mkdir(parents=True, exist_ok=True)
    debug_folder = result_folder / Path("debug/")
    debug_folder.mkdir(parents=True, exist_ok=True)

    if verbose >= 2:  # noqa: PLR2004
        logging_level = logging.DEBUG
    elif verbose == 1:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARNING
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
    log.info(
        "📖 READING DATA 📖.",
        data_path=data_path.as_posix(),
    )
    suffix = data_path.suffix
    if suffix == ".xlsx":
        dataframe = pd.read_excel(data_path)
    elif suffix == ".tsv":
        dataframe = pd.read_csv(
            data_path,
            sep="\t",
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

    log.debug("Displaying dataframe columns.", columns=dataframe.columns)
    if logging_level == logging.DEBUG:
        input_path = debug_folder / Path(f"input_{data_path.name}").with_suffix(".xlsx")
        log.debug(
            "Saving input dataframe.",
            path=input_path.as_posix(),
        )
        dataframe.to_excel(input_path.as_posix())
    apriori = Apriori(
        dataframe,
        threads=threads,
        precision=precision,
        sensitivity=sensitivity,
        nmin=nmin,
        model=model,
        silent=silent,
    )
    if logging_level == logging.DEBUG:
        preprocessed_path = debug_folder / Path(
            f"preprocessed_input_{data_path.name}",
        ).with_suffix(".xlsx")
        log.debug(
            "Saving dataframe after preprocessing.",
            path=preprocessed_path.as_posix(),
        )
        classes_path = debug_folder / Path(
            f"classes_{data_path.name}",
        ).with_suffix(".xlsx")
        log.debug(
            "Saving classes after preprocessing.",
            path=classes_path.as_posix(),
        )
        apriori.df.to_excel(preprocessed_path.as_posix())
        apriori.classes.to_excel(classes_path.as_posix())
    log.info("⏳ COMPUTING HISTOGRAMS ⏳.")
    apriori.get_histograms()
    log.info("⏳ COMPUTING PARAMETERS ⏳.")
    apriori.get_parameters()
    log.info("⏳ COMPUTING THRESHOLDS ⏳.")
    apriori.get_thresholds()
    hilary = HILARy(apriori)
    log.info("⏳ COMPUTING PRECISE AND SENSITIVE CLUSTERS ⏳.")
    hilary.compute_prec_sens_clusters()
    log.info("⏳ INFERRING FAMILIES ⏳.")
    hilary.infer()
    if logging_level == logging.DEBUG:
        output_path = debug_folder / Path(
            f"preprocessed_output_{data_path.name}",
        ).with_suffix(".xlsx")
        log.debug(
            "Saving dataframe inferred by Hilary.",
            path=output_path.as_posix(),
        )
        hilary.df.to_excel(output_path)
        parameters_path = debug_folder / Path(
            f"parameters_{data_path.name}",
        ).with_suffix(".xlsx")
        log.debug(
            "Saving all parameters inferred by Hilary.",
            path=parameters_path.as_posix(),
        )
        hilary.classes.to_excel(parameters_path)
    mask = ~dataframe.index.isin(hilary.df.index)
    dataframe["family"] = hilary.df["family"]
    dataframe.loc[mask, "family"] = 0
    output_path = result_folder / Path(f"inferred_{data_path.name}").with_suffix(
        ".xlsx",
    )
    log.info("💾 SAVING RESULTS 💾.", output_path=output_path.as_posix())
    dataframe.to_excel(output_path)


if __name__ == "__main__":
    app()
