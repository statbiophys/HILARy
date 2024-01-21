from __future__ import annotations

import logging
from multiprocessing import cpu_count
from pathlib import Path

import structlog
import typer
from scipy.special import binom

from hilary.apriori import Apriori
from hilary.inference import HILARy
from hilary.utils import read_input, save_dataframe

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
    """Infer lineages from data_path excel file."""
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
        "📖 READING DATA ",
        data_path=data_path.as_posix(),
    )
    dataframe = read_input(input_path=data_path, config=config)

    log.debug("Displaying dataframe columns.", columns=dataframe.columns)
    if logging_level == logging.DEBUG:
        input_path = debug_folder / Path(f"input_{data_path.name}")
        log.debug(
            "Saving input dataframe.",
            path=input_path.as_posix(),
        )
        save_dataframe(dataframe=dataframe, save_path=input_path)

    apriori = Apriori(
        dataframe,
        threads=threads,
        precision=precision,
        sensitivity=sensitivity,
        model=model,
        silent=silent,
    )
    if logging_level == logging.DEBUG:
        preprocessed_path = debug_folder / Path(f"preprocessed_input_{data_path.name}")
        log.debug(
            "Saving dataframe after preprocessing.",
            path=preprocessed_path.as_posix(),
        )
        classes_path = debug_folder / Path(f"classes_{data_path.name}")
        log.debug(
            "Saving classes after preprocessing.",
            path=classes_path.as_posix(),
        )
        save_dataframe(dataframe=dataframe, save_path=preprocessed_path)
        save_dataframe(dataframe=dataframe, save_path=classes_path)

    log.info("⏳ COMPUTING HISTOGRAMS ⏳.")
    apriori.get_histograms()
    if logging_level == logging.DEBUG:
        histogram_path = debug_folder / Path(f"histograms_{data_path.name}")
        log.debug("Saving histograms.", path=histogram_path.as_posix())
        save_dataframe(apriori.histograms, histogram_path)

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
        output_path = debug_folder / Path(f"preprocessed_output_{data_path.name}")
        log.debug(
            "Saving dataframe inferred by Hilary.",
            path=output_path.as_posix(),
        )
        save_dataframe(hilary.df, save_path=output_path)
        parameters_path = debug_folder / Path(f"parameters_{data_path.name}")
        log.debug(
            "Saving all parameters inferred by Hilary.",
            path=parameters_path.as_posix(),
        )
        save_dataframe(hilary.classes, parameters_path)

    if "to_resolve" in hilary.df.columns:
        mut_met_frac = (
            binom(
                hilary.df.query("to_resolve == True")
                .groupby(hilary.group + ["sensitive_cluster"])
                .size(),
                2,
            ).sum()
            / binom(
                hilary.df.groupby(["v_gene", "j_gene", "cdr3_length"]).size(), 2
            ).sum()
        )
        log.info(
            "Fraction of pairs that go through mutation method.", fraction=mut_met_frac
        )

    mask = ~dataframe.index.isin(hilary.df.index)
    dataframe["clone_id"] = hilary.df["clone_id"]
    dataframe.loc[mask, "clone_id"] = 0
    output_path = result_folder / Path(f"inferred_{data_path.name}")

    log.info("💾 SAVING RESULTS ", output_path=output_path.as_posix())
    save_dataframe(dataframe, save_path=output_path)


if __name__ == "__main__":
    app()
