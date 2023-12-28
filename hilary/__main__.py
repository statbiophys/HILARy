from __future__ import annotations

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
        100000,
        "--nmin",
        help="Infer prevalence and mu on classes of size larger than nmin. \
            Mean prevalence is assigned to lower than nmin classes.",
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
    """
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
    dataframe = pd.read_excel(data_path)
    log.debug("Displaying dataframe columns.", columns=dataframe.columns)
    if logging_level == logging.DEBUG:
        log.debug("Saving input dataframe.", path="debug_input.xlsx")
        dataframe.to_excel("debug_input.xlsx")
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
        log.debug(
            "Saving dataframe after preprocessing.",
            path="debug_input_preprocessed.xlsx",
        )
        apriori.df.to_excel("debug_input_preprocessed.xlsx")
    log.info("⏳ COMPUTING HISTOGRAMS ⏳.")
    apriori.get_histograms()
    log.info("⏳ COMPUTING PARAMETERS ⏳.")
    apriori.get_parameters()
    log.info("⏳ COMPUTING THRESHOLDS ⏳.")
    apriori.get_thresholds()
    hilary = HILARy(apriori)
    log.info("⏳ COMPUTING PRECISE AND SENSITIVE CLUSTERS ⏳.")
    hilary.compute_prec_sens_clusters()
    small_to_do, large_to_do = hilary.to_do()
    log.info("⏳ INFERRING FAMILIES ⏳.")
    hilary.infer(small_to_do, large_to_do)
    if logging_level == logging.DEBUG:
        log.debug(
            "Saving dataframe inferred by Hilary.",
            path="debug_hilary_output.xlsx",
        )
        hilary.df.to_excel("debug_hilary_output.xlsx")
    mask = ~dataframe.index.isin(hilary.df.index)
    dataframe["family"] = hilary.df["family"]
    dataframe.loc[mask, "family"] = 0
    output_path = data_path.parents[0] / Path(f"inferred_{data_path.name}").with_suffix(
        ".xlsx",
    )
    log.info("💾 SAVING RESULTS 💾.", output_path=output_path.as_posix())
    dataframe.to_excel(
        data_path.parents[0] / Path(f"inferred_{data_path.name}").with_suffix(".xlsx"),
    )


if __name__ == "__main__":
    """Run app."""
    app()
