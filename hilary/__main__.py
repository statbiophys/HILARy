from __future__ import annotations

import logging
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import structlog
import typer
from scipy.special import binom

from hilary.apriori import Apriori
from hilary.inference import HILARy
from hilary.utils import pairwise_evaluation, read_input, save_dataframe

app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_path: Path = typer.Argument(
        ...,
        help="Path of the excel file to infer lineages.",
        show_default=False,
    ),
    kappa_file: Path = typer.Option(
        None,
        "--kappa-file",
        help="Path of the kappa chain file, hilary will automatically use its paired option.",
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
        1,
        "--precision",
        "-p",
        help="Choose desired precision.",
    ),
    sensitivity: float = typer.Option(
        1,
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
        help="Where to save the result files. By default it will be saved in a 'result/' folder.",
    ),
    config: Path = typer.Option(
        None,
        "--config",
        help="""Configuration file for column names. File should be a json with keys as your \
            data's column names and values as hilary's required column names.""",
    ),
    override: bool = typer.Option(
        False,
        "--override",
        help="Override existing results.",
    ),
    use_json: bool = typer.Option(
        False,
        "--json/--text",
        help="Print logs as JSON or text.",
    ),
    cdr3only: bool = typer.Option(
        False,
        "--cdr3only",
        help="Use only the cdr3 method.",
    ),
    xy_complete: bool = typer.Option(
        False, "--xy-complete", help="Run xy method on all VJL classes."
    ),
) -> None:
    """Infer lineages from data_path excel file."""
    if result_folder is None:
        result_folder = data_path.parents[0] / Path("hilary_results/")
    result_folder.mkdir(parents=True, exist_ok=True)
    debug_folder = result_folder / Path("debug/")
    debug_folder.mkdir(parents=True, exist_ok=True)

    output_path = result_folder / Path(f"inferred_{data_path.name}")
    if output_path.exists() and not override:
        raise ValueError(
            f"{output_path.as_posix()} already exists, use override parameter to replace the file.",
        )

    if verbose >= 2:  # noqa: PLR2004
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
    log.info(
        "📖 READING DATA ",
        data_path=data_path.as_posix(),
    )
    dataframe = read_input(input_path=data_path, config=config)
    dataframe["sequence_id"] = dataframe["sequence_id"].str.strip("-igh")
    dataframe.set_index("sequence_id")

    if kappa_file:
        log.info("USING PAIRED OPTION.")
        dataframe_kappa = read_input(input_path=kappa_file, config=config)
        dataframe_kappa["sequence_id"] = dataframe_kappa["sequence_id"].str.strip(
            "-igk"
        )
        dataframe_kappa.set_index("sequence_id")
        lengths = np.arange(57, 144 + 3, 3).astype(int)
        xy_threshold = 4
    else:
        dataframe_kappa = None
        lengths = np.arange(15, 81 + 3, 3).astype(int)
        xy_threshold = 0

    if logging_level == logging.DEBUG:
        input_path = debug_folder / Path(f"input_{data_path.name}")
        log.debug(
            "Saving input dataframe.",
            path=input_path.as_posix(),
        )
        save_dataframe(dataframe=dataframe, save_path=input_path)
    apriori = Apriori(
        dataframe,
        dataframe_kappa=dataframe_kappa,
        lengths=lengths,
        threads=threads,
        precision=precision,
        sensitivity=sensitivity,
        model=model,
        silent=silent,
    )
    if logging_level == logging.DEBUG:
        preprocessed_path = debug_folder / Path(f"preprocessed_input_{data_path.name}")
        log.debug(
            "Saving dataframe used by Hilary after preprocessing.",
            path=preprocessed_path.as_posix(),
        )
        save_dataframe(dataframe=apriori.df, save_path=preprocessed_path)

        classes_path = debug_folder / Path(f"classes_{data_path.name}")
        log.debug(
            "Saving classes used by Hilary after preprocessing.",
            path=classes_path.as_posix(),
        )
        save_dataframe(dataframe=apriori.classes, save_path=classes_path)

    log.info("⏳ COMPUTING HISTOGRAMS ⏳.")
    apriori.get_histograms()

    if logging_level == logging.DEBUG:
        histogram_path = debug_folder / Path(f"histograms_{data_path.name}")
        log.debug("Saving histograms used by Hilary.", path=histogram_path.as_posix())
        save_dataframe(apriori.histograms, histogram_path)

    log.info("⏳ COMPUTING PARAMETERS ⏳.")
    apriori.get_parameters()

    log.info("⏳ COMPUTING THRESHOLDS ⏳.")
    apriori.get_thresholds()

    if logging_level == logging.DEBUG:
        parameters_path = debug_folder / Path(f"parameters_{data_path.name}")
        log.debug(
            "Saving all parameters inferred by Hilary.",
            path=parameters_path.as_posix(),
        )
        save_dataframe(apriori.classes, parameters_path)

    hilary = HILARy(apriori, xy_threshold=xy_threshold)
    log.info("⏳ COMPUTING PRECISE AND SENSITIVE CLUSTERS ⏳.")
    hilary.compute_prec_sens_clusters()

    if cdr3only:
        log.info("RETURNING CDR3 PRECISE CLUSTERS.")
        dataframe["family"] = hilary.df["precise_cluster"]
        if dataframe_kappa is not None:
            dataframe_kappa["family"] = hilary.df["precise_cluster"]

    else:
        log.info("⏳ INFERRING FAMILIES WITH FULL XY METHOD⏳.")
        hilary.infer(xy_complete=xy_complete)

        if logging_level == logging.DEBUG:
            output_path = debug_folder / Path(f"preprocessed_output_{data_path.name}")
            log.debug(
                "Saving dataframe inferred by Hilary.",
                path=output_path.as_posix(),
            )
            save_dataframe(hilary.df, save_path=output_path)

        dataframe["family"] = hilary.df["family"]
        dataframe["cdr3_only_method_clustering"] = hilary.df["precise_cluster"]
        dataframe["sequence_id"] = dataframe["sequence_id"] + "-igh"
        output_path = result_folder / Path(f"inferred_{data_path.name}")

        log.info("💾 SAVING RESULTS ", output_path=output_path.as_posix())
        save_dataframe(dataframe=dataframe, save_path=output_path)

        if dataframe_kappa is not None:
            dataframe_kappa["family"] = hilary.df["family"]
            dataframe_kappa["cdr3_only_method_clustering"] = hilary.df[
                "precise_cluster"
            ]
            dataframe_kappa["sequence_id"] = dataframe_kappa["sequence_id"] + "-igk"
            output_path_kappa = result_folder / Path(f"inferred_{kappa_file.name}")

            log.info(
                "💾 SAVING RESULTS FOR KAPPA FILE",
                output_path=output_path_kappa.as_posix(),
            )
            save_dataframe(dataframe=dataframe_kappa, save_path=output_path_kappa)

        if logging_level == logging.DEBUG and "clone_id" in dataframe.columns:
            precision_full, sensitivity_full = pairwise_evaluation(
                df=dataframe, partition="family"
            )
            hilary.df["clone_id"] = dataframe["clone_id"]
            precision_cdr3, sensitivity_cdr3 = pairwise_evaluation(
                df=hilary.df, partition="precise_cluster"
            )
            log.debug(
                "Evaluating Hilary's performance on ground truth column 'clone_id'.",
                precision_full_method=precision_full,
                sensitivity_full_method=sensitivity_full,
                precision_cdr3=precision_cdr3,
                sensitivity_cdr3=sensitivity_cdr3,
            )


if __name__ == "__main__":
    app()
