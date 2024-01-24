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
        0.95,
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
        False, "--json/--text", help="Print logs as JSON or text."
    ),
    method: str = typer.Option(
        "full",
        "--method",
        help="""Whether to use cdr3 base, xy or full method. Possible values are 'full', \
        'cdr3only', and 'xyonly'.""",
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
    if method not in ["cdr3only", "xyonly", "full"]:
        raise ValueError(
            "Method argument provided unknow. Poissble values are 'cdr3only,xyonly,full'"
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
        save_dataframe(dataframe=apriori.df, save_path=preprocessed_path)

        classes_path = debug_folder / Path(f"classes_{data_path.name}")
        log.debug(
            "Saving classes after preprocessing.",
            path=classes_path.as_posix(),
        )
        save_dataframe(dataframe=apriori.classes, save_path=classes_path)

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

    if logging_level == logging.DEBUG:
        parameters_path = debug_folder / Path(f"parameters_{data_path.name}")
        log.debug(
            "Saving all parameters inferred by Hilary.",
            path=parameters_path.as_posix(),
        )
        save_dataframe(apriori.classes, parameters_path)

    hilary = HILARy(apriori)
    log.info("⏳ COMPUTING PRECISE AND SENSITIVE CLUSTERS ⏳.")
    hilary.compute_prec_sens_clusters()
    log.info("⏳ INFERRING FAMILIES ⏳.")
    hilary.infer(method=method)

    if logging_level == logging.DEBUG:
        output_path = debug_folder / Path(f"preprocessed_output_{data_path.name}")
        log.debug(
            "Saving dataframe inferred by Hilary.",
            path=output_path.as_posix(),
        )
        save_dataframe(hilary.df, save_path=output_path)
        grouped_by_sensitive = (
            hilary.df.groupby(
                hilary.group + ["sensitive_cluster", "clone_id"], group_keys=True
            )[["to_resolve"]]
            .apply(lambda x: x.sum())
            .rename(columns={"to_resolve": "count"})
        )
        grouped_by_sensitive["method"] = grouped_by_sensitive["count"].apply(
            lambda x: "xy" if x > 0 else "cdr3"
        )
        method_summary_path = debug_folder / Path(f"method_summary_{data_path.name}")
        log.debug(
            "Saving method distribution summary.",
            path=method_summary_path.as_posix(),
        )
        save_dataframe(grouped_by_sensitive["method"], save_path=method_summary_path)

    pair_frac = (
        binom(
            hilary.df.query("to_resolve == True")
            .groupby(hilary.group + ["sensitive_cluster"])
            .size(),
            2,
        ).sum()
        / binom(
            hilary.df.groupby(hilary.group + ["sensitive_cluster"]).size(),
            2,
        ).sum()
    )
    log.info(
        "Fraction of pairs that go through xy method.",
        fraction=pair_frac,
    )

    dataframe["clone_id"] = hilary.df["clone_id"]
    dataframe["precise_cluster"] = hilary.df["precise_cluster"]
    dataframe["sensitive_cluster"] = hilary.df["sensitive_cluster"]
    output_path = result_folder / Path(f"inferred_{data_path.name}")

    log.info("💾 SAVING RESULTS ", output_path=output_path.as_posix())
    save_dataframe(dataframe=dataframe, save_path=output_path)


if __name__ == "__main__":
    app()
