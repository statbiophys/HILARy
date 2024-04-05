"""Execute hilary with command line."""

from __future__ import annotations

from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import typer

from hilary.apriori import Apriori
from hilary.inference import HILARy
from hilary.utils import (
    create_classes,
    get_logger,
    pairwise_evaluation,
    read_input,
    save_dataframe,
)

app = typer.Typer(add_completion=False)


@app.command()
def crude_method(
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
        1,
        "--threads",
        "-t",
        help="Choose number of cpus on which to run code. -1 to use all available cpus.",
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
    fixed_threshold: int = typer.Option(
        -1,
        "--fixed_threshold",
        "-ft",
        help="Threshold to perform single linkage clustering on cdr3 hamming distance.",
    ),
    normalized_threshold: float = typer.Option(
        0.2,
        "--normalized_threshold",
        "-nt",
        help="Threshold to perform single linkage clustering on cdr3 normalized hamming distance.",
    ),
) -> None:
    """Infer lineages with standard method from data_path excel file."""
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
    if threads == -1:
        threads = cpu_count()
    log = get_logger(verbose=verbose, use_json=use_json)
    log.info(
        "📖 READING DATA ",
        data_path=data_path.as_posix(),
    )
    dataframe = read_input(input_path=data_path, config=config)
    if "sequence_id" not in dataframe.columns:
        log.warning("No 'sequence_id' column present in file.")
        dataframe["sequence_id"] = dataframe.index.astype("str")
    dataframe["sequence_id"] = dataframe["sequence_id"].str.strip("-igh")
    dataframe.set_index("sequence_id")
    paired = False
    if kappa_file:
        log.info("USING PAIRED OPTION.")
        dataframe_kappa = read_input(input_path=kappa_file, config=config)
        dataframe_kappa["sequence_id"] = dataframe_kappa["sequence_id"].str.strip(
            "-igk"
        )
        dataframe_kappa.set_index("sequence_id")
        lengths = np.arange(57, 144 + 3, 3).astype(int)
        paired = True
    else:
        dataframe_kappa = None
        lengths = np.arange(15, 81 + 3, 3).astype(int)

    if verbose >= 2:
        input_path = debug_folder / Path(f"input_{data_path.name}")
        log.debug(
            "Saving input dataframe.",
            path=input_path.as_posix(),
        )
        save_dataframe(dataframe=dataframe, save_path=input_path)
    apriori = Apriori(
        paired=paired,
        lengths=lengths,
        threads=threads,
    )
    dataframe_processed = apriori.preprocess(df=dataframe, df_kappa=dataframe_kappa)
    apriori.classes = create_classes(dataframe_processed)
    hilary = HILARy(
        apriori,
        df=dataframe_processed,
        crude=True,
    )
    dataframe_crude = hilary.compute_crude_method_clusters(
        dataframe_processed,
        fixed_threshold=fixed_threshold,
        normalized_threshold=normalized_threshold,
    )
    dataframe["crude_method_family"] = dataframe_crude["crude_method_family"]
    dataframe["sequence_id"] = dataframe["sequence_id"] + "-igh"

    log.info("💾 SAVING RESULTS ", output_path=output_path.as_posix())
    output_path = result_folder / Path(f"inferred_crude_method_{data_path.name}")
    save_dataframe(dataframe=dataframe, save_path=output_path)
    if paired:
        dataframe_kappa["crude_method_family"] = dataframe_crude["crude_method_family"]
        dataframe_kappa["sequence_id"] = dataframe["sequence_id"] + "-igk"
        output_path_kappa = result_folder / Path(
            f"inferred_crude_method_{kappa_file.name}"
        )
        log.info(
            "💾 SAVING RESULTS FOR KAPPA FILE",
            output_path=output_path_kappa.as_posix(),
        )
        save_dataframe(dataframe=dataframe_kappa, save_path=output_path_kappa)

    if verbose >= 2 and "clone_id" in dataframe.columns:
        precision, sensitivity = pairwise_evaluation(
            df=dataframe, partition="crude_method_family"
        )
        log.debug(
            "Evaluating Hilary's performance on ground truth column 'clone_id'.",
            precision_crude=precision,
            sensitivity_crude=sensitivity,
        )


@app.command()
def cdr3_method(
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
        1,
        "--threads",
        "-t",
        help="Choose number of cpus on which to run code. -1 to use all available cpus.",
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
    saving: bool = True,
) -> None:
    """Infer lineages with HILARy-CDR3 from data_path excel file."""
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
    if threads == -1:
        threads = cpu_count()
    log = get_logger(verbose=verbose, use_json=use_json)
    log.info(
        "📖 READING DATA ",
        data_path=data_path.as_posix(),
    )
    dataframe = read_input(input_path=data_path, config=config)
    if "sequence_id" not in dataframe.columns:
        log.warning("No 'sequence_id' column present in file.")
        dataframe["sequence_id"] = dataframe.index.astype("str")
    dataframe["sequence_id"] = dataframe["sequence_id"].str.strip("-igh")
    dataframe.set_index("sequence_id")
    paired = False
    if kappa_file:
        log.info("USING PAIRED OPTION.")
        dataframe_kappa = read_input(input_path=kappa_file, config=config)
        dataframe_kappa["sequence_id"] = dataframe_kappa["sequence_id"].str.strip(
            "-igk"
        )
        dataframe_kappa.set_index("sequence_id")
        lengths = np.arange(57, 144 + 3, 3).astype(int)
        paired = True
    else:
        dataframe_kappa = None
        lengths = np.arange(15, 81 + 3, 3).astype(int)

    apriori = Apriori(
        paired=paired,
        lengths=lengths,
        threads=threads,
        precision=precision,
        sensitivity=sensitivity,
        silent=silent,
    )
    dataframe_processed = apriori.preprocess(df=dataframe, df_kappa=dataframe_kappa)
    apriori.classes = create_classes(dataframe_processed)

    log.info("⏳ COMPUTING HISTOGRAMS ⏳.")
    apriori.get_histograms(dataframe_processed)

    if verbose >= 2:
        histogram_path = debug_folder / Path(f"histograms_{data_path.name}")
        log.debug("Saving histograms used by Hilary.", path=histogram_path.as_posix())
        save_dataframe(apriori.histograms, histogram_path)

    log.info("⏳ COMPUTING PARAMETERS ⏳.")
    apriori.get_parameters()

    log.info("⏳ COMPUTING THRESHOLDS ⏳.")
    apriori.get_thresholds()
    if verbose >= 2:
        parameters_path = debug_folder / Path(f"parameters_{data_path.name}")
        log.debug(
            "Saving all parameters inferred by Hilary.",
            path=parameters_path.as_posix(),
        )
        save_dataframe(apriori.classes, parameters_path)

    log.info("⏳ COMPUTING PRECISE AND SENSITIVE CLUSTERS ⏳.")
    hilary = HILARy(
        apriori,
        df=dataframe_processed,
    )
    dataframe_cdr3 = hilary.compute_prec_sens_clusters(dataframe_processed)

    dataframe["cdr3_based_family"] = dataframe_cdr3["precise_cluster"]
    dataframe["sequence_id"] = dataframe["sequence_id"] + "-igh"

    if saving:
        log.info("💾 SAVING RESULTS ", output_path=output_path.as_posix())
        output_path = result_folder / Path(f"inferred_cdr3_based_{data_path.name}")
        save_dataframe(dataframe=dataframe, save_path=output_path)
        if paired:
            dataframe_kappa["cdr3_based_family"] = dataframe_cdr3["precise_cluster"]
            dataframe_kappa["sequence_id"] = dataframe["sequence_id"] + "-igk"
            output_path_kappa = result_folder / Path(
                f"inferred_cdr3_based_{kappa_file.name}"
            )
            log.info(
                "💾 SAVING RESULTS FOR KAPPA FILE",
                output_path=output_path_kappa.as_posix(),
            )
            save_dataframe(dataframe=dataframe_kappa, save_path=output_path_kappa)

    if verbose >= 2 and "clone_id" in dataframe.columns:
        precision, sensitivity = pairwise_evaluation(
            df=dataframe, partition="cdr3_based_family"
        )
        log.debug(
            "Evaluating Hilary's performance on ground truth column 'clone_id'.",
            precision_cdr3=precision,
            sensitivity_cdr3=sensitivity,
        )
    return dataframe_cdr3, dataframe, dataframe_kappa, hilary


@app.command()
def full_method(
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
        1,
        "--threads",
        "-t",
        help="Choose number of cpus on which to run code. -1 to use all available cpus.",
    ),
    precision: float = typer.Option(
        1,
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
    without_heuristic: bool = typer.Option(
        False,
        "--without-heuristic",
        help="DO not use heuristic for choosing the xy threshold.",
    ),
) -> None:
    """Infer lineages with HILARy-full from data_path excel file."""
    if result_folder is None:
        result_folder = data_path.parents[0] / Path("hilary_results/")
    result_folder.mkdir(parents=True, exist_ok=True)
    debug_folder = result_folder / Path("debug/")
    if kappa_file:
        xy_threshold = 4
    else:
        xy_threshold = 0
    log = get_logger(verbose=verbose, use_json=use_json)

    dataframe_cdr3, dataframe, dataframe_kappa, hilary = cdr3_method(
        data_path=data_path,
        kappa_file=kappa_file,
        verbose=verbose,
        threads=threads,
        precision=precision,
        sensitivity=sensitivity,
        silent=silent,
        result_folder=result_folder,
        config=config,
        override=override,
        use_json=use_json,
        saving=False,
    )
    if without_heuristic:
        hilary.classes["xy_threshold"] = xy_threshold
    else:
        log.info("⏳ COMPUTING XY THRESHOLDS ⏳.")
        hilary.get_xy_thresholds(df=dataframe_cdr3)
        hilary.classes["xy_threshold"] = hilary.classes["xy_threshold"] + xy_threshold
    if verbose >= 2:
        parameters_path = debug_folder / Path(f"parameters_{data_path.name}")
        log.debug(
            "Saving all parameters inferred by Hilary.",
            path=parameters_path.as_posix(),
        )
        save_dataframe(hilary.classes, parameters_path)

    log.info("⏳ INFERRING FAMILIES WITH FULL XY METHOD⏳.")
    dataframe_inferred = hilary.infer(df=dataframe_cdr3)

    dataframe["family"] = dataframe_inferred["family"]

    output_path = result_folder / Path(f"inferred_full_method_{data_path.name}")
    log.info("💾 SAVING RESULTS ", output_path=output_path.as_posix())

    save_dataframe(dataframe=dataframe, save_path=output_path)

    if dataframe_kappa is not None:
        dataframe_kappa["family"] = dataframe_inferred["family"]
        output_path_kappa = result_folder / Path(
            f"inferred_full_method_{kappa_file.name}"
        )
        log.info(
            "💾 SAVING RESULTS FOR KAPPA FILE",
            output_path=output_path_kappa.as_posix(),
        )
        save_dataframe(dataframe=dataframe_kappa, save_path=output_path_kappa)

    if verbose >= 2 and "clone_id" in dataframe.columns:
        precision_full, sensitivity_full = pairwise_evaluation(
            df=dataframe, partition="family"
        )
        precision_cdr3, sensitivity_cdr3 = pairwise_evaluation(
            df=dataframe, partition="cdr3_based_family"
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
