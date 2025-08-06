"""Execute hilary with command line."""
from __future__ import annotations

from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from hilary.apriori import Apriori
from hilary.cdr3_clustering import CDR3Clustering
from hilary.inference import HILARy
from hilary.utils import (
    create_classes,
    get_logger,
    pairwise_evaluation,
    preprocess,
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
    light_file: Path = typer.Option(
        None,
        "--light-file",
        help="Path of the light chain file, hilary will automatically use its paired option.",
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
    output_path = result_folder / Path(f"inferred_crude_method_{data_path.name}")
    if output_path.exists() and not override:
        raise ValueError(
            f"{output_path.as_posix()} already exists, use override parameter to replace the file.",
        )
    if threads == -1:
        threads = cpu_count()
    log = get_logger(verbose=verbose, use_json=use_json)
    log.info("🚀 RUNNING CRUDE METHOD 🚀")
    log.info(
        "📖 READING DATA 📖",
        data_path=data_path.as_posix(),
    )
    dataframe = read_input(input_path=data_path, config=config)
    if "sequence_id" not in dataframe.columns:
        log.warning("No 'sequence_id' column present in file.")
        dataframe["sequence_id"] = dataframe.index.astype("str")
    dataframe["sequence_id"] = dataframe["sequence_id"].astype(str)
    dataframe["sequence_id"] = dataframe["sequence_id"].str.strip("-igh")
    dataframe.set_index("sequence_id")
    paired = False
    if light_file:
        log.info("USING PAIRED OPTION.")
        dataframe_light = read_input(input_path=light_file, config=config)
        dataframe_light["sequence_id"] = dataframe_light["sequence_id"].str.strip("-igk")
        dataframe_light.set_index("sequence_id")
        paired = True
    else:
        dataframe_light = None

    dataframe_processed = preprocess(df=dataframe, df_light=dataframe_light, threads=threads)
    classes = create_classes(dataframe_processed)
    if fixed_threshold >= 0:
        log.debug("Using crude method with a fixed threshold.", threshold=fixed_threshold)
        classes["threshold"] = fixed_threshold
    else:
        log.debug("Using crude method with a normalized threshold.", threshold=normalized_threshold)
        classes["threshold"] = (classes["cdr3_length_value"] * normalized_threshold).astype(int)
    clustering = CDR3Clustering(thresholds=classes, threads=threads)
    dataframe["clone_id"] = clustering.infer(dataframe_processed, silent=False)
    dataframe["sequence_id"] = dataframe["sequence_id"] + "-igh"
    save_dataframe(dataframe=dataframe, save_path=output_path)

    if paired:
        dataframe_light["clone_id"] = dataframe["clone_id"]
        dataframe_light["sequence_id"] = dataframe["sequence_id"] + "-igk"
        output_path_light = result_folder / Path(f"inferred_crude_method_{light_file.name}")
        save_dataframe(dataframe=dataframe_light, save_path=output_path_light)

    if "ground_truth" in dataframe.columns:
        precision, sensitivity = pairwise_evaluation(df=dataframe, partition="clone_id")
        log.debug(
            "Evaluating Hilary's performance on ground truth column 'ground_truth'.",
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
    light_file: Path = typer.Option(
        None,
        "--light-file",
        help="Path of the light chain file, hilary will automatically use its paired option.",
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
        0.995,
        "--precision",
        "-p",
        help="Choose desired precision.",
    ),
    sensitivity: float = typer.Option(
        0.995,
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
    model: str = typer.Option(
        "human_B_heavy",
        "--model",
        help="Model to use among 'human_B_heavy','human_B_kappa','human_B_lambda',\
                'human_paired', 'mouse_B_heavy','mouse_B_kappa','mouse_B_lambda','mouse_B_paired'.\
                Defaul to 'human_B_heavy'.",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """Infer lineages with HILARy-CDR3 from data_path excel file."""
    if result_folder is None:
        result_folder = data_path.parents[0] / Path("hilary_results/")
    result_folder.mkdir(parents=True, exist_ok=True)
    debug_folder = result_folder / Path("debug/")
    debug_folder.mkdir(parents=True, exist_ok=True)
    output_path = result_folder / Path(f"inferred_cdr3_based_{data_path.name}")
    if output_path.exists() and not override:
        raise ValueError(
            f"{output_path.as_posix()} already exists, use override parameter to replace the file.",
        )
    if threads == -1:
        threads = cpu_count()
    log = get_logger(verbose=verbose, use_json=use_json)
    log.info("📖 READING DATA 📖", data_path=data_path.as_posix())
    log.info("🚀 RUNNING CDR3 METHOD 🚀")
    dataframe = read_input(input_path=data_path, config=config)
    if "sequence_id" not in dataframe.columns:
        log.warning("No 'sequence_id' column present in file.")
        dataframe["sequence_id"] = dataframe.index.astype("str")
    dataframe["sequence_id"] = dataframe["sequence_id"].astype(str)
    dataframe["sequence_id"] = dataframe["sequence_id"].str.strip("-igh")
    dataframe.set_index("sequence_id")
    paired = False
    if light_file:
        log.info("USING PAIRED OPTION.")
        dataframe_light = read_input(input_path=light_file, config=config)
        dataframe_light["sequence_id"] = dataframe_light["sequence_id"].str.strip("-igk")
        dataframe_light.set_index("sequence_id")
        paired = True
    else:
        dataframe_light = None

    # running cdr3 method
    apriori = Apriori(
        paired=paired,
        threads=threads,
        precision=precision,
        model=model,
        sensitivity=sensitivity,
        silent=silent,
    )
    dataframe_processed = preprocess(df=dataframe, df_light=dataframe_light, threads=threads)
    apriori.classes = create_classes(dataframe_processed)
    apriori.get_histograms(dataframe_processed)
    apriori.get_parameters()
    apriori.classes["threshold"] = apriori.classes["precise_threshold"]
    clustering = CDR3Clustering(thresholds=apriori.classes, threads=threads)
    dataframe["clone_id"] = clustering.infer(dataframe_processed, silent=False)
    dataframe["sequence_id"] = dataframe["sequence_id"] + "-igh"
    save_dataframe(dataframe=dataframe, save_path=output_path)

    # Save results for light chain if exists
    if paired:
        dataframe_light["clone_id"] = dataframe["precise_cluster"]
        dataframe_light["sequence_id"] = dataframe_light["sequence_id"] + "-igk"
        output_path_light = result_folder / Path(f"inferred_cdr3_based_{light_file.name}")
        save_dataframe(dataframe=dataframe_light, save_path=output_path_light)
    # Save parameters and histograms if debug mode
    if verbose >= 2:
        parameters_path = debug_folder / Path(f"parameters_{data_path.name}")
        log.debug(
            "Saving all parameters inferred by Hilary.",
            path=parameters_path.as_posix(),
        )
        save_dataframe(apriori.classes, parameters_path)
        histogram_path = debug_folder / Path(f"histograms_{data_path.name}")
        log.debug("Saving histograms used by Hilary.", path=histogram_path.as_posix())
        save_dataframe(apriori.histograms, histogram_path)
    # show results if we have ground truth info
    if "ground_truth" in dataframe.columns:
        precision, sensitivity = pairwise_evaluation(df=dataframe, partition="clone_id")
        log.debug(
            "Evaluating Hilary's performance on ground truth column 'ground_truth'.",
            precision_cdr3=precision,
            sensitivity_cdr3=sensitivity,
        )
    return dataframe


@app.command()
def full_method(
    data_path: Path = typer.Argument(
        ...,
        help="Path of the excel file to infer lineages.",
        show_default=False,
    ),
    light_file: Path = typer.Option(
        None,
        "--light-file",
        help="Path of the light chain file, hilary will automatically use its paired option.",
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
    silent: bool = typer.Option(
        False,
        "--silent",
        help="Do not show progress bars if used.",
    ),
    result_folder: Path = typer.Option(
        None,
        "--result-folder",
        "-r",
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
) -> None:
    """Infer lineages with HILARy-full from data_path excel file."""
    if result_folder is None:
        result_folder = data_path.parents[0] / Path("hilary_results/")
    result_folder.mkdir(parents=True, exist_ok=True)
    debug_folder = result_folder / Path("debug/")
    debug_folder.mkdir(parents=True, exist_ok=True)
    output_path = result_folder / Path(f"inferred_full_method_{data_path.name}")
    if output_path.exists() and not override:
        raise ValueError(
            f"{output_path.as_posix()} already exists, use override parameter to replace the file.",
        )
    if threads == -1:
        threads = cpu_count()
    if light_file:
        xy_threshold = 4
    else:
        xy_threshold = 0
    log = get_logger(verbose=verbose, use_json=use_json)
    log.info(
        "📖 READING DATA 📖",
        data_path=data_path.as_posix(),
    )
    log.info("🚀 RUNNING FULL PHILOGENETIC METHOD 🚀")
    dataframe = read_input(input_path=data_path, config=config)
    if "sequence_id" not in dataframe.columns:
        log.warning("No 'sequence_id' column present in file.")
        dataframe["sequence_id"] = dataframe.index.astype("str")
    dataframe["sequence_id"] = dataframe["sequence_id"].astype(str)
    dataframe["sequence_id"] = dataframe["sequence_id"].str.strip("-igh")
    dataframe.set_index("sequence_id")
    paired = False
    if light_file:
        log.info("USING PAIRED OPTION.")
        dataframe_light = read_input(input_path=light_file, config=config)
        dataframe_light["sequence_id"] = dataframe_light["sequence_id"].str.strip("-igk")
        dataframe_light.set_index("sequence_id")
        paired = True
    else:
        dataframe_light = None

    # running full method
    dataframe_processed = preprocess(
        df=dataframe, df_light=dataframe_light, threads=threads, silent=silent
    )
    dataframe_processed["split_up_cluster"] = dataframe_processed.groupby(
        ["v_gene", "j_gene", "cdr3_length"]
    ).ngroup()
    cluster_sizes = dataframe_processed.groupby("split_up_cluster").size()
    dataframe_processed["VJL_class_size"] = dataframe_processed["split_up_cluster"].map(
        cluster_sizes
    )
    limit = 20000
    if dataframe_processed["VJL_class_size"].max() > limit:
        apriori = Apriori(
            paired=paired,
            threads=threads,
            precision=1,
            sensitivity=0.99,
            model="human_B_heavy",
            silent=silent,
        )
        dataframe_big_vjl = dataframe_processed.query("VJL_class_size>@limit")
        apriori.classes = create_classes(dataframe_big_vjl)
        apriori.get_histograms(dataframe_big_vjl)
        apriori.get_parameters()
        apriori.classes["threshold"] = apriori.classes["sensitive_threshold"]
        clustering = CDR3Clustering(thresholds=apriori.classes, threads=threads)
        dataframe_big_vjl["split_up_cluster"] = clustering.infer(dataframe_big_vjl, silent=False)
        dataframe_processed = pd.concat(
            [dataframe_processed.query("VJL_class_size<=@limit"), dataframe_big_vjl]
        ).sort_index()
    hilary = HILARy(
        df=dataframe_processed,
        paired=paired,
        threads=threads,
        silent=silent,
    )
    dataframe["sequence_id"] = dataframe["sequence_id"] + "-igh"
    hilary.get_xy_thresholds(df=dataframe_processed)
    hilary.classes["xy_threshold"] = hilary.classes["xy_threshold"] + xy_threshold
    dataframe_inferred = hilary.infer(df=dataframe_processed)
    dataframe["clone_id"] = dataframe_inferred["clone_id"]
    save_dataframe(dataframe=dataframe, save_path=output_path)

    # Save results for light chain if exists
    if dataframe_light is not None:
        dataframe_light["clone_id"] = dataframe_inferred["clone_id"]
        output_path_light = result_folder / Path(f"inferred_full_method_{light_file.name}")
        save_dataframe(dataframe=dataframe_light, save_path=output_path_light)
    # Save parameters and histograms if debug mode
    if verbose >= 2 and dataframe_processed["VJL_class_size"].max() > limit:
        parameters_path = debug_folder / Path(f"parameters_{data_path.name}")
        log.debug(
            "Saving all parameters inferred by Hilary.",
            path=parameters_path.as_posix(),
        )
        save_dataframe(apriori.classes, parameters_path)
        histogram_path = debug_folder / Path(f"histograms_{data_path.name}")
        log.debug("Saving histograms used by Hilary.", path=histogram_path.as_posix())
        save_dataframe(apriori.histograms, histogram_path)
    # show results if we have ground truth info
    if "ground_truth" in dataframe.columns:
        precision_full, sensitivity_full = pairwise_evaluation(df=dataframe, partition="clone_id")
        log.debug(
            "Evaluating Hilary's performance on ground truth column 'ground_truth'.",
            precision_full_method=precision_full,
            sensitivity_full_method=sensitivity_full,
        )


if __name__ == "__main__":
    app()
