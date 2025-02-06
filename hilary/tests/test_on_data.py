from pathlib import Path

import pandas as pd
import structlog
from tqdm import tqdm

from hilary.apriori import Apriori
from hilary.inference import HILARy
from hilary.utils import create_classes, pairwise_evaluation

log = structlog.get_logger(__name__)
file_path = Path(__file__).parent / "data_for_tests"

thresholds_dict={
    "partis_single_20":{"precision_cdr":0.995,"sensitivity_full":0.89, "precision_full":0.995},
    "partis_single_05":{"precision_cdr":0.965,"sensitivity_cdr":0.925,"sensitivity_full":0.985, "precision_full":0.97},# downgrade of sensitivity_full 0.975
    "nat_15":{"precision_cdr":0.995,"precision_full":0.995,"sensitivity_full":0.975},
    "nat_24":{"precision_cdr":0.995,"precision_full":0.99,"sensitivity_full":0.975},
    "nat_39":{"precision_cdr":0.995,"sensitivity_cdr":0.96,"precision_full":0.995,"sensitivity_full":0.985},
    "naive_human":{"precision":0.995},
    "naive_mouse":{"precision":0.995}}

def check_performance_on_nat_data():
    for length in [15, 24, 39]:
        log.info(
            "Processing file.", file=f"families1_1e4_ppost326651_mut326713_cdr3l{length}.csv.gz"
        )
        dataframe = pd.read_csv(
            file_path / f"families1_1e4_ppost326651_mut326713_cdr3l{length}.csv.gz",
            compression="gzip",
        )
        dataframe = dataframe.rename(
            columns={
                "alt_sequence_alignment_bis": "alt_sequence_alignment",
                "alt_germline_alignment_bis": "alt_germline_alignment",
                "V_GENE": "v_gene",
                "J_GENE": "j_gene",
                "CDR3_LENGTH": "cdr3_length",
                "CDR3": "cdr3",
                "FAMILY": "ground_truth",
            }
        )
        dataframe["sequence_id"] = dataframe.index.astype("str")
        apriori = Apriori(
            silent=False, threads=-1, precision=1, sensitivity=0.95, null_model="l"
        )  # show progress bars, use all threads
        dataframe_processed = apriori.preprocess(df=dataframe, df_kappa=None)
        apriori.classes = create_classes(dataframe_processed)
        apriori.get_histograms(dataframe_processed)
        apriori.get_parameters()
        hilary = HILARy(apriori, df=dataframe_processed)
        dataframe_cdr3 = hilary.compute_prec_sens_clusters(df=dataframe_processed)
        dataframe["cdr3_based_family"] = dataframe_cdr3["precise_cluster"]
        precision_cdr3, sensitivity_cdr3 = pairwise_evaluation(
            df=dataframe, partition="cdr3_based_family"
        )

        assert precision_cdr3 > thresholds_dict[f"nat_{length}"]["precision_cdr"]
        hilary.get_xy_thresholds(df=dataframe_cdr3)
        dataframe_inferred = hilary.infer(df=dataframe_cdr3)
        dataframe["clone_id"] = dataframe_inferred["clone_id"]
        precision_full, sensitivity_full = pairwise_evaluation(df=dataframe, partition="clone_id")
        assert precision_full > thresholds_dict[f"nat_{length}"]["precision_full"]
        assert sensitivity_full > thresholds_dict[f"nat_{length}"]["sensitivity_full"]
        log.info(
            "Showing metrics for given file.",
            file=f"families1_1e4_ppost326651_mut326713_cdr3l{length}",
            precision_cdr3=precision_cdr3,
            sensitivity_cdr3=sensitivity_cdr3,
            precision_full_method=precision_full,
            sensitivity_full_method=sensitivity_full,
        )


def check_performance_on_partis_data():
    for mut in ["05","20"]:
        log.info("Processing file.", file=f"partis_{mut}/single_chain/igh.csv.gz")
        dataframe = pd.read_csv(
            file_path / f"partis_{mut}/single_chain/igh.csv.gz",
            compression="gzip",
        )
        dataframe = dataframe.rename(
            columns={
                "v_gl_seq": "v_germline_alignment",
                "v_qr_seqs": "v_sequence_alignment",
                "j_gl_seq": "j_germline_alignment",
                "j_qr_seqs": "j_sequence_alignment",
                "clone_id": "ground_truth",
            }
        )
        dataframe["sequence_id"] = dataframe.index.astype("str")
        apriori = Apriori(
            silent=False, threads=-1, sensitivity=1, precision=1, null_model="l"
        )  # show progress bars, use all threads
        dataframe_processed = apriori.preprocess(df=dataframe, df_kappa=None)
        apriori.classes = create_classes(dataframe_processed)
        apriori.get_histograms(dataframe_processed)
        apriori.get_parameters()
        hilary = HILARy(apriori, df=dataframe_processed)
        dataframe_cdr3 = hilary.compute_prec_sens_clusters(df=dataframe_processed)
        dataframe["cdr3_based_family"] = dataframe_cdr3["precise_cluster"]
        precision_cdr3, sensitivity_cdr3 = pairwise_evaluation(
            df=dataframe, partition="cdr3_based_family"
        )

        assert precision_cdr3 > thresholds_dict[f"partis_single_{mut}"]["precision_cdr"]
        hilary.get_xy_thresholds(df=dataframe_cdr3)
        dataframe_inferred = hilary.infer(df=dataframe_cdr3)
        dataframe["clone_id"] = dataframe_inferred["clone_id"]
        precision_full, sensitivity_full = pairwise_evaluation(df=dataframe, partition="clone_id")
        assert precision_full > thresholds_dict[f"partis_single_{mut}"]["precision_full"]
        assert sensitivity_full > thresholds_dict[f"partis_single_{mut}"]["sensitivity_full"]
        log.info(
            "Showing metrics for given file.",
            file=f"partis_{mut}/single_chain/igh.csv.gz",
            precision_cdr3=precision_cdr3,
            sensitivity_cdr3=sensitivity_cdr3,
            precision_full_method=precision_full,
            sensitivity_full_method=sensitivity_full,
        )


def check_performance_on_naive_mouse_data():
    log.info("Processing file.", file="generated_mouse_post_aligned_subsampled50K.csv.gz")
    dataframe = pd.read_csv(
        file_path / "generated_mouse_post_aligned_subsampled50K.csv.gz",
        compression="gzip",
    )
    dataframe["sequence_id"] = dataframe.index.astype("str")
    apriori = Apriori(
        silent=False, threads=-1, precision=1, sensitivity=0.95, species="mouse", null_model="l"
    )  # show progress bars, use all threads
    dataframe_processed = apriori.preprocess(df=dataframe, df_kappa=None)
    apriori.classes = create_classes(dataframe_processed)
    apriori.get_histograms(dataframe_processed)
    apriori.get_parameters()
    hilary = HILARy(apriori, df=dataframe_processed)
    dataframe_cdr3 = hilary.compute_prec_sens_clusters(df=dataframe_processed)
    dataframe["cdr3_based_family"] = dataframe_cdr3["precise_cluster"]
    precision_cdr3 = len(dataframe["cdr3_based_family"].unique()) / len(dataframe)
    assert precision_cdr3 > thresholds_dict["naive_mouse"]["precision"]
    hilary.get_xy_thresholds(df=dataframe_cdr3)
    dataframe_inferred = hilary.infer(df=dataframe_cdr3)
    dataframe["clone_id"] = dataframe_inferred["clone_id"]
    precision_full = len(dataframe["clone_id"].unique()) / len(dataframe)
    assert precision_full > thresholds_dict["naive_mouse"]["precision"]
    log.info(
        "Showing metrics for given file.",
        file="generated_mouse_post_aligned_subsampled50K.csv.gz",
        precision_cdr3=precision_cdr3,
        precision_full_method=precision_full,
    )


def check_performance_on_naive_human_data():
    log.info("Processing file.", file="sonia_human_igh_aligned_subsampled50K.csv.gz")
    dataframe = pd.read_csv(
        file_path / "sonia_human_igh_aligned_subsampled50K.csv.gz",
        compression="gzip",
    )
    dataframe["sequence_id"] = dataframe.index.astype("str")
    apriori = Apriori(
        silent=False, threads=-1, precision=1, sensitivity=0.95, null_model="l"
    )  # show progress bars, use all threads
    dataframe_processed = apriori.preprocess(df=dataframe, df_kappa=None)
    apriori.classes = create_classes(dataframe_processed)
    apriori.get_histograms(dataframe_processed)
    apriori.get_parameters()
    hilary = HILARy(apriori, df=dataframe_processed)
    dataframe_cdr3 = hilary.compute_prec_sens_clusters(df=dataframe_processed)
    dataframe["cdr3_based_family"] = dataframe_cdr3["precise_cluster"]
    precision_cdr3 = len(dataframe["cdr3_based_family"].unique()) / len(dataframe)
    assert precision_cdr3 > thresholds_dict["naive_human"]["precision"]
    hilary.get_xy_thresholds(df=dataframe_cdr3)
    dataframe_inferred = hilary.infer(df=dataframe_cdr3)
    dataframe["clone_id"] = dataframe_inferred["clone_id"]
    precision_full = len(dataframe["clone_id"].unique()) / len(dataframe)
    assert precision_full > thresholds_dict["naive_human"]["precision"]
    log.info(
        "Showing metrics for given file.",
        file="sonia_human_igh_aligned_subsampled50K.csv.gz",
        precision_cdr3=precision_cdr3,
        precision_full_method=precision_full,
    )

check_performance_on_nat_data()
check_performance_on_naive_human_data()
check_performance_on_partis_data()
check_performance_on_naive_mouse_data()
