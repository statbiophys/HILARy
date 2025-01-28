import pandas as pd
import structlog
from tqdm import tqdm

from hilary.apriori import Apriori
from hilary.inference import HILARy
from hilary.utils import create_classes, pairwise_evaluation

log = structlog.get_logger(__name__)

def check_performance_on_nat_data():
    for length in [15,24,39]:
        log.info("Processing file.", file=f"families1_1e4_ppost326651_mut326713_cdr3l{length}.csv.gz")
        dataframe = pd.read_csv(
            f"/home/athenes/gitlab/HILARy/hilary/tests/data_for_tests/families1_1e4_ppost326651_mut326713_cdr3l{length}.csv.gz",
            compression="gzip"
            )
        dataframe = dataframe.rename(columns={"alt_sequence_alignment_bis":"alt_sequence_alignment",
            "alt_germline_alignment_bis":"alt_germline_alignment",
            "V_GENE":"v_gene",
            "J_GENE":"j_gene",
            "CDR3_LENGTH":"cdr3_length",
            "CDR3":"cdr3",
            "FAMILY":"ground_truth"
        })
        dataframe["sequence_id"] = dataframe.index.astype("str")
        apriori = Apriori(silent=False, threads=-1, precision=1, sensitivity=0.95) # show progress bars, use all threads
        dataframe_processed = apriori.preprocess(df=dataframe, df_kappa=None)
        apriori.classes= create_classes(dataframe_processed)
        apriori.get_histograms(dataframe_processed)
        apriori.get_parameters()
        apriori.get_thresholds()
        hilary=HILARy(apriori,df=dataframe_processed)
        dataframe_cdr3=hilary.compute_prec_sens_clusters(df=dataframe_processed)
        dataframe["cdr3_based_family"] = dataframe_cdr3["precise_cluster"]
        precision_cdr3, sensitivity_cdr3=pairwise_evaluation(df=dataframe, partition="cdr3_based_family")

        assert precision_cdr3 > 0.95
        hilary.get_xy_thresholds(df=dataframe_cdr3)
        dataframe_inferred = hilary.infer(df=dataframe_cdr3)
        dataframe["clone_id"] = dataframe_inferred["clone_id"]
        precision_full, sensitivity_full = pairwise_evaluation(df=dataframe, partition="clone_id")
        assert precision_full > 0.95
        assert sensitivity_full > 0.95
        log.info("Showing metrics for given file.",
                file=f"families1_1e4_ppost326651_mut326713_cdr3l{length}",
                precision_cdr3 = precision_cdr3,
                sensitivity_cdr3 = sensitivity_cdr3,
                precision_full_method=precision_full,
                sensitivity_full_method=sensitivity_full
            )

def check_performance_on_partis_data():
    for mut in ["20","05"]:
        log.info("Processing file.", file=f"partis_{mut}/single_chain/igh.csv.gz")
        dataframe = pd.read_csv(
            f"/home/athenes/gitlab/HILARy/hilary/tests/data_for_tests/partis_{mut}/single_chain/igh.csv.gz",
            compression="gzip"
            )
        dataframe = dataframe.rename(columns={
            "v_gl_seq":"v_germline_alignment",
            "v_qr_seqs": "v_sequence_alignment",
            "j_gl_seq":"j_germline_alignment",
            "j_qr_seqs":"j_sequence_alignment",
            "clone_id":"ground_truth"
        })
        dataframe["sequence_id"] = dataframe.index.astype("str")
        apriori = Apriori(silent=False, threads=-1, sensitivity=1, precision=1) # show progress bars, use all threads
        dataframe_processed = apriori.preprocess(df=dataframe, df_kappa=None)
        apriori.classes= create_classes(dataframe_processed)
        apriori.get_histograms(dataframe_processed)
        apriori.get_parameters()
        apriori.get_thresholds()
        hilary=HILARy(apriori,df=dataframe_processed)
        dataframe_cdr3=hilary.compute_prec_sens_clusters(df=dataframe_processed)
        dataframe["cdr3_based_family"] = dataframe_cdr3["precise_cluster"]
        precision_cdr3, sensitivity_cdr3=pairwise_evaluation(df=dataframe, partition="cdr3_based_family")

        assert precision_cdr3 > 0.95
        hilary.get_xy_thresholds(df=dataframe_cdr3)
        dataframe_inferred = hilary.infer(df=dataframe_cdr3)
        dataframe["clone_id"] = dataframe_inferred["clone_id"]
        precision_full, sensitivity_full = pairwise_evaluation(df=dataframe, partition="clone_id")
        assert precision_full > 0.95
        assert sensitivity_full > 0.89
        log.info("Showing metrics for given file.",
                file=f"partis_{mut}/single_chain/igh.csv.gz",
                precision_cdr3 = precision_cdr3,
                sensitivity_cdr3 = sensitivity_cdr3,
                precision_full_method=precision_full,
                sensitivity_full_method=sensitivity_full
            )

def check_performance_on_naive_mouse_data():
    log.info("Processing file.", file="generated_mouse_post_aligned_subsampled50K.csv.gz")
    dataframe = pd.read_csv(
        "/home/athenes/gitlab/HILARy/hilary/tests/data_for_tests/generated_mouse_post_aligned_subsampled50K.csv.gz",
        compression="gzip"
        )
    dataframe["sequence_id"] = dataframe.index.astype("str")
    apriori = Apriori(silent=False, threads=-1, precision=1, sensitivity=0.95) # show progress bars, use all threads
    dataframe_processed = apriori.preprocess(df=dataframe, df_kappa=None)
    apriori.classes= create_classes(dataframe_processed)
    apriori.get_histograms(dataframe_processed)
    apriori.get_parameters()
    apriori.get_thresholds()
    hilary=HILARy(apriori,df=dataframe_processed)
    dataframe_cdr3=hilary.compute_prec_sens_clusters(df=dataframe_processed)
    dataframe["cdr3_based_family"] = dataframe_cdr3["precise_cluster"]
    precision_cdr3 = len(dataframe["cdr3_based_family"].unique())/len(dataframe)
    assert precision_cdr3 > 0.95
    hilary.get_xy_thresholds(df=dataframe_cdr3)
    dataframe_inferred = hilary.infer(df=dataframe_cdr3)
    dataframe["clone_id"] = dataframe_inferred["clone_id"]
    precision_full= len(dataframe["clone_id"].unique())/len(dataframe)
    assert precision_full > 0.95
    log.info("Showing metrics for given file.",
            file="generated_mouse_post_aligned_subsampled50K.csv.gz",
            precision_cdr3 = precision_cdr3,
            precision_full_method=precision_full,
        )

def check_performance_on_naive_human_data():
    log.info("Processing file.", file="sonia_human_igh_aligned_subsampled50K.csv.gz")
    dataframe = pd.read_csv(
        "/home/athenes/gitlab/HILARy/hilary/tests/data_for_tests/sonia_human_igh_aligned_subsampled50K.csv.gz",
        compression="gzip"
        )
    dataframe["sequence_id"] = dataframe.index.astype("str")
    apriori = Apriori(silent=False, threads=-1, precision=1, sensitivity=0.95) # show progress bars, use all threads
    dataframe_processed = apriori.preprocess(df=dataframe, df_kappa=None)
    apriori.classes= create_classes(dataframe_processed)
    apriori.get_histograms(dataframe_processed)
    apriori.get_parameters()
    apriori.get_thresholds()
    hilary=HILARy(apriori,df=dataframe_processed)
    dataframe_cdr3=hilary.compute_prec_sens_clusters(df=dataframe_processed)
    dataframe["cdr3_based_family"] = dataframe_cdr3["precise_cluster"]
    precision_cdr3 = len(dataframe["cdr3_based_family"].unique())/len(dataframe)
    assert precision_cdr3 > 0.99
    hilary.get_xy_thresholds(df=dataframe_cdr3)
    dataframe_inferred = hilary.infer(df=dataframe_cdr3)
    dataframe["clone_id"] = dataframe_inferred["clone_id"]
    precision_full= len(dataframe["clone_id"].unique())/len(dataframe)
    assert precision_full > 0.99
    log.info("Showing metrics for given file.",
            file="sonia_human_igh_aligned_subsampled50K.csv.gz",
            precision_cdr3 = precision_cdr3,
            precision_full_method=precision_full,
        )

check_performance_on_naive_human_data()
check_performance_on_naive_mouse_data()
check_performance_on_partis_data()
check_performance_on_nat_data()
