from pathlib import Path

import pandas as pd
import structlog

from hilary.apriori import Apriori
from hilary.cdr3_clustering import CDR3Clustering
from hilary.inference import HILARy
from hilary.utils import create_classes, pairwise_evaluation, preprocess

log = structlog.get_logger(__name__)
file_path = Path(__file__).parent / "data_for_tests"

thresholds_dict = {
    "partis_single_20": {"precision_cdr": 0.995, "sensitivity_full": 0.80, "precision_full": 0.992},
    "partis_single_05": {
        "precision_cdr": 0.965,
        "sensitivity_cdr": 0.925,
        "sensitivity_full": 0.96,
        "precision_full": 0.97,
    },  # downgrade of sensitivity_full 0.975
    "nat_15": {"precision_cdr": 0.985, "precision_full": 0.985, "sensitivity_full": 0.975},
    "nat_24": {"precision_cdr": 0.99, "precision_full": 0.99, "sensitivity_full": 0.975},
    "nat_39": {
        "precision_cdr": 0.99,
        "sensitivity_cdr": 0.96,
        "precision_full": 0.99,
        "sensitivity_full": 0.985,
    },
    "naive_human": {"precision": 0.995},
    "naive_mouse": {"precision": 0.995},
    "crude": {"precision": 0.2, "sensitivity": 0.2},  # Added crude method thresholds
}
hilary_pars = {"precision": 1, "sensitivity": 0.995}


def check_performance_on_nat_data():
    """Test HILARy full method on natural data."""
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

        # Test CDR3 method
        apriori = Apriori(
            silent=False,
            threads=48,
            precision=hilary_pars["precision"],
            sensitivity=hilary_pars["sensitivity"],
        )
        dataframe_processed = preprocess(df=dataframe, df_light=None, threads=48)
        apriori.classes = create_classes(dataframe_processed)
        apriori.get_histograms(dataframe_processed)
        apriori.get_parameters()
        apriori.classes["threshold"] = apriori.classes["precise_threshold"]
        clustering = CDR3Clustering(thresholds=apriori.classes, threads=48)
        dataframe["cdr3_based_family"] = clustering.infer(dataframe_processed, silent=False)
        precision_cdr3, sensitivity_cdr3 = pairwise_evaluation(
            df=dataframe, partition="cdr3_based_family"
        )
        assert precision_cdr3 > thresholds_dict[f"nat_{length}"]["precision_cdr"]

        # Test full phylogenetic method
        dataframe_processed["split_up_cluster"] = dataframe_processed.groupby(
            ["v_gene", "j_gene", "cdr3_length"]
        ).ngroup()
        cluster_sizes = dataframe_processed.groupby("split_up_cluster").size()
        dataframe_processed["VJL_class_size"] = dataframe_processed["split_up_cluster"].map(
            cluster_sizes
        )

        limit = 20000
        if dataframe_processed["VJL_class_size"].max() > limit:
            apriori_big = Apriori(
                paired=False,
                threads=48,
                precision=1,
                sensitivity=0.99,
                model="human_B_heavy",
                silent=False,
            )
            dataframe_big_vjl = dataframe_processed.query("VJL_class_size>@limit")
            apriori_big.classes = create_classes(dataframe_big_vjl)
            apriori_big.get_histograms(dataframe_big_vjl)
            apriori_big.get_parameters()
            apriori_big.classes["threshold"] = apriori_big.classes["sensitive_threshold"]
            clustering_big = CDR3Clustering(thresholds=apriori_big.classes, threads=48)
            dataframe_big_vjl["split_up_cluster"] = clustering_big.infer(
                dataframe_big_vjl, silent=False
            )
            dataframe_processed = pd.concat(
                [dataframe_processed.query("VJL_class_size<=@limit"), dataframe_big_vjl]
            ).sort_index()

        hilary = HILARy(
            df=dataframe_processed,
            paired=False,
            threads=48,
            silent=False,
        )
        hilary.get_xy_thresholds(df=dataframe_processed)
        dataframe_inferred = hilary.infer(df=dataframe_processed)
        dataframe["clone_id"] = dataframe_inferred["clone_id"]
        precision_full, sensitivity_full = pairwise_evaluation(df=dataframe, partition="clone_id")
        print(length, precision_full, sensitivity_full)
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
    """Test HILARy methods on PARTIS data."""
    for mut in ["05", "20"]:
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

        # Test crude method with normalized threshold
        dataframe_processed = preprocess(df=dataframe, df_light=None, threads=48)
        classes = create_classes(dataframe_processed)
        normalized_threshold = 0.2
        classes["threshold"] = (classes["cdr3_length_value"] * normalized_threshold).astype(int)
        clustering_crude = CDR3Clustering(thresholds=classes, threads=48)
        dataframe["crude_clone_id"] = clustering_crude.infer(dataframe_processed, silent=False)
        precision_crude, sensitivity_crude = pairwise_evaluation(
            df=dataframe, partition="crude_clone_id"
        )
        assert precision_crude > thresholds_dict["crude"]["precision"]
        assert sensitivity_crude > thresholds_dict["crude"]["sensitivity"]

        # Test CDR3 method
        apriori = Apriori(
            silent=False,
            threads=48,
            precision=hilary_pars["precision"],
            sensitivity=hilary_pars["sensitivity"],
        )
        apriori.classes = create_classes(dataframe_processed)
        apriori.get_histograms(dataframe_processed)
        apriori.get_parameters()
        apriori.classes["threshold"] = apriori.classes["precise_threshold"]
        clustering_cdr3 = CDR3Clustering(thresholds=apriori.classes, threads=48)
        dataframe["cdr3_based_family"] = clustering_cdr3.infer(dataframe_processed, silent=False)
        precision_cdr3, sensitivity_cdr3 = pairwise_evaluation(
            df=dataframe, partition="cdr3_based_family"
        )
        assert precision_cdr3 > thresholds_dict[f"partis_single_{mut}"]["precision_cdr"]

        # Test full phylogenetic method
        dataframe_processed["split_up_cluster"] = dataframe_processed.groupby(
            ["v_gene", "j_gene", "cdr3_length"]
        ).ngroup()
        cluster_sizes = dataframe_processed.groupby("split_up_cluster").size()
        dataframe_processed["VJL_class_size"] = dataframe_processed["split_up_cluster"].map(
            cluster_sizes
        )

        limit = 20000
        if dataframe_processed["VJL_class_size"].max() > limit:
            apriori_big = Apriori(
                paired=False,
                threads=48,
                precision=1,
                sensitivity=0.99,
                model="human_B_heavy",
                silent=False,
            )
            dataframe_big_vjl = dataframe_processed.query("VJL_class_size>@limit")
            apriori_big.classes = create_classes(dataframe_big_vjl)
            apriori_big.get_histograms(dataframe_big_vjl)
            apriori_big.get_parameters()
            apriori_big.classes["threshold"] = apriori_big.classes["sensitive_threshold"]
            clustering_big = CDR3Clustering(thresholds=apriori_big.classes, threads=48)
            dataframe_big_vjl["split_up_cluster"] = clustering_big.infer(
                dataframe_big_vjl, silent=False
            )
            dataframe_processed = pd.concat(
                [dataframe_processed.query("VJL_class_size<=@limit"), dataframe_big_vjl]
            ).sort_index()

        hilary = HILARy(
            df=dataframe_processed,
            paired=False,
            threads=48,
            silent=False,
        )
        hilary.get_xy_thresholds(df=dataframe_processed)
        dataframe_inferred = hilary.infer(df=dataframe_processed)
        dataframe["clone_id"] = dataframe_inferred["clone_id"]
        precision_full, sensitivity_full = pairwise_evaluation(df=dataframe, partition="clone_id")
        assert precision_full > thresholds_dict[f"partis_single_{mut}"]["precision_full"]
        assert sensitivity_full > thresholds_dict[f"partis_single_{mut}"]["sensitivity_full"]

        log.info(
            "Showing metrics for given file.",
            file=f"partis_{mut}/single_chain/igh.csv.gz",
            precision_crude=precision_crude,
            sensitivity_crude=sensitivity_crude,
            precision_cdr3=precision_cdr3,
            sensitivity_cdr3=sensitivity_cdr3,
            precision_full_method=precision_full,
            sensitivity_full_method=sensitivity_full,
        )


def check_performance_on_naive_mouse_data():
    """Test HILARy methods on naive mouse data."""
    log.info("Processing file.", file="generated_mouse_post_aligned_subsampled50K.csv.gz")
    dataframe = pd.read_csv(
        file_path / "generated_mouse_post_aligned_subsampled50K.csv.gz",
        compression="gzip",
    )
    dataframe["sequence_id"] = dataframe.index.astype("str")

    # Test crude method
    dataframe_processed = preprocess(df=dataframe, df_light=None, threads=48)
    classes = create_classes(dataframe_processed)
    normalized_threshold = 0.2
    classes["threshold"] = (classes["cdr3_length_value"] * normalized_threshold).astype(int)
    clustering_crude = CDR3Clustering(thresholds=classes, threads=48)
    dataframe["crude_clone_id"] = clustering_crude.infer(dataframe_processed, silent=False)
    precision_crude = len(dataframe["crude_clone_id"].unique()) / len(dataframe)
    assert precision_crude > thresholds_dict["crude"]["precision"]

    # Test CDR3 method
    apriori = Apriori(
        silent=False,
        threads=48,
        precision=hilary_pars["precision"],
        sensitivity=hilary_pars["sensitivity"],
        model="mouse_B_heavy",
    )
    apriori.classes = create_classes(dataframe_processed)
    apriori.get_histograms(dataframe_processed)
    apriori.get_parameters()
    apriori.classes["threshold"] = apriori.classes["precise_threshold"]
    clustering_cdr3 = CDR3Clustering(thresholds=apriori.classes, threads=48)
    dataframe["cdr3_based_family"] = clustering_cdr3.infer(dataframe_processed, silent=False)
    precision_cdr3 = len(dataframe["cdr3_based_family"].unique()) / len(dataframe)
    assert precision_cdr3 > thresholds_dict["naive_mouse"]["precision"]

    # Test full phylogenetic method
    dataframe_processed["split_up_cluster"] = dataframe_processed.groupby(
        ["v_gene", "j_gene", "cdr3_length"]
    ).ngroup()
    cluster_sizes = dataframe_processed.groupby("split_up_cluster").size()
    dataframe_processed["VJL_class_size"] = dataframe_processed["split_up_cluster"].map(
        cluster_sizes
    )

    limit = 20000
    if dataframe_processed["VJL_class_size"].max() > limit:
        apriori_big = Apriori(
            paired=False,
            threads=48,
            precision=1,
            sensitivity=0.99,
            model="mouse_B_heavy",
            silent=False,
        )
        dataframe_big_vjl = dataframe_processed.query("VJL_class_size>@limit")
        apriori_big.classes = create_classes(dataframe_big_vjl)
        apriori_big.get_histograms(dataframe_big_vjl)
        apriori_big.get_parameters()
        apriori_big.classes["threshold"] = apriori_big.classes["sensitive_threshold"]
        clustering_big = CDR3Clustering(thresholds=apriori_big.classes, threads=48)
        dataframe_big_vjl["split_up_cluster"] = clustering_big.infer(
            dataframe_big_vjl, silent=False
        )
        dataframe_processed = pd.concat(
            [dataframe_processed.query("VJL_class_size<=@limit"), dataframe_big_vjl]
        ).sort_index()

    hilary = HILARy(
        df=dataframe_processed,
        paired=False,
        threads=48,
        silent=False,
    )
    hilary.get_xy_thresholds(df=dataframe_processed)
    dataframe_inferred = hilary.infer(df=dataframe_processed)
    dataframe["clone_id"] = dataframe_inferred["clone_id"]
    precision_full = len(dataframe["clone_id"].unique()) / len(dataframe)
    assert precision_full > thresholds_dict["naive_mouse"]["precision"]

    log.info(
        "Showing metrics for given file.",
        file="generated_mouse_post_aligned_subsampled50K.csv.gz",
        precision_crude=precision_crude,
        precision_cdr3=precision_cdr3,
        precision_full_method=precision_full,
    )


def check_performance_on_naive_human_data():
    """Test HILARy methods on naive human data."""
    log.info("Processing file.", file="sonia_human_igh_aligned_subsampled50K.csv.gz")
    dataframe = pd.read_csv(
        file_path / "sonia_human_igh_aligned_subsampled50K.csv.gz",
        compression="gzip",
    )
    dataframe["sequence_id"] = dataframe.index.astype("str")
    dataframe = dataframe.dropna()
    # Test crude method
    dataframe_processed = preprocess(df=dataframe, df_light=None, threads=48)
    classes = create_classes(dataframe_processed)
    normalized_threshold = 0.2
    classes["threshold"] = (classes["cdr3_length_value"] * normalized_threshold).astype(int)
    clustering_crude = CDR3Clustering(thresholds=classes, threads=48)
    dataframe["crude_clone_id"] = clustering_crude.infer(dataframe_processed, silent=False)
    precision_crude = len(dataframe["crude_clone_id"].unique()) / len(dataframe)
    assert precision_crude > thresholds_dict["crude"]["precision"]

    # Test CDR3 method
    apriori = Apriori(
        silent=False,
        threads=48,
        precision=hilary_pars["precision"],
        sensitivity=hilary_pars["sensitivity"],
    )
    apriori.classes = create_classes(dataframe_processed)
    apriori.get_histograms(dataframe_processed)
    apriori.get_parameters()
    apriori.classes["threshold"] = apriori.classes["precise_threshold"]
    clustering_cdr3 = CDR3Clustering(thresholds=apriori.classes, threads=48)
    dataframe["cdr3_based_family"] = clustering_cdr3.infer(dataframe_processed, silent=False)
    precision_cdr3 = len(dataframe["cdr3_based_family"].unique()) / len(dataframe)
    assert precision_cdr3 > thresholds_dict["naive_human"]["precision"]

    # Test full phylogenetic method
    dataframe_processed["split_up_cluster"] = dataframe_processed.groupby(
        ["v_gene", "j_gene", "cdr3_length"]
    ).ngroup()
    cluster_sizes = dataframe_processed.groupby("split_up_cluster").size()
    dataframe_processed["VJL_class_size"] = dataframe_processed["split_up_cluster"].map(
        cluster_sizes
    )

    limit = 20000
    if dataframe_processed["VJL_class_size"].max() > limit:
        apriori_big = Apriori(
            paired=False,
            threads=48,
            precision=1,
            sensitivity=0.99,
            model="human_B_heavy",
            silent=False,
        )
        dataframe_big_vjl = dataframe_processed.query("VJL_class_size>@limit")
        apriori_big.classes = create_classes(dataframe_big_vjl)
        apriori_big.get_histograms(dataframe_big_vjl)
        apriori_big.get_parameters()
        apriori_big.classes["threshold"] = apriori_big.classes["sensitive_threshold"]
        clustering_big = CDR3Clustering(thresholds=apriori_big.classes, threads=48)
        dataframe_big_vjl["split_up_cluster"] = clustering_big.infer(
            dataframe_big_vjl, silent=False
        )
        dataframe_processed = pd.concat(
            [dataframe_processed.query("VJL_class_size<=@limit"), dataframe_big_vjl]
        ).sort_index()

    hilary = HILARy(
        df=dataframe_processed,
        paired=False,
        threads=48,
        silent=False,
    )
    hilary.get_xy_thresholds(df=dataframe_processed)
    dataframe_inferred = hilary.infer(df=dataframe_processed)
    dataframe["clone_id"] = dataframe_inferred["clone_id"]
    precision_full = len(dataframe["clone_id"].unique()) / len(dataframe)
    assert precision_full > thresholds_dict["naive_human"]["precision"]

    log.info(
        "Showing metrics for given file.",
        file="sonia_human_igh_aligned_subsampled50K.csv.gz",
        precision_crude=precision_crude,
        precision_cdr3=precision_cdr3,
        precision_full_method=precision_full,
    )


def check_performance_on_partis_paired_data():
    """Test HILARy methods on PARTIS paired data."""
    for mut in ["05", "20"]:
        log.info("Processing file.", file=f"partis_{mut}/both_chains/igh.csv.gz")
        dataframe = pd.read_csv(
            file_path / f"partis_{mut}/both_chains/igh.csv.gz",
            compression="gzip",
        )
        dataframe_light = pd.read_csv(
            file_path / f"partis_{mut}/both_chains/igk.csv.gz",
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
        dataframe_light = dataframe_light.rename(
            columns={
                "v_gl_seq": "v_germline_alignment",
                "v_qr_seqs": "v_sequence_alignment",
                "j_gl_seq": "j_germline_alignment",
                "j_qr_seqs": "j_sequence_alignment",
                "clone_id": "ground_truth",
            }
        )
        dataframe["sequence_id"] = dataframe.index.astype("str")

        # Test crude method
        dataframe_processed = preprocess(df=dataframe, df_light=dataframe_light, threads=48)
        classes = create_classes(dataframe_processed)
        normalized_threshold = 0.2
        classes["threshold"] = (classes["cdr3_length_value"] * normalized_threshold).astype(int)
        clustering_crude = CDR3Clustering(thresholds=classes, threads=48)
        dataframe["crude_clone_id"] = clustering_crude.infer(dataframe_processed, silent=False)
        precision_crude, sensitivity_crude = pairwise_evaluation(
            df=dataframe, partition="crude_clone_id"
        )
        assert precision_crude > thresholds_dict["crude"]["precision"]
        assert sensitivity_crude > thresholds_dict["crude"]["sensitivity"]

        # Test CDR3 method
        apriori = Apriori(
            silent=False,
            threads=48,
            precision=hilary_pars["precision"],
            sensitivity=hilary_pars["sensitivity"],
            model="human_paired",
            paired=True,
        )
        apriori.classes = create_classes(dataframe_processed)
        apriori.get_histograms(dataframe_processed)
        apriori.get_parameters()
        apriori.classes["threshold"] = apriori.classes["precise_threshold"]
        clustering_cdr3 = CDR3Clustering(thresholds=apriori.classes, threads=48)
        dataframe["cdr3_based_family"] = clustering_cdr3.infer(dataframe_processed, silent=False)
        precision_cdr3, sensitivity_cdr3 = pairwise_evaluation(
            df=dataframe, partition="cdr3_based_family"
        )
        assert precision_cdr3 > thresholds_dict[f"partis_single_{mut}"]["precision_cdr"]

        # Test full phylogenetic method
        dataframe_processed["split_up_cluster"] = dataframe_processed.groupby(
            ["v_gene", "j_gene", "cdr3_length"]
        ).ngroup()
        cluster_sizes = dataframe_processed.groupby("split_up_cluster").size()
        dataframe_processed["VJL_class_size"] = dataframe_processed["split_up_cluster"].map(
            cluster_sizes
        )

        limit = 20000
        if dataframe_processed["VJL_class_size"].max() > limit:
            apriori_big = Apriori(
                paired=True,
                threads=48,
                precision=1,
                sensitivity=0.99,
                model="human_paired",
                silent=False,
            )
            dataframe_big_vjl = dataframe_processed.query("VJL_class_size>@limit")
            apriori_big.classes = create_classes(dataframe_big_vjl)
            apriori_big.get_histograms(dataframe_big_vjl)
            apriori_big.get_parameters()
            apriori_big.classes["threshold"] = apriori_big.classes["sensitive_threshold"]
            clustering_big = CDR3Clustering(thresholds=apriori_big.classes, threads=48)
            dataframe_big_vjl["split_up_cluster"] = clustering_big.infer(
                dataframe_big_vjl, silent=False
            )
            dataframe_processed = pd.concat(
                [dataframe_processed.query("VJL_class_size<=@limit"), dataframe_big_vjl]
            ).sort_index()

        hilary = HILARy(
            df=dataframe_processed,
            paired=True,
            threads=48,
            silent=False,
        )
        hilary.get_xy_thresholds(df=dataframe_processed)
        # Add xy_threshold adjustment for paired data
        hilary.classes["xy_threshold"] = hilary.classes["xy_threshold"] + 4
        dataframe_inferred = hilary.infer(df=dataframe_processed)
        dataframe["clone_id"] = dataframe_inferred["clone_id"]
        precision_full, sensitivity_full = pairwise_evaluation(df=dataframe, partition="clone_id")
        assert precision_full > thresholds_dict[f"partis_single_{mut}"]["precision_full"]
        assert sensitivity_full > thresholds_dict[f"partis_single_{mut}"]["sensitivity_full"]

        log.info(
            "Showing metrics for given file.",
            file=f"partis_{mut}/both_chains/igh.csv.gz",
            precision_crude=precision_crude,
            sensitivity_crude=sensitivity_crude,
            precision_cdr3=precision_cdr3,
            sensitivity_cdr3=sensitivity_cdr3,
            precision_full_method=precision_full,
            sensitivity_full_method=sensitivity_full,
        )


# Run all tests
if __name__ == "__main__":
    check_performance_on_partis_data()
    check_performance_on_naive_human_data()
    check_performance_on_partis_paired_data()
    check_performance_on_naive_mouse_data()
    check_performance_on_nat_data()
