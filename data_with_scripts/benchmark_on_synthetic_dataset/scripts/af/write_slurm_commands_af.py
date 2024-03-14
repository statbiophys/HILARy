from pathlib import Path

current_folder = (
    "/home/gathenes/gitlab/HILARy/data_with_scripts/benchmark_on_synthetic_dataset/"
)

with open("commands_af.txt", "w") as f:
    for set in range(2, 6):
        for l in range(15, 45 + 3, 3):
            simulations = Path(
                f"{current_folder}subsampled_simulations/families{set}_1e4_ppost326651_mut326713_cdr3l{l}.csv.gz"
            )
            result_file = Path(
                f"{current_folder}/alignment_free/set-{set}/families{set}_1e4_ppost326651_mut326713_cdr3l{l}.csv"
            )
            data_negative = Path(
                f"/home/gathenes/generated_seqs-20240220T164400Z-001/generated_seqs/ppost_326651_cdr3l_{l}.csv.gz"
            )
            result_file.parents[0].mkdir(exist_ok=True, parents=True)
            f.write(
                f"python -m run_alignment_free_synthetic_data {simulations} {result_file} {data_negative} \n"
            )
