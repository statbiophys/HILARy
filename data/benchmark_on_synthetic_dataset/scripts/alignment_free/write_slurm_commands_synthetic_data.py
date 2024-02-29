from pathlib import Path

current_folder = "/home/gathenes/benchmark-missing-CDR3-length-15"
with open("commands_synthetic.txt", "w") as f:
    for seed in range(2, 6):
        for l in range(18, 45 + 3, 3):
            simulations = Path(
                f"{current_folder}/subsampled_benchmark/families_cdr3l_{l}_1e5_set_{seed}.csv.gz"
            )
            result_file = Path(
                f"{current_folder}/alignment_free/seed-{seed}/families_cdr3l_{l}_1e5_set_{seed}.csv"
            )
            data_negative = Path(
                f"/home/gathenes/generated_seqs-20240220T164400Z-001/generated_seqs/ppost_326651_cdr3l_{l}.csv.gz"
            )
            result_file.parents[0].mkdir(exist_ok=True, parents=True)
            f.write(
                f"python -m run_alignment_free_synthetic_data {simulations} {result_file} {data_negative} \n"
            )
