import pandas as pd
import numpy as np
import warnings
from hilary.apriori import Apriori
from hilary.utils import create_classes
from hilary.inference import HILARy
import math 

class InferLineages:
    """
    Convert MiXCR output to HILARy input format and run lineage inference.

    This class allows users to process MiXCR output (from bulk or 10x sequencing)
    and run lineage inference using HILARy with minimal preprocessing steps.

    Required columns:
        - v_call
        - j_call
        - cdr3
        - sequence_alignment
        - germline_alignment

    Optional columns:
        - sequence_id (will be generated if missing or NaN)
        - junction_aa (used later for merging lineage features)

    Example usage:
    --------------
    from hilary.mixcr_formatter import InferLineages
    df = pd.read_csv("mixcr_output.tsv", sep="\t")
    infer = InferLineages()
    df_result = infer.infer(df)
    """
    
    def __init__(self, precision=0.99, sensitivity=0.9, threads=-1, silent=False):
        self.apriori = Apriori(
            precision=precision,
            sensitivity=sensitivity,
            threads=threads,
            silent=silent
        )

    def dash(self, a: str, cdr3: str, missing=math.nan):
        """Return (#before, #inside, #after, start_fake, len_cdr3, start_orig, end_orig_excl).
        If cdr3 is not found (after removing dashes), all values are NaN."""
        pos_dash = [i for i, ch in enumerate(a) if ch == '-']

        seq_fake = a.replace('-', '')
        start_fake = seq_fake.find(cdr3)
        if start_fake == -1:
            return (missing,) * 7

        L = len(cdr3)
        end_fake = start_fake + L - 1  # inclusive in fake coords

        before = inside = after = 0
        for seen, p in enumerate(pos_dash):
            adj = p - seen  # dash position in dash-less coordinates
            if adj <= start_fake:          # dash just before first base counts as "before"
                before += 1
            elif adj < end_fake:           # strictly inside the segment
                inside += 1
            else:
                after += 1

        # Convert back to original string indices
        start_orig = before + start_fake
        end_orig_excl = start_orig + inside + L  # exclusive slice end

        return before, inside, after, start_fake, L, start_orig, end_orig_excl
    
    def preprocess(self, df):
        # Define required and optional columns
        required_columns = [
            "v_call", "j_call", "cdr3",
            "sequence_alignment", "germline_alignment"
        ]
        optional_columns = ["sequence_id", "junction_aa"]

        # Warn about missing columns
        for col in required_columns:
            if col not in df.columns:
                warnings.warn(f"⚠️ Required column '{col}' is missing. Lineage inference may fail.")
        for col in optional_columns[1:]:  # Skip 'sequence_id' for this check
            if col not in df.columns:
                warnings.warn(f"ℹ️ Optional column '{col}' is missing — it will be skipped.")

        # Handle missing or NaN sequence_id
        if "sequence_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "sequence_id"})
            warnings.warn("ℹ️ 'sequence_id' column was missing — created from index.")
        elif df["sequence_id"].isna().any():
            df = df.rename(columns={"sequence_id": "sequence_id_old"})
            df = df.reset_index().rename(columns={"index": "sequence_id"})
            warnings.warn("⚠️ 'sequence_id' column contained NaNs — renamed to 'sequence_id_old' and replaced with new unique IDs.")

        # Select only existing relevant columns
        all_columns = required_columns + optional_columns
        existing_columns = [col for col in all_columns if col in df.columns]

        # Drop rows with NaN in any of the used columns
        df = df[existing_columns].dropna().reset_index(drop=True)

        # Remove rows where alignment lengths don't match
        if "sequence_alignment" in df.columns and "germline_alignment" in df.columns:
            df = df[df["sequence_alignment"].str.len() == df["germline_alignment"].str.len()].copy()

        # Remove sequences shorter than CDR3
        if "cdr3" in df.columns and "sequence_alignment" in df.columns:
            df = df[df["sequence_alignment"].str.len() > df["cdr3"].str.len()].copy()

        # Find position of CDR3 within sequence_alignment
        df["pos"] = np.char.find(df["sequence_alignment"].values.astype(str),
                                 df["cdr3"].values.astype(str))
        temp = df[df["pos"] == -1].copy()
        df = df[df["pos"] != -1].copy()
        #different approach if the cdr3 sequence is not found in the sequence due to "-"
        cols = ['d_before','d_inside','d_after','start_fake','len_cdr3','start','end']
        temp[cols] = temp.apply(lambda r: self.dash(r['sequence_alignment'], r['cdr3']), axis=1, result_type='expand')
        mask = temp['start'].notna()
        temp.loc[mask, 'alt_sequence_alignment'] = (
            temp.loc[mask].apply(
                lambda r: r['sequence_alignment'][:int(r['start'])] + r['sequence_alignment'][int(r['end']):], axis=1)
        )
        temp.loc[mask, 'alt_germline_alignment'] = (
            temp.loc[mask].apply(
                lambda r: r['germline_alignment'][:int(r['start'])] + r['germline_alignment'][int(r['end']):], axis=1)
        )
        temp.loc[~mask, ['alt_sequence_alignment','alt_germline_alignment']] = math.nan
        # Remove CDR3 from alignments efficiently
        df["alt_sequence_alignment"] = [
            s[:p] + s[p+len(c):] for s, p, c in zip(df["sequence_alignment"], df["pos"], df["cdr3"])
        ]
        df["alt_germline_alignment"] = [
            g[:p] + g[p+len(c):] for g, p, c in zip(df["germline_alignment"], df["pos"], df["cdr3"])
        ]
        temp.drop(columns=['d_before','d_inside','d_after','start_fake','len_cdr3','start','end'], inplace=True, errors='ignore')
        df = pd.concat([df, temp], ignore_index=False)
        # Drop sequences with ambiguous germline alignment
        df = df[~df["alt_germline_alignment"].str.contains("N")].reset_index(drop=True)
        return df

    def infer(self, df):
        df_preprocessed = self.preprocess(df)

        if df_preprocessed.empty:
            print("⚠️ No sequences passed preprocessing.")
            df["clone_id"] = np.nan
            return df

        # Apply HILARy inference
        df_processed = self.apriori.preprocess(df=df_preprocessed, df_kappa=None)
        self.apriori.classes = create_classes(df_processed)
        self.apriori.get_histograms(df_processed)
        self.apriori.get_parameters()
        self.apriori.get_thresholds()

        hilary = HILARy(self.apriori, df=df_processed)
        df_cdr3 = hilary.compute_prec_sens_clusters(df=df_processed)
        hilary.get_xy_thresholds(df=df_cdr3)
        hilary.classes["xy_threshold"] = hilary.classes["xy_threshold"]  # Optional overwrite
        df_inferred = hilary.infer(df=df_cdr3)

        # Compute mutation fraction
        df_inferred["mutation_fraction"] = (
            df_inferred["mutation_count"] / df_inferred["alt_germline_alignment"].str.len()
        )

        # Map results back to original DataFrame
        df_out = df.copy()
        df_out["clone_id"] = df_out["sequence_id"].map(
            df_inferred.set_index("sequence_id")["clone_id"].to_dict()
        )
        df_out["mutation_count"] = df_out["sequence_id"].map(
            df_inferred.set_index("sequence_id")["mutation_count"].to_dict()
        )
        df_out["mutation_fraction"] = df_out["sequence_id"].map(
            df_inferred.set_index("sequence_id")["mutation_fraction"].to_dict()
        )

        return df_out
