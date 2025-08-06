"""Optimized clonal family inference implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import jit, prange

pd.options.mode.chained_assignment = None  # default='warn'


@jit(nopython=True, cache=True)
def levenshtein_bytes_fast(a: np.ndarray, b: np.ndarray) -> int:
    """Levenshtein distance using dynamic programming with Numba JIT."""
    len_a, len_b = len(a), len(b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=np.int32)

    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[len_a][len_b]


@jit(nopython=True, fastmath=True, cache=True)
def hamming_bytes_fast(a: np.ndarray, b: np.ndarray) -> int:
    """Optimized hamming distance calculation."""
    return np.sum(a != b)


@jit(nopython=True, fastmath=True, cache=True)
def compute_metric_fast(
    cdr31: np.ndarray,
    cdr32: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    n1: int,
    n2: int,
    l_L: float,
    l_L_L: float,
    L: int,
) -> float:
    """Optimized metric computation."""
    if n1 * n2 == 0:
        return L
    n = hamming_bytes_fast(cdr31, cdr32)
    nl = hamming_bytes_fast(s1, s2)
    n0 = (n1 + n2 - nl) / 2

    exp_n = l_L * (nl + 1)
    std_n = np.sqrt(exp_n * l_L_L)

    exp_n0 = n1 * n2 / L
    std_n0 = np.sqrt(exp_n0)

    x = (n - exp_n) / std_n
    y = (n0 - exp_n0) / std_n0
    return x - y


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_distance_matrix_fast(
    cdr3_array: np.ndarray,
    align_array: np.ndarray,
    mut_array: np.ndarray,
    l_L: float,
    l_L_L: float,
    L: int,
) -> np.ndarray:
    """Optimized distance matrix computation using parallel processing."""
    n = cdr3_array.shape[0]
    distances = np.zeros(n * (n - 1) // 2, dtype=np.float64)

    for i in prange(n):
        for j in range(i + 1, n):
            idx = i * n - i * (i + 1) // 2 + j - i - 1
            distances[idx] = compute_metric_fast(
                cdr3_array[i],
                cdr3_array[j],
                align_array[i],
                align_array[j],
                mut_array[i],
                mut_array[j],
                l_L,
                l_L_L,
                L,
            )

    return distances + L


class DistanceMatrix:
    """Optimized distance matrix computation with better memory management."""

    def __init__(
        self, cdr3_l: int, alignment_length: int, df: pd.DataFrame, threads: int = 1
    ) -> None:
        self.threads = threads
        self.l = cdr3_l
        self.L = alignment_length
        self.l_L = cdr3_l / alignment_length
        self.l_L_L = (cdr3_l + alignment_length) / alignment_length
        self._precompute_byte_arrays(df)
        self.cdr3 = np.stack(df["cdr3_bytes"].to_numpy())
        self.align = np.stack(df["align_bytes"].to_numpy())

        self.mut = df["mutation_count"].to_numpy()
        self.n = self.cdr3.shape[0]

    def compute(self) -> np.ndarray:
        """Compute distance matrix using optimized numba functions."""
        if self.n <= 1:
            return np.array([])

        return compute_distance_matrix_fast(
            self.cdr3, self.align, self.mut, self.l_L, self.l_L_L, self.L
        )

    def _precompute_byte_arrays(self, df: pd.DataFrame) -> None:
        """Pre-compute and pad byte arrays for all sequences."""
        df["cdr3_padded"] = df["cdr3"].str.pad(
            df["cdr3"].str.len().max(), side="right", fillchar="-"
        )
        df["alt_sequence_alignment_padded"] = df["alt_sequence_alignment"].str.pad(
            df["alt_sequence_alignment"].str.len().max(), side="right", fillchar="-"
        )
        df["cdr3_bytes"] = df["cdr3_padded"].apply(
            lambda x: np.frombuffer(x.encode("utf-8"), dtype=np.uint8)
        )
        df["align_bytes"] = df["alt_sequence_alignment_padded"].apply(
            lambda x: np.frombuffer(x.encode("utf-8"), dtype=np.uint8)
        )
