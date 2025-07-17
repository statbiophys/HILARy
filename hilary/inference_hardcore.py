"""Optimized clonal family inference implementation with memoization."""

from __future__ import annotations

from itertools import combinations
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from functools import lru_cache, wraps
import hashlib
import pickle

import numpy as np
import pandas as pd
import structlog
from atriegc import TrieNucl as Trie
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from textdistance import hamming
from tqdm import tqdm
from fastcluster import linkage
import numba
from numba import jit, prange

from hilary.utils import apply_chunked_parallel, apply_parallel, p_required

log = structlog.get_logger()

NUM_RELIABLE_SEQ = 100
SUFFICIENT_MUT_NUM = 3

# Memoization utilities
class MemoizationManager:
    """Manages various caching strategies for performance optimization."""

    def __init__(self, max_cache_size: int = 100000):
        self.max_cache_size = max_cache_size
        self.distance_cache = {}
        self.hamming_cache = {}
        self.metric_cache = {}
        self.threshold_cache = {}
        self.simulation_cache = {}

    def clear_all_caches(self):
        """Clear all caches to free memory."""
        self.distance_cache.clear()
        self.hamming_cache.clear()
        self.metric_cache.clear()
        self.threshold_cache.clear()
        self.simulation_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cache usage."""
        return {
            'distance_cache': len(self.distance_cache),
            'hamming_cache': len(self.hamming_cache),
            'metric_cache': len(self.metric_cache),
            'threshold_cache': len(self.threshold_cache),
            'simulation_cache': len(self.simulation_cache)
        }

# Global memoization manager
_memo_manager = MemoizationManager()

def hash_sequence(seq: np.ndarray) -> str:
    """Create a hash for a sequence array for memoization."""
    return hashlib.md5(seq.tobytes()).hexdigest()

def hash_sequences(seq1: np.ndarray, seq2: np.ndarray) -> str:
    """Create a hash for a pair of sequences (order-independent)."""
    h1 = hash_sequence(seq1)
    h2 = hash_sequence(seq2)
    return f"{min(h1, h2)}_{max(h1, h2)}"

def memoize_hamming(func):
    """Decorator for memoizing hamming distance calculations."""
    @wraps(func)
    def wrapper(a: np.ndarray, b: np.ndarray) -> int:
        # Create cache key
        cache_key = hash_sequences(a, b)

        # Check cache
        if cache_key in _memo_manager.hamming_cache:
            return _memo_manager.hamming_cache[cache_key]

        # Compute and cache result
        result = func(a, b)
        if len(_memo_manager.hamming_cache) < _memo_manager.max_cache_size:
            _memo_manager.hamming_cache[cache_key] = result

        return result
    return wrapper

def memoize_metric(func):
    """Decorator for memoizing metric calculations."""
    @wraps(func)
    def wrapper(cdr31, cdr32, s1, s2, n1, n2, l_L, l_L_L, L):
        # Create cache key from sequence hashes and parameters
        seq_hash = hash_sequences(cdr31, cdr32) + hash_sequences(s1, s2)
        param_hash = f"{n1}_{n2}_{l_L:.6f}_{l_L_L:.6f}_{L}"
        cache_key = f"{seq_hash}_{param_hash}"

        # Check cache
        if cache_key in _memo_manager.metric_cache:
            return _memo_manager.metric_cache[cache_key]

        # Compute and cache result
        result = func(cdr31, cdr32, s1, s2, n1, n2, l_L, l_L_L, L)
        if len(_memo_manager.metric_cache) < _memo_manager.max_cache_size:
            _memo_manager.metric_cache[cache_key] = result

        return result
    return wrapper

# Memoized numba-optimized functions
@memoize_hamming
@jit(nopython=True, fastmath=True, cache=True)
def hamming_bytes_fast(a: np.ndarray, b: np.ndarray) -> int:
    """Optimized hamming distance calculation with memoization."""
    return np.sum(a != b)

# Note: For numba functions, we need a separate non-decorated version for the JIT
@jit(nopython=True, fastmath=True, cache=True)
def _hamming_bytes_jit(a: np.ndarray, b: np.ndarray) -> int:
    """JIT-compiled hamming distance for use in other JIT functions."""
    return np.sum(a != b)

@memoize_metric
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
    L: int
) -> float:
    """Optimized metric computation with memoization."""
    if n1 * n2 == 0:
        return L

    n = _hamming_bytes_jit(cdr31, cdr32)
    nl = _hamming_bytes_jit(s1, s2)
    n0 = (n1 + n2 - nl) / 2

    exp_n = l_L * (nl + 1)
    std_n = np.sqrt(exp_n * l_L_L)

    exp_n0 = n1 * n2 / L
    std_n0 = np.sqrt(exp_n0)

    x = (n - exp_n) / std_n
    y = (n0 - exp_n0) / std_n0

    return x - y

# For use in parallel JIT functions, we need non-memoized versions
@jit(nopython=True, fastmath=True, cache=True)
def _compute_metric_jit(
    cdr31: np.ndarray,
    cdr32: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    n1: int,
    n2: int,
    l_L: float,
    l_L_L: float,
    L: int
) -> float:
    """JIT-compiled metric computation for use in parallel functions."""
    if n1 * n2 == 0:
        return L

    n = _hamming_bytes_jit(cdr31, cdr32)
    nl = _hamming_bytes_jit(s1, s2)
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
    L: int
) -> np.ndarray:
    """Optimized distance matrix computation using parallel processing."""
    n = cdr3_array.shape[0]
    distances = np.zeros(n * (n - 1) // 2, dtype=np.float64)

    for i in prange(n):
        for j in range(i + 1, n):
            idx = i * n - i * (i + 1) // 2 + j - i - 1
            distances[idx] = _compute_metric_jit(
                cdr3_array[i], cdr3_array[j],
                align_array[i], align_array[j],
                mut_array[i], mut_array[j],
                l_L, l_L_L, L
            )

    return distances + L

class MemoizedDistanceMatrix:
    """Distance matrix with intelligent caching for repeated computations."""

    def __init__(self, cdr3_l: int, alignment_length: int, df: pd.DataFrame, threads: int = 1):
        self.cdr3_l = cdr3_l
        self.alignment_length = alignment_length
        self.threads = threads
        self.l_L = cdr3_l / alignment_length
        self.l_L_L = (cdr3_l + alignment_length) / alignment_length

        # Create matrix cache key
        self.matrix_key = self._create_matrix_cache_key(df)

        # Pre-validate and convert to numpy arrays
        cdr3_lengths = df["cdr3_bytes"].apply(len)
        if cdr3_lengths.nunique() > 1:
            raise ValueError("All CDR3 sequences must be the same length for stacking.")

        self.cdr3 = np.stack(df["cdr3_bytes"].to_numpy())

        align_lengths = df["align_bytes"].apply(len)
        if align_lengths.nunique() > 1:
            raise ValueError("All alignment sequences must be the same length.")

        self.align = np.stack(df["align_bytes"].to_numpy())
        self.mut = df["mutation_count"].to_numpy()
        self.n = self.cdr3.shape[0]

    def _create_matrix_cache_key(self, df: pd.DataFrame) -> str:
        """Create a unique key for the entire distance matrix."""
        # Hash the entire dataframe content for caching
        content = df[["cdr3_bytes", "align_bytes", "mutation_count"]].to_string()
        return hashlib.md5(content.encode()).hexdigest()

    def compute(self) -> np.ndarray:
        """Compute distance matrix with caching."""
        if self.n <= 1:
            return np.array([])

        # Check if we already computed this exact matrix
        cache_key = f"{self.matrix_key}_{self.cdr3_l}_{self.alignment_length}"
        if cache_key in _memo_manager.distance_cache:
            return _memo_manager.distance_cache[cache_key]

        # Compute the matrix
        distances = compute_distance_matrix_fast(
            self.cdr3, self.align, self.mut,
            self.l_L, self.l_L_L, self.alignment_length
        )

        # Cache the result if not too large
        if len(_memo_manager.distance_cache) < _memo_manager.max_cache_size // 10:  # Reserve cache space
            _memo_manager.distance_cache[cache_key] = distances

        return distances

class OptimizedDistanceMatrix:
    """Optimized distance matrix computation with better memory management."""

    def __init__(self, cdr3_l: int, alignment_length: int, df: pd.DataFrame, threads: int = 1) -> None:
        self.threads = threads
        self.l = cdr3_l
        self.L = alignment_length
        self.l_L = cdr3_l / alignment_length
        self.l_L_L = (cdr3_l + alignment_length) / alignment_length

        # Pre-validate and convert to numpy arrays
        cdr3_lengths = df["cdr3_bytes"].apply(len)
        if cdr3_lengths.nunique() > 1:
            raise ValueError("All CDR3 sequences must be the same length for stacking.")

        self.cdr3 = np.stack(df["cdr3_bytes"].to_numpy())

        align_lengths = df["align_bytes"].apply(len)
        if align_lengths.nunique() > 1:
            raise ValueError("All alignment sequences must be the same length.")

        self.align = np.stack(df["align_bytes"].to_numpy())
        self.mut = df["mutation_count"].to_numpy()
        self.n = self.cdr3.shape[0]

    def compute(self) -> np.ndarray:
        """Compute distance matrix using optimized numba functions."""
        if self.n <= 1:
            return np.array([])

        return compute_distance_matrix_fast(
            self.cdr3, self.align, self.mut,
            self.l_L, self.l_L_L, self.L
        )

class OptimizedHILARy:
    """Optimized version of HILARy with comprehensive memoization."""

    def __init__(
        self,
        df: pd.DataFrame,
        classes: pd.DataFrame,
        threads: int = 1,
        *,
        silent: bool = False,
        paired: bool = False,
        chunk_size: int = 10000,
        enable_memoization: bool = True,
        max_cache_size: int = 100000
    ) -> None:
        self.group = ["v_gene", "j_gene", "cdr3_length"]
        self.use = ["cdr3", "alt_sequence_alignment", "mutation_count", "index"]
        self.alignment_length = len(df["alt_sequence_alignment"].values[0])
        self.threads = threads if threads > 0 else cpu_count()
        self.silent = silent
        self.paired = paired
        self.classes = classes.copy()
        self.chunk_size = chunk_size
        self.enable_memoization = enable_memoization

        # Initialize memoization
        if enable_memoization:
            global _memo_manager
            _memo_manager.max_cache_size = max_cache_size
            if not silent:
                log.info(f"Memoization enabled with max cache size: {max_cache_size}")

        # Pre-compute cdr3_length_value with caching
        self._precompute_length_values()

        self.classes.index = self.classes.class_id

        # Pre-compute byte arrays to avoid repeated conversion
        self._precompute_byte_arrays(df)

    def _precompute_length_values(self) -> None:
        """Pre-compute CDR3 length values with caching."""
        cache_key = "cdr3_length_values"

        if self.enable_memoization and cache_key in _memo_manager.threshold_cache:
            self.classes["cdr3_length_value"] = _memo_manager.threshold_cache[cache_key]
            return

        if self.paired:
            length_values = self.classes.cdr3_length.apply(
                lambda x: int(x.split(",")[0]) + int(x.split(",")[1])
            )
        else:
            length_values = self.classes.cdr3_length.astype(int)

        self.classes["cdr3_length_value"] = length_values

        if self.enable_memoization:
            _memo_manager.threshold_cache[cache_key] = length_values

    def _precompute_byte_arrays(self, df: pd.DataFrame) -> None:
        """Pre-compute byte arrays for all sequences with caching."""
        if not self.silent:
            log.info("Pre-computing byte arrays for sequences...")

        # Check if we can reuse cached byte arrays
        if self.enable_memoization:
            seq_hash = hashlib.md5(df["cdr3"].str.cat().encode()).hexdigest()
            align_hash = hashlib.md5(df["alt_sequence_alignment"].str.cat().encode()).hexdigest()

            cdr3_cache_key = f"cdr3_bytes_{seq_hash}"
            align_cache_key = f"align_bytes_{align_hash}"

            if (cdr3_cache_key in _memo_manager.threshold_cache and
                align_cache_key in _memo_manager.threshold_cache):
                df["cdr3_bytes"] = _memo_manager.threshold_cache[cdr3_cache_key]
                df["align_bytes"] = _memo_manager.threshold_cache[align_cache_key]
                return

        # Vectorized byte array conversion
        cdr3_bytes = df["cdr3"].apply(
            lambda x: np.frombuffer(x.encode("utf-8"), dtype=np.uint8)
        )
        align_bytes = df["alt_sequence_alignment"].apply(
            lambda x: np.frombuffer(x.encode("utf-8"), dtype=np.uint8)
        )

        df["cdr3_bytes"] = cdr3_bytes
        df["align_bytes"] = align_bytes

        # Cache the results
        if self.enable_memoization:
            _memo_manager.threshold_cache[cdr3_cache_key] = cdr3_bytes
            _memo_manager.threshold_cache[align_cache_key] = align_bytes

    @lru_cache(maxsize=1000)
    def _cached_simulate_xs_ys(self, mutations_tuple: Tuple, alignment_length: int,
                               cdr3_length: int, class_id: int) -> Tuple[float, int]:
        """Cached version of simulation for threshold computation."""
        return self._simulate_xs_ys_impl(mutations_tuple, alignment_length, cdr3_length, class_id)

    def _simulate_xs_ys_impl(self, mutations_tuple: Tuple, alignment_length: int,
                            cdr3_length: int, class_id: int) -> Tuple[float, int]:
        """Implementation of simulation logic."""
        mutations = np.array(mutations_tuple)
        rng = np.random.default_rng(seed=42)
        size = int(1e6)

        if len(mutations) < NUM_RELIABLE_SEQ:
            return (0, class_id)

        # Optimized histogram computation
        max_mut = np.max(mutations)
        bins = np.arange(max_mut + 1)
        pni, nis = np.histogram(mutations, bins=bins)

        if len(nis) < SUFFICIENT_MUT_NUM:
            return (0, class_id)

        # Vectorized random sampling
        p = pni[1:] / np.sum(pni[1:])
        valid_indices = np.arange(1, len(nis) - 1)

        n1s = rng.choice(valid_indices, size=size, replace=True, p=p)
        n2s = rng.choice(valid_indices, size=size, replace=True, p=p)

        # Vectorized calculations
        exp_n0 = n1s * n2s / alignment_length
        n0s = rng.poisson(lam=exp_n0)
        std_n0 = np.sqrt(exp_n0)
        ys = (n0s - exp_n0) / std_n0

        ns = rng.choice(np.arange(cdr3_length + 1), size=size, replace=True)
        nls = np.maximum(n1s + n2s - 2 * n0s, 0)

        exp_n = (cdr3_length / alignment_length) * (nls + 1)
        std_n = np.sqrt(exp_n * (cdr3_length + alignment_length) / alignment_length)
        xs = (ns - exp_n) / std_n

        zs = xs - ys
        return (
            np.sort(zs)[min(int(size * p_required(0.2)), size - 1)],
            class_id,
        )

    def simulate_xs_ys(self, args) -> Tuple[float, int]:
        """Memoized simulation with better caching."""
        (_, _, _, mutations, alignment_length, class_id) = args

        classes_temp = self.classes.loc[self.classes.class_id == class_id]
        cdr3_length = classes_temp.cdr3_length_value.values[0]

        # Convert mutations to tuple for hashing
        mutations_tuple = tuple(sorted(mutations))

        if self.enable_memoization:
            return self._cached_simulate_xs_ys(mutations_tuple, alignment_length, cdr3_length, class_id)
        else:
            return self._simulate_xs_ys_impl(mutations_tuple, alignment_length, cdr3_length, class_id):
            return (0, class_id)

        # Vectorized random sampling
        p = pni[1:] / np.sum(pni[1:])
        valid_indices = np.arange(1, len(nis) - 1)

        n1s = rng.choice(valid_indices, size=size, replace=True, p=p)
        n2s = rng.choice(valid_indices, size=size, replace=True, p=p)

        # Vectorized calculations
        exp_n0 = n1s * n2s / alignment_length
        n0s = rng.poisson(lam=exp_n0)
        std_n0 = np.sqrt(exp_n0)
        ys = (n0s - exp_n0) / std_n0

        ns = rng.choice(np.arange(cdr3_length + 1), size=size, replace=True)
        nls = np.maximum(n1s + n2s - 2 * n0s, 0)

        exp_n = (cdr3_length / alignment_length) * (nls + 1)
        std_n = np.sqrt(exp_n * (cdr3_length + alignment_length) / alignment_length)
        xs = (ns - exp_n) / std_n

        zs = xs - ys
        return (
            np.sort(zs)[min(int(size * p_required(0.2)), size - 1)],
            class_id,
        )

    def get_xy_thresholds(self, df: pd.DataFrame) -> None:
        """Optimized threshold computation."""
        alignment_length = len(df["alt_sequence_alignment"].values[0])

        if not self.silent:
            log.debug("Computing xy_thresholds for each class...")

        self.classes["alignment_length"] = alignment_length

        # Optimized merge operation
        merge_cols = ["v_gene", "j_gene", "cdr3_length", "mutation_count"]
        class_cols = ["v_gene", "j_gene", "cdr3_length", "class_id", "alignment_length"]

        merged = df[merge_cols].merge(self.classes[class_cols], how='left')

        # Use more efficient groupby operations
        mutations_grouped = apply_chunked_parallel(
            merged.groupby("class_id"),
            group_mutations,
            cpu_count=self.threads,
            silent=self.silent,
        )

        # Parallel threshold computation
        result = apply_parallel(
            mutations_grouped.values,
            self.simulate_xs_ys,
            cpu_count=self.threads,
            silent=self.silent,
            isint=True,
        )

        thresholds_data = pd.DataFrame(result, columns=["xy_threshold", "class_id"]).set_index("class_id")
        self.classes["xy_threshold"] = thresholds_data["xy_threshold"]

    def single_linkage(self, indices: np.ndarray, dist: np.ndarray, threshold: float) -> Dict[int, int]:
        """Optimized single linkage clustering."""
        if len(indices) <= 1:
            return dict(zip(indices, indices))

        clusters = fcluster(
            linkage(dist, method="single", preserve_input=False),
            criterion="distance",
            t=threshold,
        )
        return dict(zip(indices, clusters))

    def class2pairs_optimized(self, args: Tuple[Tuple[str, str, str], pd.DataFrame]) -> pd.Series:
        """Optimized clustering for small groups."""
        df = args[1]
        v_gene, j_gene, cdr3_length = args[0]

        if self.paired:
            cdr3_length_value = int(cdr3_length.split(",")[0]) + int(cdr3_length.split(",")[1])
        else:
            cdr3_length_value = int(cdr3_length)

        xy_threshold = self.classes.query(
            "v_gene==@v_gene and j_gene==@j_gene and cdr3_length==@cdr3_length"
        )["xy_threshold"].values[0]

        indices = np.arange(len(df))
        df["index"] = indices

        if len(indices) <= 1:
            return df["index"]

        # Use optimized distance matrix computation
        try:
            dm = OptimizedDistanceMatrix(
                cdr3_l=cdr3_length_value,
                alignment_length=self.alignment_length,
                df=df[["cdr3_bytes", "align_bytes", "mutation_count"]],
                threads=1  # Use single thread for small groups
            )
            distances = dm.compute()

            sl = self.single_linkage(
                indices, distances, threshold=self.alignment_length + xy_threshold
            )
            return df["index"].map(sl)

        except Exception as e:
            log.warning(f"Falling back to original method for group {v_gene}_{j_gene}_{cdr3_length}: {e}")
            return self._fallback_class2pairs(args)

    def _fallback_class2pairs(self, args: Tuple[Tuple[str, str, str], pd.DataFrame]) -> pd.Series:
        """Fallback to original method if optimized version fails."""
        df = args[1]
        v_gene, j_gene, cdr3_length = args[0]

        if self.paired:
            cdr3_length_value = int(cdr3_length.split(",")[0]) + int(cdr3_length.split(",")[1])
        else:
            cdr3_length_value = int(cdr3_length)

        xy_threshold = self.classes.query(
            "v_gene==@v_gene and j_gene==@j_gene and cdr3_length==@cdr3_length"
        )["xy_threshold"].values[0]

        indices = np.arange(len(df))
        df["index"] = indices

        if len(indices) <= 1:
            return df["index"]

        dim = len(indices)
        distance_matrix = np.ones((dim, dim), dtype=float) * (2 * self.alignment_length)
        np.fill_diagonal(distance_matrix, 0)

        for (cdr31, s1, n1, i1), (cdr32, s2, n2, i2) in combinations(df[self.use].values, 2):
            if i1 != i2:
                n1n2 = n1 * n2
                if n1n2 > 0:
                    n = hamming(cdr31, cdr32)
                    nl = hamming(s1, s2)
                    n0 = (n1 + n2 - nl) / 2

                    exp_n = cdr3_length_value / self.alignment_length * (nl + 1)
                    std_n = np.sqrt(exp_n * (cdr3_length_value + self.alignment_length) / self.alignment_length)
                    exp_n0 = n1n2 / self.alignment_length
                    std_n0 = np.sqrt(exp_n0)

                    x = (n - exp_n) / std_n
                    y = (n0 - exp_n0) / std_n0
                    distance = x - y + self.alignment_length

                    distance_matrix[i1, i2] = min(distance, distance_matrix[i1, i2])
                    distance_matrix[i2, i1] = min(distance, distance_matrix[i2, i1])

        sl = self.single_linkage(
            indices, squareform(distance_matrix), threshold=self.alignment_length + xy_threshold
        )
        return df["index"].map(sl)

    def infer(self, df: pd.DataFrame, size_threshold: int = 1000) -> pd.DataFrame:
        """Optimized family cluster inference."""
        # Pre-compute byte arrays if not already done
        if "cdr3_bytes" not in df.columns:
            self._precompute_byte_arrays(df)

        # Group by class and separate small/large groups
        df_grouped = df.groupby(self.group)
        sizes = df_grouped.size()

        mask = sizes > size_threshold
        large_groups = sizes[mask].index
        small_groups = sizes[~mask].index

        if not self.silent:
            log.info(f"Processing {len(small_groups)} small groups and {len(large_groups)} large groups")

        # Initialize results
        df["family_cluster"] = np.nan
        df["index"] = df.index.values

        # Process small groups efficiently
        if len(small_groups) > 0:
            small_df = df[df[self.group].apply(tuple, axis=1).isin(small_groups)]

            family_clusters = apply_chunked_parallel(
                small_df.groupby(self.group),
                self.class2pairs_optimized,
                silent=self.silent,
                cpu_count=self.threads,
            )

            df.loc[small_df.index, "family_cluster"] = family_clusters

        # Process large groups with optimized distance computation
        if len(large_groups) > 0:
            large_df = df[df[self.group].apply(tuple, axis=1).isin(large_groups)]

            if self.paired:
                large_df["cdr3_length_value"] = large_df.cdr3_length.apply(
                    lambda x: int(x.split(",")[0]) + int(x.split(",")[1])
                )
            else:
                large_df["cdr3_length_value"] = large_df.cdr3_length.astype(int)

            # Process large groups with better memory management
            large_clusters = self._process_large_groups(large_df)

            for idx, cluster in large_clusters.items():
                df.loc[idx, "family_cluster"] = cluster

        # Fill NaN values and compute final clone IDs
        df["family_cluster"] = df["family_cluster"].fillna(0)
        df["clone_id"] = df.groupby([*self.group, "family_cluster"]).ngroup() + 1

        # Clean up temporary columns
        return df.drop(columns=["family_cluster", "cdr3_bytes", "align_bytes", "index"])

    def _process_large_groups(self, large_df: pd.DataFrame) -> Dict[int, int]:
        """Process large groups with optimized distance computation."""
        large_clusters = {}

        grouped_list = list(large_df.groupby(["v_gene", "j_gene", "cdr3_length", "cdr3_length_value"]))

        for g, grouped_df in tqdm(grouped_list, disable=self.silent):
            v_gene, j_gene, cdr3_length, cdr3_length_value = g

            xy_threshold = self.classes.query(
                "v_gene==@v_gene and j_gene==@j_gene and cdr3_length==@cdr3_length"
            )["xy_threshold"].values[0]

            try:
                dm = OptimizedDistanceMatrix(
                    cdr3_l=cdr3_length_value,
                    alignment_length=self.alignment_length,
                    df=grouped_df[["cdr3_bytes", "align_bytes", "mutation_count"]],
                    threads=self.threads,
                )
                distances = dm.compute()

                dct = self.single_linkage(
                    indices=grouped_df.index,
                    dist=distances,
                    threshold=self.alignment_length + xy_threshold,
                )

                large_clusters.update(dct)

            except Exception as e:
                log.warning(f"Error processing large group {g}: {e}")
                # Fallback: each sequence is its own cluster
                for idx in grouped_df.index:
                    large_clusters[idx] = idx

        return large_clusters

# Keep the original functions that are still needed
def group_mutations(args: Tuple[int, pd.DataFrame]) -> pd.DataFrame:
    """Get list of mutations for a given VJL class."""
    _, df = args
    v_gene, j_gene, cdr3_length, _, class_id, alignment_length = df.iloc[0]
    mutations = df["mutation_count"].values
    return pd.DataFrame([
        v_gene, j_gene, cdr3_length, mutations, alignment_length, class_id,
    ]).T
