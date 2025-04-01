import numpy as np
import righor as rg
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence
import polars as pl
from tqdm import tqdm
import pandas as pd

def pl_container_func_rg(seqs: Sequence[Sequence[str]]) -> pl.DataFrame:
    df = pl.DataFrame(
        seqs, orient="row", schema=["junction_aa", "v_gene", "j_gene", "junction"]
    ).with_columns(
        pl.col("v_gene").str.split("*").list[0], pl.col("j_gene").str.split("*").list[0]
    )
    return df
    
def generate_pgen_seqs_righor(
    model: str,
    num_monte_carlo: int,
    seed: int
    | np.random.Generator
    | np.random.BitGenerator
    | np.random.SeedSequence
    | None = None,
    available_v: Iterable | None = None,
    available_j: Iterable | None = None,
    container_func: Callable[Sequence[Sequence[str]], Any] | None = None,
    filter_func: Callable | pl.Expr | None = None,
    functional: bool = True,
    without_error: bool = True,
) -> pd.DataFrame:
    """
    Generate sequences from the recombination using CPU parallelization.

    Parameters
    ----------
    model : str or righor.Model or Sonia or SoNNia
        A string pointing to the directory containing the  model files or a Sonia/SoNNia model object.
    num_monte_carlo : int
        The number of sequences to generate.
    seed : int, numpy.random.Generator, numpy.random.BitGenerator, numpy.random.SeedSequence, optional
        The initial seed used to generate child seeds.
    available_v : iterable of str, optional
        The V genes used to produce recombinations.
        If None, there is no restriction on the V genes.
    available_j : iterable of str, optional
        The J genes used to produce recombinations.
        If None, there is no restriction on the J genes.
    container_func : callable, optional
        A function initializing the container for the resulting sequences.
        If None, a polars DataFrame is return.
    filter_func : callable or polars.Expr, optional
        Function which takes in the containerized generated sequences and
        filters them to a subset of the sequences. If a polars.Expr,
        a polars.Expr which returns True for sequences which are desired.
    functional : bool, default True
        If True, return sequences only from the productive repertoire.
    without_errors : bool, default True
        Generate sequences without introducing errors.

    Returns
    -------
    container of str
        Monte Carlo sequences produced from VDJ recombination. The default
        container is a polars.DataFrame.
    """

    if functional:
        model.v_segments = [v for v in model.v_segments if v.functional in {"F", "(F)"}]
        model.j_segments = [j for j in model.j_segments if j.functional in {"F", "(F)"}]

    if available_v is not None:
        available_v = set(available_v)
        available_v = [
            v for v in model.v_segments if v.name.partition("*")[0] in available_v
        ]
    if available_j is not None:
        available_j = set(available_j)
        available_j = [
            j for j in model.j_segments if j.name.partition("*")[0] in available_j
        ]

    if seed is not None and not isinstance(seed, int):
        seed = np.random.default_rng(seed).integers(2**32 - 1)

    generator = model.generator(seed, available_v, available_j)
    if without_error:
        #gen_seqs = generator.generate_many_without_errors(num_monte_carlo, functional)
        gen_seqs = [generator.generate_without_errors(functional) for _ in tqdm(range(num_monte_carlo))]
        gen_seqs = np.array([[s.cdr3_aa,s.v_gene,s.j_gene, s.cdr3_nt] for s in gen_seqs],dtype=object)
    else:
        gen_seqs = generator.generate(num_monte_carlo, functional)

    if container_func is None:
        container_func = pl_container_func_rg
    gen_seqs = container_func(gen_seqs)

    if filter_func is not None:
        if isinstance(filter_func, pl.Expr):
            if not isinstance(gen_seqs, pl.DataFrame):
                msg = (
                    "A polars Expression cannot be used if the sequences are not "
                    "contained in a polars.DataFrame"
                )
                raise TypeError(msg)
            gen_seqs = gen_seqs.filter(filter_func)
        else:
            gen_seqs = filter_func(gen_seqs)

    return gen_seqs.to_pandas()

def generate_ppost_seqs(
        sonia_model,
        righor_model,
        n_seqs: int = int(1e5),
        upper_bound:int = 10,
        available_v: Iterable | None = None,
        available_j: Iterable | None = None) -> pd.DataFrame:
    """Generate post-selection sequences using SONIA and RIGHOR models.
    
    Parameters:
    sonia_model: A trained SONIA model used to evaluate selection factors.
    righor_model: A trained RIGHOR model used to generate sequences.
    n_seqs (int): Number of sequences to generate. Default is 100,000.
    upper_bound (int): Upper bound for selection factor normalization. Default is 10.
    available_v (list, optional): List of available V genes. Default is None.
    available_j (list, optional): List of available J genes. Default is None.
    
    Returns:
    pd.DataFrame: A DataFrame containing the generated sequences after selection.
    """   
    seqs=generate_pgen_seqs_righor(righor_model,int(n_seqs*1.1*upper_bound),available_j=available_j,available_v=available_v)
    seqs['cdr3']=seqs['junction'].apply(lambda x: x[3:-3])
    seqs['cdr3_length']=seqs['cdr3'].apply(len)
    Qs=sonia_model.evaluate_selection_factors(seqs[['junction_aa','v_gene','j_gene']].values)
    random_samples = np.random.uniform(size=len(Qs))
    selection=random_samples < Qs / upper_bound
    seqs=seqs[selection].drop_duplicates()
    return seqs[:n_seqs].reset_index(drop=True)