#!/usr/bin/env python
from collections import Counter

import numpy as np


def bin2int(array: np.array) -> int:
    return int("".join(array.astype(str)), 2)


def bin2str(array: np.array) -> str:
    return "".join(array.astype(str))


switcher = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


def whereMutations(parent: str, child: str):
    """
    Identifies the positions and types of nucleotide substitutions between a parent and child sequence.
    Returns an array, indexed by position and nt type, 500x4
    Equals 1 if given substitution is present in child

    Args:
        parent (str): The parent nucleotide sequence.
        child (str): The child nucleotide sequence.

    Returns:
        np.ndarray: A 500x4 boolean array where each row represents a position in the sequence and each column represents a nucleotide type (A, C, G, T). The value is True if the corresponding substitution is present in the child sequence, otherwise False.

    Notes:
        - The function assumes that the sequences are of equal length.
        - Positions with 'N' or '.' in the child sequence and 'N' in the parent sequence are ignored.
    """
    indicate = np.zeros((500, 4), dtype=bool)
    for x, (nt, old_nt) in enumerate(zip(child, parent)):
        if nt != "N" and nt != "." and old_nt != "N":
            if nt != old_nt:
                indicate[x, switcher.get(nt)] = 1
    return indicate


def mutationsInFamily(df):
    """whereMutations across a dataframe of related sequences"""
    indicators = []
    for pair in df[["alt_germline_alignment", "alt_sequence_alignment"]].values:
        indicators.append(whereMutations(*pair))
    return np.array(indicators)


def mutations2Spectrum(indicators):
    """Input: mutations localizations and types
    Returns binary-indexed histogram"""
    spectrum = np.zeros(2 ** indicators.shape[0] - 1, dtype=int)
    for column in indicators.astype(int).T:
        mask = column.sum(axis=1).nonzero()[0]
        for row in column[mask]:
            spectrum[bin2int(row) - 1] += 1
    return spectrum


def mutations2Counter(indicators):
    """Input: mutations localizations and types
    Returns 01-indexed counter"""
    permutations = []
    for column in indicators.astype(int).T:
        mask = column.sum(axis=1).nonzero()[0]
        permutations.extend([bin2str(_) for _ in column[mask]])
    return np.array(Counter(permutations).most_common())


mutate2 = {"A": ["C", "G", "T"], "C": ["A", "G", "T"], "G": ["A", "C", "T"], "T": ["A", "C", "G"]}


def mutate(sequence, x, to=None):
    if to == None:
        to = np.random.choice(mutate2.get(sequence[x]))
    return "".join([sequence[:x], to, sequence[x + 1 :]])


def getCDR3(args):
    sequence, anchor, l = args
    return sequence[anchor + 3 : anchor + 3 + int(l)]


def mutateNbOfTimes(args):
    sequence, nb_of_times = args
    for x in np.random.randint(len(sequence), size=nb_of_times):
        sequence = mutate(sequence, x)
    return sequence


def nt2aa(ntseq):
    """Translate a nucleotide sequence into an amino acid sequence.

    Parameters
    ----------
    ntseq : str
        Nucleotide sequence composed of A, C, G, or T (uppercase or lowercase)

    Returns
    -------
    aaseq : str
        Amino acid sequence

    Example
    --------
    >>> nt2aa('TGTGCCTGGAGTGTAGCTCCGGACAGGGGTGGCTACACCTTC')
    'CAWSVAPDRGGYTF'

    """
    nt2num = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}
    aa_dict = "KQE*TPASRRG*ILVLNHDYTPASSRGCILVFKQE*TPASRRGWMLVLNHDYTPASSRGCILVF"

    return "".join(
        [
            aa_dict[nt2num[ntseq[i]] + 4 * nt2num[ntseq[i + 1]] + 16 * nt2num[ntseq[i + 2]]]
            for i in range(0, len(ntseq), 3)
            if i + 2 < len(ntseq)
        ]
    )
