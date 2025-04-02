# Based on the code from Natanael Spisak

import pandas as pd

pd.options.mode.chained_assignment = None
import numpy as np
from tqdm import tqdm

from hilary.utils import applyParallel
from hilary.utils_simulate import mutations2Counter, mutations2Spectrum, mutationsInFamily


class Spectra:
    """
    A class to represent and compute mutation spectra for clonal families.

    Attributes
    ----------
    individuals : list
        A list of individuals to process.
    threshold : int, optional
        The threshold family size to distinguish between small and large spectra (default is 10).
    ls : list, optional
        A list of values to iterate over (default is np.arange(45,81,3)).
    inputMacro : str, optional
        A string template for input file paths.
    outputMacro : str, optional
        A string template for output file paths.

    Methods
    -------
    computeSmall(args, verbatim=False):
        Computes mutation spectra for small families.
    computeLarge(args, verbatim=False):
        Computes mutation spectra for large families.
    compute():
        Computes both small and large spectra for all individuals.
    save():
        Saves the computed small and large spectra to files.
    read():
        Reads the saved small and large spectra from files.
    """

    def __init__(self, threshold=10, ls=None):
        self.threshold = threshold
        if ls is None:
            self.ls = np.arange(45, 81, 3)
        else:
            self.ls = ls

    def compute(self, dataframe: str | pd.DataFrame, family_column: str = "FAMILY"):
        """
        Compute small and large spectra in the file for each value in self.ls.
        This method reads data from the input file, processes the data in parallel to compute
        small and large spectra, and then concatenates the results into two DataFrames:
        self.small and self.large. The results are sorted by 'family_size' in ascending order.
        The input files are expected to be tab-separated and their paths are formatted using
        self.inputMacro with the individual and value from self.ls.
        The method performs the following steps:
        1. Initializes empty DataFrames for small and large spectra.
        2. Iterates over each individual and each value in self.ls.
        3. Reads the input file corresponding to the current individual and value.
        4. Computes the small spectrum in parallel
        5. Concatenates the result to the small DataFrame.
        6. Computes the large spectrum in parallel
        7. Concatenates the result to the large DataFrame.
        8. Sorts the small and large DataFrames by 'family_size' in ascending order.
        Attributes:
            self.small (pd.DataFrame): DataFrame containing the computed small spectra.
            self.large (pd.DataFrame): DataFrame containing the computed large spectra.
        """
        self.small = pd.DataFrame()
        self.large = pd.DataFrame()
        if isinstance(dataframe, str):
            dataframe = pd.read_table(dataframe)
        self.family_column = family_column
        dataframe["cdr3_length"] = dataframe["cdr3"].apply(len)
        for cdr3_length in self.ls:
            df = dataframe.loc[dataframe["cdr3_length"] == cdr3_length]
            self.local = applyParallel(df.groupby(df[self.family_column] % 64), self.computeSmall)
            self.small = pd.concat([self.small, self.local], ignore_index=True)
            self.local = applyParallel(df.groupby(df[self.family_column] % 64), self.computeLarge)
            self.large = pd.concat([self.large, self.local], ignore_index=True)
        self.small = self.small.sort_values(
            by="family_size", ascending=True, inplace=False, ignore_index=True
        )
        self.large = self.large.sort_values(
            by="family_size", ascending=True, inplace=False, ignore_index=True
        )

    def computeSmall(self, args, verbatim=False):
        """
        Compute mutation spectra for small clonal families.
        Args:
            args (tuple): A tuple containing an index and a DataFrame with mutation data.
            verbatim (bool, optional): If True, enables verbose output. Defaults to False.
        Returns:
            pd.DataFrame: A DataFrame with columns 'family_size' and 'mutation_spectrum'.
                          'family_size' is the size of the clonal family.
                          'mutation_spectrum' is a string representation of the mutation spectrum.
        """

        _, df = args
        spectra = [[] for _ in range(self.threshold)]
        for f, fdf in tqdm(df.groupby([self.family_column]), disable=~verbatim):
            size = len(fdf)
            max_nb = int(1e9)
            if size <= self.threshold and len(spectra[size - 1]) < max_nb:
                spectrum = mutations2Spectrum(mutationsInFamily(fdf))
                spectra[size - 1].append(":".join(spectrum.astype(str)))
        result = pd.DataFrame()
        for i, local_spectra in enumerate(spectra):
            size = i + 1
            local_result = pd.DataFrame()
            local_result["mutation_spectrum"] = local_spectra
            local_result["family_size"] = size
            result = pd.concat([result, local_result], ignore_index=True)
        return result[["family_size", "mutation_spectrum"]].astype(
            {"family_size": int, "mutation_spectrum": str}
        )

    def computeLarge(self, args, verbatim=False):
        """
        Computes the configurations and number of mutations for large families in the given DataFrame.
        Args:
            args (tuple): A tuple containing an unused element and a pandas DataFrame with family data.
            verbatim (bool, optional): If True, enables verbose output. Defaults to False.
        Returns:
            pd.DataFrame: A DataFrame with columns 'FAMILY_SIZE', 'CONFIGURATIONS', and 'NUMBERS_OF_MUTATIONS',
                          where 'FAMILY_SIZE' is the size of each family, 'CONFIGURATIONS' is a string representation
                          of mutation configurations, and 'NUMBERS_OF_MUTATIONS' is a string representation of the
                          number of mutations for each configuration.
        """
        _, df = args
        sizes = []
        counter_keys = []
        counter_counts = []
        for f, fdf in tqdm(df.groupby([self.family_column]), disable=~verbatim):
            size = len(fdf)
            if size > self.threshold:
                sizes.append(size)
                counter = mutations2Counter(mutationsInFamily(fdf))
                counter_keys.append(":".join(counter[:, 0].astype(str)))
                counter_counts.append(":".join(counter[:, 1].astype(str)))
        result = pd.DataFrame()
        result["family_size"] = sizes
        result["configurations"] = counter_keys
        result["number_of_mutations"] = counter_counts
        return result.astype(
            {"family_size": int, "configurations": str, "number_of_mutations": str}
        )

    def save(self, directory=""):
        """Save small and large spectra for individuals"""
        self.small.to_csv(directory + "small_spectra.csv.gz", index=False, compression="gzip")
        self.large.to_csv(directory + "large_spectra.csv.gz", index=False, compression="gzip")

    def read(self, directory=""):
        """Read small and large spectra for individuals"""
        self.small = pd.read_csv(directory + "small_spectra.csv.gz")
        self.large = pd.read_csv(directory + "large_spectra.csv.gz")
