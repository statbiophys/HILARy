"""Code to fit prevalence and mu."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import factorial

# pylint: disable=invalid-name


class EM:
    """Finds the mixture distribution that explains the statistics of distances."""

    def __init__(
        self,
        cdfs: pd.DataFrame,
        l: float,
        h: np.ndarray,
        howmany: int = 10,
        positives: str = "poisson",
    ):
        """Initialize class.

        Args:
            cdfs (pd.DataFrame): P0 distribution.
            l (float): cdr3 length
            h (np.ndarray): histogram of pairwise distances.
            howmany (int, optional): Hopw many iterations to run for expectmax algo. Defaults to 10.
            positives (str, optional): Distribution to choose for P1. Defaults to "poisson".
        """
        self.l = int(l)
        self.h = h.astype(int)[: self.l + 1]
        self.b = np.arange(self.l + 1, dtype=int)
        self.cdfs = cdfs
        self.const_p0 = self.readNull()
        self.howmany = howmany
        self.positives = positives

    def readNull(self) -> np.ndarray:
        """Read estimated null distribution.

        Returns:
            np.ndarray: Histogram of null distributions (Ppost).
        """
        cdf = self.cdfs.loc[self.cdfs["l"] == self.l].values[0, 1 : self.l + 1]
        return np.diff(cdf, prepend=[0], append=[1])

    def discreteExpectation(self, theta: tuple[float, float]) -> tuple[float, float]:
        """Calculate membership probabilities.

        Args:
            theta (tuple(float, float)): (Prevalence,mu)
        Returns:
            tuple(float,float):P1 and P0 computed with updated prevalence and mu.
        """
        rho, mu = theta
        if self.positives == "geometric":
            P1 = rho / (mu + 1) * (mu / (mu + 1)) ** self.b
        elif self.positives == "poisson":
            P1 = rho * mu**self.b * np.exp(-mu) / factorial(self.b)
        P0 = (1 - rho) * self.const_p0[self.b]
        return np.array([P1, P0]) / (P1 + P0 + 1e-5)

    def dicreteMaximization(self, theta: tuple[float, float]) -> tuple[float, float]:
        """Maximize current likelihood.

        Args:
            theta (tuple(float, float)): Prevalence,mu

        Returns:
            tuple(float,float):Updated prevalence and mu.
        """
        P1, P0 = self.discreteExpectation(theta)
        P1Sum, P0Sum = (self.h * P1).sum(), (self.h * P0).sum()
        rho = min(P1Sum / (P1Sum + P0Sum), 1.0)
        mu = np.dot(self.h * P1, self.b) / (P1Sum + 1e-5)
        return rho, mu

    def discreteEM(self) -> tuple[float, float]:
        """Estimate theta=(prevalence,mu).

        Returns:
            tuple(float,float):Fitted prevalence and mu.
        """
        mu = 0.02 * self.l
        rho = self.h[0] / sum(self.h) * (1 + mu)
        theta = (max(min(1.0, rho), 0.1), mu)
        for _ in range(self.howmany):
            theta = self.dicreteMaximization(theta)
        return theta
