"""Code to fit prevalence and mu."""

from __future__ import annotations

import numpy as np
from scipy.special import factorial


class EM:
    """Finds the mixture distribution that explains the statistics of distances."""

    def __init__(
        self,
        cdf: np.array,
        h: np.ndarray,
        howmany: int = 10,
    ) -> None:
        """Initialize class.

        Args:
            cdf (np.array): P0 distribution.
            h (np.ndarray): histogram of pairwise distances.
            howmany (int, optional): How many iterations to run for expectmax algo. Defaults to 10.
        """
        self.cdf = cdf[:-1]
        self.l = len(self.cdf)
        self.h = h
        self.b = np.arange(self.l + 1, dtype=int)
        self.const_p0 = self.read_null()
        self.howmany = howmany

    def read_null(self) -> np.ndarray:
        """Read estimated null distribution.

        Returns
        -------
        np.ndarray
            Histogram of null distributions (Ppost).
        """
        return np.diff(self.cdf, prepend=[0], append=[1])

    def discrete_expectation(self, theta: tuple[float, float]) -> tuple[float, float]:
        """Calculate membership probabilities.

        Args:
            theta (tuple(float, float)): (Prevalence, mu)

        Returns
        -------
        tuple(float, float)
            P1 and P0 computed with updated prevalence and mu.
        """
        rho, mu = theta
        p1 = rho * mu**self.b * np.exp(-mu) / factorial(self.b)
        p0 = (1 - rho) * self.const_p0[self.b]
        return np.array([p1, p0]) / (p1 + p0 + 1e-5)

    def discrete_maximization(self, theta: tuple[float, float]) -> tuple[float, float]:
        """Maximize current likelihood.

        Args:
            theta (tuple(float, float)): Prevalence, mu

        Returns
        -------
        tuple(float, float)
            Updated prevalence and mu.
        """
        p1, p0 = self.discrete_expectation(theta)
        p1_sum, p0_sum = (self.h * p1).sum(), (self.h * p0).sum()
        rho = min(p1_sum / (p1_sum + p0_sum + 1e-5), 1.0)
        mu = np.dot(self.h * p1, self.b) / (p1_sum + 1e-5)
        return rho, mu

    def discrete_em(self) -> tuple[float, float]:
        """Estimate theta=(prevalence, mu).

        Returns
        -------
        tuple(float, float)
            Fitted prevalence and mu.
        """
        mu = 0.02 * self.l
        rho = self.h[0] / (sum(self.h)+1e-6) * (1 + mu)
        theta = (max(min(1.0, rho), 0.1), mu)
        for _ in range(self.howmany):
            theta = self.discrete_maximization(theta)
        return theta

    def discrete_mix(self, x, theta):
        """Evaluate mixture distribution."""
        rho, mu = theta
        return rho * mu**x * np.exp(-mu) / factorial(x) + (1 - rho) * self.const_p0[x]

    def error(self, theta: list[float]) -> float:
        """
        Compute RMSE between observed histogram and fitted model.

        Parameters
        ----------
        theta : list[float]
            (Prevalence, mu).

        Returns
        -------
        float
            Root mean squared error of the normalized histogram.
        """
        observed = self.h / (self.h.sum()+1e-6)
        model = self.discrete_mix(self.b, theta)
        return np.sqrt(((observed - model) ** 2).sum())
