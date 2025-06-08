"""Code to fit prevalence, mu, and beta distribution parameters."""

from __future__ import annotations

import numpy as np
from scipy.special import factorial, beta, gammaln
from scipy.optimize import minimize_scalar


class EM:
    """Finds the mixture distribution that explains the statistics of distances."""

    def __init__(
        self,
        h: np.ndarray,
        howmany: int = 20,
    ) -> None:
        """Initialize class.

        Args:
            h (np.ndarray): histogram of pairwise distances.
            l (int): Number of trials for beta-binomial distribution.
            howmany (int, optional): How many iterations to run for expectmax algo. Defaults to 10.
        """
        self.h = h
        self.l = len(self.h)
        self.b = np.arange(self.l, dtype=int)
        self.howmany = howmany

    def beta_binomial_pmf(self, k: np.ndarray, n: int, alpha: float, beta_param: float) -> np.ndarray:
        """Calculate beta-binomial probability mass function.

        Args:
            k (np.ndarray): Number of successes
            n (int): Number of trials
            alpha (float): Alpha parameter of beta distribution
            beta_param (float): Beta parameter of beta distribution

        Returns:
            np.ndarray: Probability mass function values
        """
        # Using log-space computation for numerical stability
        log_comb = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        log_beta_num = gammaln(k + alpha) + gammaln(n - k + beta_param)
        log_beta_den = gammaln(alpha) + gammaln(beta_param)
        log_beta_norm = gammaln(alpha + beta_param) - gammaln(n + alpha + beta_param)

        log_pmf = log_comb + log_beta_num + log_beta_norm - log_beta_den
        return np.exp(log_pmf)

    def discrete_expectation(self, theta: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        """Calculate membership probabilities.

        Args:
            theta (tuple): (Prevalence, mu, alpha, beta)

        Returns:
            tuple: P1 and P0 membership probabilities
        """
        rho, mu, alpha, beta_param = theta

        # P1: Poisson component
        p1 = rho * mu**self.b * np.exp(-mu) / factorial(self.b)

        # P0: Beta-binomial component
        p0 = (1 - rho) * self.beta_binomial_pmf(self.b, self.l, alpha, beta_param)

        # Normalize to get membership probabilities
        total = p1 + p0 + 1e-10  # Small epsilon for numerical stability
        return p1 / total, p0 / total

    def discrete_maximization(self, theta: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Maximize current likelihood.

        Args:
            theta (tuple): (Prevalence, mu, alpha, beta)

        Returns:
            tuple: Updated (prevalence, mu, alpha, beta)
        """
        p1, p0 = self.discrete_expectation(theta)

        # Update prevalence
        p1_sum = (self.h * p1).sum()
        p0_sum = (self.h * p0).sum()
        rho = p1_sum / (p1_sum + p0_sum + 1e-10)
        rho = np.clip(rho, 0.01, 0.99)  # Keep away from boundaries

        # Update mu (Poisson parameter)
        mu = np.dot(self.h * p1, self.b) / (p1_sum + 1e-10)
        mu = max(mu, 0.01)  # Ensure positive

        # Update alpha and beta for beta-binomial
        # Using method of moments approach
        weighted_counts = self.h * p0
        total_weight = weighted_counts.sum()

        if total_weight > 1e-10:
            # Sample mean and variance
            mean_x = np.dot(weighted_counts, self.b) / total_weight
            mean_x2 = np.dot(weighted_counts, self.b**2) / total_weight
            var_x = mean_x2 - mean_x**2

            # Convert to beta-binomial parameters
            p_hat = mean_x / self.l
            p_hat = np.clip(p_hat, 0.01, 0.99)

            # Overdispersion parameter
            if var_x > mean_x * (1 - mean_x/self.l):
                phi = (var_x - self.l * p_hat * (1 - p_hat)) / (self.l**2 * p_hat * (1 - p_hat) - var_x)
                phi = max(phi, 0.01)
            else:
                phi = 0.01

            # Convert to alpha, beta
            alpha = p_hat / phi
            beta_param = (1 - p_hat) / phi

            # Ensure reasonable bounds
            alpha = max(alpha, 0.05)
            beta_param = max(beta_param, 0.05)
        else:
            # Fallback values
            alpha, beta_param = theta[2], theta[3]

        return rho, mu, alpha, beta_param

    def discrete_em(self) -> tuple[float, float, float, float]:
        """Estimate theta=(prevalence, mu, alpha, beta).

        Returns:
            tuple: Fitted (prevalence, mu, alpha, beta)
        """
        # Initialize parameters
        mu = 0.04 * self.l
        rho = 0.2  # Start with equal mixture
        alpha = 1.0  # Symmetric beta initially
        beta_param = 1.0

        theta = (rho, mu, alpha, beta_param)

        for iteration in range(self.howmany):
            old_theta = theta
            theta = self.discrete_maximization(theta)

            # Check for convergence
            if iteration > 0:
                change = sum(abs(new - old) for new, old in zip(theta, old_theta))
                if change < 1e-6:
                    break
        return rho, mu, alpha, beta_param

    def discrete_mix(self, x: np.ndarray, theta: tuple[float, float, float, float]) -> np.ndarray:
        """Evaluate mixture distribution.

        Args:
            x (np.ndarray): Values to evaluate
            theta (tuple): (prevalence, mu, alpha, beta)

        Returns:
            np.ndarray: Mixture distribution values
        """
        rho, mu, alpha, beta_param = theta

        poisson_part = rho * mu**x * np.exp(-mu) / factorial(x)
        beta_binomial_part = (1 - rho) * self.beta_binomial_pmf(x, self.l, alpha, beta_param)

        return poisson_part + beta_binomial_part

    def error(self, theta: tuple[float, float, float, float]) -> float:
        """
        Compute RMSE between observed histogram and fitted model.

        Parameters:
            theta (tuple): (Prevalence, mu, alpha, beta).

        Returns:
            float: Root mean squared error of the normalized histogram.
        """
        observed = self.h / self.h.sum()
        model = self.discrete_mix(self.b, theta)
        return np.sqrt(((observed - model) ** 2).sum())

    def log_likelihood(self, theta: tuple[float, float, float, float]) -> float:
        """
        Compute log-likelihood of the model.

        Parameters:
            theta (tuple): (Prevalence, mu, alpha, beta).

        Returns:
            float: Log-likelihood value.
        """
        model_probs = self.discrete_mix(self.b, theta)
        model_probs = np.maximum(model_probs, 1e-10)  # Avoid log(0)
        return np.sum(self.h * np.log(model_probs))
