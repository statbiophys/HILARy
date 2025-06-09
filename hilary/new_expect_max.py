import numpy as np
from scipy.stats import poisson
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize_scalar
import warnings

class EM:
    def __init__(self, h):
        self.h = np.asarray(h, dtype=np.float64)
        self.n = len(h) - 1
        self.k = np.arange(len(h))
        self.N = np.sum(h)

        # Precompute constants for efficiency
        self.log_binom_coeff = (
            gammaln(self.n + 1) - gammaln(self.k + 1) - gammaln(self.n - self.k + 1)
        )

    def beta_binomial_log_pmf(self, alpha, beta):
        """Compute log PMF of beta-binomial distribution for numerical stability"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_pmf = (
                self.log_binom_coeff +
                gammaln(self.k + alpha) +
                gammaln(self.n - self.k + beta) -
                gammaln(self.n + alpha + beta) +
                gammaln(alpha + beta) -
                gammaln(alpha) -
                gammaln(beta)
            )
        return log_pmf

    def beta_binomial_pmf(self, alpha, beta):
        """Compute PMF of beta-binomial distribution"""
        log_pmf = self.beta_binomial_log_pmf(alpha, beta)
        # Clip to prevent overflow/underflow
        log_pmf = np.clip(log_pmf, -500, 500)
        return np.exp(log_pmf)

    def method_of_moments_init(self):
        """Initialize parameters using method of moments"""
        if self.N == 0:
            return 1.0, 1.0

        mean = np.sum(self.k * self.h) / self.N
        second_moment = np.sum(self.k**2 * self.h) / self.N
        var = second_moment - mean**2

        if var <= 0 or mean <= 0 or mean >= self.n:
            return 1.0, 1.0

        # Method of moments estimators
        p_hat = mean / self.n
        if p_hat <= 0 or p_hat >= 1:
            return 1.0, 1.0

        # Overdispersion parameter
        rho = max(0, (var - self.n * p_hat * (1 - p_hat)) / (self.n * p_hat * (1 - p_hat) * (self.n - 1)))

        if rho >= 1:
            return 1.0, 1.0

        # Convert to alpha, beta
        if rho > 0:
            alpha = p_hat * (1 - rho) / rho
            beta = (1 - p_hat) * (1 - rho) / rho
        else:
            alpha = beta = 1.0

        return max(0.1, alpha), max(0.1, beta)

    def compute_log_likelihood(self, rho, mu, alpha, beta):
        """Compute log-likelihood of the mixture model"""
        poisson_probs = poisson.pmf(self.k, mu)
        beta_binom_probs = self.beta_binomial_pmf(alpha, beta)

        # Mixture probabilities
        mix_probs = rho * poisson_probs + (1 - rho) * beta_binom_probs
        mix_probs = np.clip(mix_probs, 1e-15, None)

        return np.sum(self.h * np.log(mix_probs))

    def optimize_beta_binomial_params(self, weights):
        """Optimize alpha and beta parameters for beta-binomial component"""
        if np.sum(weights) < 1e-10:
            return 1.0, 1.0

        # Weighted moments
        total_weight = np.sum(weights)
        weighted_mean = np.sum(weights * self.k) / total_weight
        weighted_var = np.sum(weights * (self.k - weighted_mean)**2) / total_weight

        if weighted_var <= 0:
            return 1.0, 1.0

        # Method of moments initialization
        p_est = weighted_mean / self.n
        if p_est <= 0 or p_est >= 1:
            return 1.0, 1.0

        # Estimate overdispersion
        theoretical_var = self.n * p_est * (1 - p_est)
        if theoretical_var <= 0:
            return 1.0, 1.0

        overdispersion = max(0, (weighted_var - theoretical_var) / (theoretical_var * (self.n - 1)))

        if overdispersion >= 1:
            alpha_init = beta_init = 1.0
        else:
            alpha_init = max(0.1, p_est * (1 - overdispersion) / max(overdispersion, 1e-6))
            beta_init = max(0.1, (1 - p_est) * (1 - overdispersion) / max(overdispersion, 1e-6))

        # Optimize using maximum likelihood
        def neg_log_likelihood(log_params):
            if len(log_params) == 2:
                log_alpha, log_beta = log_params
            else:
                log_alpha = log_beta = log_params[0]

            alpha_val = np.exp(np.clip(log_alpha, -10, 10))
            beta_val = np.exp(np.clip(log_beta, -10, 10))

            try:
                log_pmf = self.beta_binomial_log_pmf(alpha_val, beta_val)
                log_pmf = np.clip(log_pmf, -500, 500)
                return -np.sum(weights * log_pmf)
            except:
                return np.inf

        # Try optimization
        try:
            from scipy.optimize import minimize
            bounds = [(-5, 5), (-5, 5)]
            result = minimize(
                neg_log_likelihood,
                x0=[np.log(alpha_init), np.log(beta_init)],
                method='L-BFGS-B',
                bounds=bounds
            )

            if result.success:
                alpha_opt = np.exp(np.clip(result.x[0], -10, 10))
                beta_opt = np.exp(np.clip(result.x[1], -10, 10))
                return max(0.01, alpha_opt), max(0.01, beta_opt)
        except:
            pass

        return max(0.01, alpha_init), max(0.01, beta_init)

    def discrete_em(self, max_iter=1000, tol=1e-6, verbose=False):
        """Run EM algorithm to fit mixture model"""
        if self.N == 0:
            return 0.5, 1.0, 1.0, 1.0

        # Initialize parameters
        rho = 0.5
        mu = max(0.1, np.sum(self.k * self.h) / self.N)
        alpha, beta = self.method_of_moments_init()

        prev_log_likelihood = -np.inf

        for iteration in range(max_iter):
            # E-step: compute responsibilities
            try:
                poisson_probs = poisson.pmf(self.k, mu)
                beta_binom_probs = self.beta_binomial_pmf(alpha, beta)

                # Mixture probabilities
                mix_probs = rho * poisson_probs + (1 - rho) * beta_binom_probs
                mix_probs = np.clip(mix_probs, 1e-15, None)

                # Responsibilities (posterior probabilities)
                w_poisson = (rho * poisson_probs) / mix_probs
                w_beta_binom = (1 - rho) * beta_binom_probs / mix_probs

                # Weighted counts
                z_poisson = self.h * w_poisson
                z_beta_binom = self.h * w_beta_binom

            except:
                if verbose:
                    print(f"Numerical error in E-step at iteration {iteration}")
                break

            # M-step: update parameters
            total_poisson = np.sum(z_poisson)
            total_beta_binom = np.sum(z_beta_binom)

            # Update mixing proportion
            rho_new = np.clip(total_poisson / self.N, 0.01, 0.99)

            # Update Poisson parameter
            if total_poisson > 1e-10:
                mu_new = max(0.01, np.sum(z_poisson * self.k) / total_poisson)
            else:
                mu_new = mu

            # Update beta-binomial parameters
            if total_beta_binom > 1e-10:
                alpha_new, beta_new = self.optimize_beta_binomial_params(z_beta_binom)
            else:
                alpha_new, beta_new = alpha, beta

            # Check convergence
            current_log_likelihood = self.compute_log_likelihood(rho_new, mu_new, alpha_new, beta_new)

            param_change = (
                abs(rho_new - rho) + abs(mu_new - mu) +
                abs(alpha_new - alpha) + abs(beta_new - beta)
            )

            likelihood_change = abs(current_log_likelihood - prev_log_likelihood)

            if verbose:
                print(f"[{iteration}] rho={rho_new:.4f}, mu={mu_new:.3f}, "
                      f"alpha={alpha_new:.3f}, beta={beta_new:.3f}, "
                      f"loglik={current_log_likelihood:.2f}, "
                      f"param_change={param_change:.6f}")

            # Convergence criteria
            if param_change < tol and likelihood_change < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break

            # Check for likelihood decrease (shouldn't happen in EM)
            if current_log_likelihood < prev_log_likelihood - 1e-6:
                if verbose:
                    print(f"Warning: likelihood decreased at iteration {iteration}")

            # Update parameters
            rho, mu, alpha, beta = rho_new, mu_new, alpha_new, beta_new
            prev_log_likelihood = current_log_likelihood

        return rho, mu, alpha, beta

    def predict_proba(self, rho, mu, alpha, beta):
        """Predict mixture probabilities for each count"""
        poisson_probs = poisson.pmf(self.k, mu)
        beta_binom_probs = self.beta_binomial_pmf(alpha, beta)
        mix_probs = rho * poisson_probs + (1 - rho) * beta_binom_probs
        return mix_probs

    def component_responsibilities(self, rho, mu, alpha, beta):
        """Get posterior probabilities for each component"""
        poisson_probs = poisson.pmf(self.k, mu)
        beta_binom_probs = self.beta_binomial_pmf(alpha, beta)
        mix_probs = rho * poisson_probs + (1 - rho) * beta_binom_probs
        mix_probs = np.clip(mix_probs, 1e-15, None)

        w_poisson = (rho * poisson_probs) / mix_probs
        w_beta_binom = (1 - rho) * beta_binom_probs / mix_probs

        return w_poisson, w_beta_binom
