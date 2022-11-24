import abc
import math
import numpy as np
import pandas as pd
from typing import Tuple

from algo.sde.ornstein_uhlenbeck_parameters import HedgeParamsOU, ModelParamsOU, ModelParamsOUCandidates


class Optimiser(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    def optimise(self, *args, **kwargs):
        raise NotImplementedError


class OptimiserOU(Optimiser):

    def __init__(
            self,
            dt: float,
            A: float = 1.0,
    ):
        super().__init__()

        """
        Ornstein-Uhlenbeck Model Parameter Optimiser.
        Given a choice of ratio for asset 1, optimise for the ratio of asset 2.

        Args:
        """
        self.A = A
        self.dt = dt

    def optimise(
            self, asset1, asset2, optimisation_metric: str = "log_likelihood",
    ) -> Tuple[HedgeParamsOU, ModelParamsOUCandidates]:
        """
        Args:
            optimisation_metric: ["log_likelihood", "mu"]
        """
        candidates = self._create_candidates(asset1, asset2)

        optimisation = {
            "log_likelihood": candidates.max_loglikelihood,
            "mean_reversion": candidates.max_mean_reversion,
        }
        ou_model_params_optimal = optimisation[optimisation_metric]

        hedge_parameters = HedgeParamsOU(
            ou_params=ou_model_params_optimal,
            A=self.A,
            B=ou_model_params_optimal.B,
            series1_initial_value=asset1[0],
            series2_initial_value=asset2[0],
        )

        return hedge_parameters, candidates

    def model_params_ou(self, x: np.ndarray) -> ModelParamsOU:
        dt = self.dt

        n = x.shape[0]

        # 1. Define sums - all sums are from i=1:n

        # x_{i-1} --> from 0:n-1 --> sum all but last value.
        X_x = np.sum(x[:-1])

        # x_{i} --> from 1:n --> sum all but 0th value.
        X_y = np.sum(x[1:])

        # (x_{i-1})^2
        X_xx = np.sum(x[:-1]**2)

        # (x_{i})^2
        X_yy = np.sum(x[1:]**2)

        # x_{i-1} * x_{i}
        X_xy = np.sum(x[:-1] * x[1:])


        # 2. Optimal OU Parameters, given (alpha, beta). Explicit solution to MLE.

        # Long-run mean: theta
        theta = (X_y * X_xx - X_x*X_xy) / ( n*(X_xx - X_xy) - (X_x**2 - X_x*X_y) )

        # Speed of mean reversion: mu
        phi = (X_xy - theta*(X_x + X_y) + n*(theta**2)) / (X_xx - 2*theta*X_x + n*(theta**2))
        mu = -np.log(phi) / dt

        assert phi < 1.0, "Plot ln(x), e.g. Wolfram."
        assert mu > 0.0, "Speed of MR must be postive."

        # Volatility parameter: sigma. Cleaner to find sigma_sq first, then take sqrt.
        a = n * (1.0 - np.exp(-2.0 * mu * dt))
        b = X_yy - 2.0*np.exp(-mu*dt)*X_xy + np.exp(-2.0*mu*dt)*X_xx - 2.0*theta*(1.0 - np.exp(-mu*dt))*(X_y - np.exp(-mu*dt)*X_x) + n*(theta**2)*( (1 - np.exp(-mu*dt))**2 )

        sigma_sq = 2 * mu * b / a
        assert sigma_sq > 0.0, "Vol_sq of MR must be positive."

        log_likelihood = self.log_likelihood_ou(x=x, theta=theta, mu=mu, sigma_sq=sigma_sq)

        return ModelParamsOU(theta=theta, mu=mu, sigma_sq=sigma_sq, log_likelihood=log_likelihood)

    def log_likelihood_ou(self, theta: float, mu: float, sigma_sq: float, x: np.ndarray) -> float:
        dt = self.dt
        n = x.shape[0]

        # tau := sigma_tilde in the paper.
        tau_sq = sigma_sq*(1 - np.exp(-2.0*mu*dt))/(2.0*mu)
        tau = np.sqrt(tau_sq)

        c = 1/(2.0*n*tau_sq)

        sq_sum = np.sum([(x[i] - x[i-1]*np.exp(-mu*dt) - theta*(1 - np.exp(-mu*dt)))**2 for i in range(1, n)])
        log_likelihood = -0.5*np.log(2.0*math.pi) - np.log(tau) - c * sq_sum

        return log_likelihood

    def _create_candidates(self, asset1, asset2):
        start = 0.001
        end = 1.0
        num = int(1 / start)
        B_candidates = np.linspace(start, end, num)

        alpha = self.A / asset1[0]

        # Create set of candidates
        model_params_candidates = []
        for B in B_candidates:
            beta = B / asset2[0]

            # Define the spread: X_t = alpha * S1_t - beta * S2_t
            x = alpha * asset1 - beta * asset2

            # Ornstein-Uhlenbeck model parameters
            model_params = self.model_params_ou(x)
            model_params.B = B
            model_params_candidates.append(model_params)

        return ModelParamsOUCandidates(model_params=model_params_candidates)


def estimate_halflife_ou(spread: pd.Series) -> float:
    # Shape: (m, 2), where m = n-1; n := len(spread)
    X = spread.shift().iloc[1:].to_frame().assign(const=1)

    # Shape: (m, )
    y = spread.diff().iloc[1:]

    # Shape: ((2, m) * (m, 2)) * ((2, m) * (m,)) = (2 * 2) * (2,) = (2,)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    # Consider: int(round(halflife, 0))
    halflife = -np.log(2) / beta[0]

    return halflife
