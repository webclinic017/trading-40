import numpy as np
from typing import Optional

from trading.algo.mean_reversion.brownian_motion import BrownianMotion


class OrnsteinUhlenbeck(BrownianMotion):

    def __init__(
            self,
            k: float,
            theta: float,
            sigma: float,
            X_0: Optional[float] = None,
            mean: float = 0.0,
            std_dev: float = 1.0,
    ):
        """
        Args: variables are given symbolic names to mirror mathematical literature.
            k:      mean reversion parameter
            theta:  asymptotic mean
            sigma:  volatility of the process, i.e. scale of Brownian motion (std dev).
            X_0:    initial value. Free to choose any.

        Examples of X_0 choices:
            - Long term mean (theta).
            - Most recent data point.

        In general:         X_t = X_0*exp(-k * t) + theta*(1 - exp(-k * t)) + sigma * exp(-k * t)*W
        If X_0 = theta:     X_t = theta + sigma * exp()-k * t * W
        """
        super().__init__(mean, std_dev)
        self.k = k
        self.sigma = sigma
        self.theta = theta

        # Default: set initial value to long-run mean.
        self.X_0 = X_0 or theta

    def __call__(self, num_samples: int) -> np.ndarray:
        # Construct vector of time steps for each sample data point.
        t = np.arange(num_samples)
        dW = self.dW(num_samples)

        # Integrate wrt. Brownian Motion (W), ∫...dW.
        exp_k_s = np.exp(self.k * t)
        W = np.cumsum(exp_k_s * dW)
        W = np.insert(W, 0, 0)[:-1]  # Enforce W(0)=0.

        exp_k_t = np.exp(-self.k * t)

        return self.X_0 * exp_k_t + self.theta * (1 - exp_k_t) + self.sigma * exp_k_t * W
