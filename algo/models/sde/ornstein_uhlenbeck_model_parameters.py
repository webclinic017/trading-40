import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelParamsOU:
    theta: float
    mu: float
    sigma_sq: float
    log_likelihood: float
    B: Optional[float] = None

    @property
    def sigma(self) -> float:
        return np.sqrt(self.sigma_sq)


@dataclass
class ModelParamsOUCandidates:
    model_params: List[ModelParamsOU]

    @property
    def max_loglikelihood(self) -> ModelParamsOU:
        """Finds the set of model parameters with the maximum log-likelihood."""
        best_index = 0
        best_params = self.model_params[best_index]

        for index, params in enumerate(self.model_params):
            if params.log_likelihood > best_params.log_likelihood:
                best_params = params
                best_index = index

        return self.model_params[best_index]

    @property
    def max_mean_reversion(self) -> ModelParamsOU:
        """Finds the set of model parameters with the fastest speed of mean reversion, mu."""
        best_index = 0
        best_params = self.model_params[best_index]

        for index, params in enumerate(self.model_params):
            if params.mu > best_params.mu:
                best_params = params
                best_index = index

        return self.model_params[best_index]

    @property
    def log_likelihoods(self) -> List[float]:
        """Convenience function to access the log-likelihoods across all candidates. Used for plotting."""
        return [p.log_likelihood for p in self.model_params]

    @property
    def B_candidates(self) -> List[float]:
        """Convenience function to access the log-likelihoods across all candidates. Used for plotting."""
        return [p.B for p in self.model_params]


@dataclass
class HedgeParamsOU:

    # OU model parameters
    ou_params: ModelParamsOU

    # Starting cash value of each asset
    A: float
    B: float

    # S1[0] and S2[0]
    series1_initial_value: float
    series2_initial_value: float

    @property
    def alpha(self) -> float:
        # Hedge Ratio for asset 1
        return self.A / self.series1_initial_value

    @property
    def beta(self) -> float:
        # Hedge Ratio for asset 2
        return self.B / self.series2_initial_value



