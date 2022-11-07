import math
import numpy as np
from collections import namedtuple


# (float, float, float, tuple)
OUParams = namedtuple("ou_params", "theta mu sigma sigma_sq sums")


def calc_optimal_ou_params(x, dt):
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
    X_xy = np.sum(x[:-1] *x[1:])


    # 2. Optimal OU Parameters, given (alpha, beta). Explicit solution to MLE.

    # Long-run mean: theta
    theta = (X_y * X_xx - X_x*X_xy) / ( n*(X_xx - X_xy) - (X_x**2 - X_x*X_y))

    # Speed of mean reversion: mu
    phi = (X_xy - theta*(X_x + X_y) + n*(theta**2)) / (X_xx - 2*theta*X_x + n*(theta**2))
    mu = -np.log(phi) / dt

    assert phi < 1.0, "Plot ln(x), e.g. Wolfram."
    assert mu > 0.0, "Speed of MR must be postive."

    # Volatility parameter: sigma. Cleaner to find sigma_sq first, then take sqrt.
    a = n * (1.0 - np.exp(-2.0 * mu * dt))
    b = X_yy - 2.0*np.exp(-mu*dt)*X_xy + np.exp(-2.0*mu*dt)*X_xx - 2.0*theta*(1.0 - np.exp(-mu*dt))*(X_y - np.exp(-mu*dt)*X_x) + n*(theta**2)*( (1 - np.exp(-mu*dt))**2 )

    sigma_sq = 2*mu*b/a
    assert sigma_sq > 0.0, "Vol_sq of MR must be positive."

    sigma = np.sqrt(sigma_sq)

    return OUParams(theta, mu, sigma, sigma_sq, (X_x, X_y, X_xx, X_yy, X_xy))


def log_likelihood_ou(theta, mu, sigma_sq, x, dt):
    n = x.shape[0]

    # tau := sigma_tilde in the paper.
    tau_sq = sigma_sq*(1 - np.exp(-2.0*mu*dt))/(2.0*mu)
    tau = np.sqrt(tau_sq)

    c = 1/(2.0*n*tau_sq)

    sq_sum = np.sum([(x[i] - x[i-1]*np.exp(-mu*dt) - theta*(1 - np.exp(-mu*dt)))**2 for i in range(1,n)])
    log_likelihood = -0.5*np.log(2.0*math.pi) - np.log(tau) - c * sq_sum

    return log_likelihood



def ou_bet_size_loglikelihoods(asset1, asset2, dt, alpha, B_candidates):
    """

    Args:
        asset1:
        asset2:
        dt: time interval
        alpha: ratio of asset1.
        B_candidates: potential choices of quantity of asset2 (to later calculate ratio, beta).

    Returns:

    """
    ou_params_candidates = []
    log_likelihoods = []
    for B in B_candidates:
        beta = B / asset2.iloc[0]

        # Define:  X_t = alpha * S1_t - beta * S2_t
        x = (alpha*asset1 - beta*asset2).to_numpy()

        ou_params = calc_optimal_ou_params(x, dt)
        ll = log_likelihood_ou(theta=ou_params.theta, mu=ou_params.mu, sigma_sq=ou_params.sigma_sq, x=x, dt=dt)

        log_likelihoods.append(ll)
        ou_params_candidates.append(ou_params)

    return log_likelihoods, ou_params_candidates
