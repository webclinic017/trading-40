import functools
import numpy as np
from scipy import stats


"""
Manual implementation of the Black76 Model for call and put options where the underlying is a futures contract,
e.g. on a commodity.

The main implementation of Black76 is a child class of Generalised Black-Scholes, which sit in
`black76.py` and `generalised_black_scholes.py` respectively. 

This file is for testing.
"""


def _params_black76(F_t, X_t, t, sigma):
    """
    Args:
        F_t: futures price
        X_t: options contract strike price
        t: time to t
        r_free: risk free rate
        sigma: sigma
    """
    d1 = (np.log(F_t / X_t) + 0.5 * np.square(sigma) * t) / (sigma * np.sqrt(t))
    d2 = (np.log(F_t / X_t) - 0.5 * np.square(sigma) * t) / (sigma * np.sqrt(t))

    return d1, d2


def call_black76(F_t, X_t, t, r_free, sigma, normal):
    """
    Args:
        F_t: futures price
        X_t: options contract strike price
        t: time to t
        r_free: risk free rate
        sigma: sigma
        normal: CDF of normal distribution
    """
    d1, d2 = _params_black76(F_t, X_t, t, sigma)
    call_value = np.exp(-r_free * t) * (F_t * normal(d1) - X_t * normal(d2))
    return call_value


def put_black76(F_t, X_t, t, r_free, sigma, normal):
    """
    Args:
        F_t: futures price
        X_t: options contract strike price
        t: time to t
        r_free: risk free rate
        sigma: sigma
        normal: CDF of normal distribution
    """
    d1, d2 = _params_black76(F_t, X_t, t, sigma)
    put_value = np.exp(-r_free * t) * (X_t * normal(-d2) - F_t * normal(-d1))
    return put_value


# TODO(JP): convert to pytest
if __name__ == "__main__":
    F_t = 50
    X_t = 55
    t = 1/12
    r_free = 0.04
    sigma = 0.20

    normal = functools.partial(stats.norm.cdf, loc=0.0, scale=1.0)
    print(call_black76(F_t, X_t, t, r_free, sigma, normal))
    print(put_black76(F_t, X_t, t, r_free, sigma, normal))
