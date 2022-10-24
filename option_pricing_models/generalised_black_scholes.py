import abc
import numpy as np
from collections import namedtuple
from scipy.stats import norm

Greeks = namedtuple("greeks", "option_value delta gamma theta vega rho")


class GeneralisedEuropeanBlackScholes(abc.ABC):
    """
    Args:
        F_t: price of the underlying
        X_t: underlying strike price
        t:   time to expiration
        vol: implied volatility
        r:   risk-free rate
        q:   (potential) dividend payment
        b:   (potential) cost of carry

    Returns:
        option_value: calculated price of the option
        delta:        1st derivative of value wrt. the price of the underlying (F_t)
        gamma:        2nd derivative of value wrt. the price of the underlying (F_t)
        theta:        1st derivative of value wrt. time to expiration (t)
        vega:         1st derivative of value wrt. implied volatility (vol)
        rho:          1st derivative of value wrt. risk-free rate (r)
    """

    def __init__(self, name="GBS"):
        self.name = name

    def call(self, F_t, X_t, t, r, b, vol):
        # Pre-calculations
        sqrt_t = np.sqrt(t)
        d1, d2 = self._distribution_parameters(F_t, X_t, t, r, b, vol, sqrt_t)

        # Fair value price
        option_value = F_t * np.exp((b - r) * t) * norm.cdf(d1) - X_t * np.exp(-r * t) * norm.cdf(d2)

        # Greeks
        delta = np.exp((b - r) * t) * norm.cdf(d1)
        gamma = np.exp((b - r) * t) * norm.pdf(d1) / (F_t * vol * sqrt_t)
        theta = -(F_t * vol * np.exp((b - r) * t) * norm.pdf(d1)) / (2 * sqrt_t) - (b - r) * F_t * np.exp(
            (b - r) * t) * norm.cdf(d1) - r * X_t * np.exp(-r * t) * norm.cdf(d2)
        vega = np.exp((b - r) * t) * F_t * sqrt_t * norm.pdf(d1)
        rho = X_t * t * np.exp(-r * t) * norm.cdf(d2)

        return Greeks(option_value, delta, gamma, theta, vega, rho)

    def put(self, F_t, X_t, t, r, b, vol):
        # Pre-calculations
        sqrt_t = np.sqrt(t)
        d1, d2 = self._distribution_parameters(F_t, X_t, t, r, b, vol, sqrt_t)

        # Fair value price
        option_value = X_t * np.exp(-r * t) * norm.cdf(-d2) - (F_t * np.exp((b - r) * t) * norm.cdf(-d1))

        # Greeks
        delta = -np.exp((b - r) * t) * norm.cdf(-d1)
        gamma = np.exp((b - r) * t) * norm.pdf(d1) / (F_t * vol * sqrt_t)
        theta = -(F_t * vol * np.exp((b - r) * t) * norm.pdf(d1)) / (2 * sqrt_t) + (b - r) * F_t * np.exp(
            (b - r) * t) * norm.cdf(-d1) + r * X_t * np.exp(-r * t) * norm.cdf(-d2)
        vega = np.exp((b - r) * t) * F_t * sqrt_t * norm.pdf(d1)
        rho = -X_t * t * np.exp(-r * t) * norm.cdf(-d2)

        return Greeks(option_value, delta, gamma, theta, vega, rho)

    @staticmethod
    def _distribution_parameters(F_t, X_t, t, r, b, vol, sqrt_t):
        # Static method allows for cleaner use of scipy.optimize.
        d1 = (np.log(F_t / X_t) + (b + np.square(vol) / 2) * t) / (vol * sqrt_t)
        d2 = d1 - vol * sqrt_t

        return d1, d2

    @property
    def cost_of_carry(self, *args, **kwargs):
        raise NotImplementedError
