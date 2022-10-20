import numpy as np

# Reference: https://github.com/dedwards25/Python_Option_Pricing/blob/master/GBS.ipynb


def _approx_implied_volatility(option_type, F_t, X_t, t, r, b, option_price_t):
    """
    Choose an initial value from which to start a search function, e.g. Newton Raphson.
    From: Brenner & Subrahmanyam (1988), Feinstein (1988).
    """
    a = np.sqrt(2.0*np.pi) / (F_t*np.exp((b - r)*t) + X_t*np.exp(-r*t))

    # Calls
    payoff = F_t*np.exp((b-r)*t) - X_t*np.exp(- r *t)

    # Puts
    if option_type == "put":
        payoff *= -1

    b = option_price_t - payoff / 2.0
    c = (payoff ** 2) / np.pi
    v = (a * (b + np.sqrt(b ** 2 + c))) / np.sqrt(t)

    return v


def _implied_volatility_gbs(
        option_value_fn,
        option_type,
        F_t,
        X_t,
        t,
        r,
        b,
        option_price_t,
        precision=0.0001,
        max_num_iterations=100,
        vol_min=0.0,
        vol_max=10.0,  # `10` represents 1000%, since `1` represents 100%.
):
    """
    Calculate Implied Volatility with a Newton Raphson search

    Args:
        option_value_fn: a function capable of pricing call or put options
        F_t: price of the underlying
        X_t: underlying strike price
        t:   time to expiration
        b:   (potential) cost of carry
        option_price_t: known market price of the option (put or call) at time t
        precision: tolerance
        max_num_iterations: computation effort
        vol_min: volatility must be positive, some formulae may need a small positive minimum, e.g. 0.5%, i.e. 0.005.
        vol_max: not striclty necessary. Prevents erroneous choices, e.g. 20% should be 0.2, not `20`.
    """

    # Estimate starting volatility, making sure it is allowable range
    implied_vol = _approx_implied_volatility(
        option_type=option_type, F_t=F_t, X_t=X_t, t=t, r=r, b=b, option_price_t=option_price_t
    )
    implied_vol = max(vol_min, min(vol_max, implied_vol))

    # Back-calculate the value of the option at the current estimate of volatility.
    greeks = option_value_fn(F_t=F_t, X_t=X_t, t=t, r=r, vol=implied_vol)
    option_price_estimate = greeks.option_value
    vega = greeks.vega

    # Iterate until the value, given estimated implied vol, is near to the known option price.
    min_valuation_error = abs(option_price_t - option_price_estimate)

    # Newton-Raphson Search
    num_iterations = 0
    while precision <= abs(
            option_price_t - option_price_estimate) <= min_valuation_error and num_iterations < max_num_iterations:

        implied_vol = implied_vol - (option_price_estimate - option_price_t) / vega
        if (implied_vol > vol_max) or (implied_vol < vol_min):
            print("Volatility out of bounds")
            break

        greeks = option_value_fn(F_t=F_t, X_t=X_t, t=t, r=r, vol=implied_vol)
        option_price_estimate = greeks.option_value
        vega = greeks.vega

        min_valuation_error = min(abs(option_price_t - option_price_estimate), min_valuation_error)
        num_iterations += 1

    # Check for convergence
    if abs(option_price_t - option_price_estimate) < precision:
        return implied_vol
    else:
        raise ValueError("Failed to converge with sufficient precision")
