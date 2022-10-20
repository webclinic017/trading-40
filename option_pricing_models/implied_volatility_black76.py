from trading.option_pricing_models.black76 import Black76Eur
from trading.option_pricing_models.implied_volatility_gbs import _implied_volatility_gbs


def implied_volatility_black76_european(option_type, F_t, X_t, t, r, option_price_t, precision, max_num_iterations):
    black76 = Black76Eur()

    # Map option type to a function to be optimised.
    pricing_funcs = {
        "put": black76.put,
        "call": black76.call,
    }

    b = black76.cost_of_carry

    return _implied_volatility_gbs(
        option_value_fn=pricing_funcs[option_type],
        option_type=option_type,
        F_t=F_t,
        X_t=X_t,
        t=t,
        r=r,
        b=b,
        option_price_t=option_price_t,
        precision=precision,
        max_num_iterations=max_num_iterations,
    )
