import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR


def johansen_95(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Johansen Test for cointegration.
    Note: best used with larger sample sizes.
    Args:
        x:
        y:

    Returns:

    """
    critical_values = {
        0: {0.9: 13.4294, 0.95: 15.4943, 0.99: 19.9349},  # Critical values for 0 cointegration relationships.
        1: {0.9: 2.7055, 0.95: 3.8415, 0.99: 6.6349},     # Critical values for 1 cointegration relationship.
    }

    trace0_cv_95 = critical_values[0][0.95]
    trace1_cv_95 = critical_values[1][0.95]

    # (n, 2) <- (n,) and (n,)
    prices = np.vstack([x, y]).T

    # Vector Autoregressive Model
    var = VAR(prices)
    lags = var.select_order()
    k_ar_diff = lags.selected_orders["aic"]

    cj = coint_johansen(prices, det_order=0, k_ar_diff=k_ar_diff)

    # lr1 := Trace Statistic
    trace0, trace1 = cj.lr1

    # Did we pass the Johansen Test? - Require: (True, True).
    return (trace0 > trace0_cv_95) and (trace1 > trace1_cv_95)
