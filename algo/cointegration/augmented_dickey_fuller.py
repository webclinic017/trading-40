import numpy as np
from statsmodels.tsa.stattools import adfuller


def adf_stationarity(x: np.ndarray, trend: str, verbose: bool = False) -> bool:
    _msg = lambda l: f"REJECT the null hypothesis of a unit root in the residuals at the {l} significance level. " \
                     f"S1 and S2 are cointegrated."

    adf = adfuller(x, regression=trend, autolag="AIC")

    test_pass = False
    test_score = adf[0]
    thresholds = adf[4]
    for k, v in thresholds.items():
        if test_score < v and not test_pass:
            test_pass = True
            if verbose:
                print(_msg(f"{k}"))

    if verbose and not test_pass:
        print("Failed to reject the null hypothesis - no cointegration.")

    return test_pass
