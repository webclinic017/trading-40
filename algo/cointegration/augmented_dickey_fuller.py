import numpy as np
from statsmodels.tsa.stattools import adfuller


def adf_stationarity(x: np.ndarray, trend: str, verbose: bool = False) -> bool:
    """
    Null Hypothesis, H0:         there is a unit root. I.e. the series is non-stationary.
    Alternative Hypothesis, H1:  there is no unit root. I.e. the series is stationary.

    If p_value > critical_value: cannot reject the null that there is a unit root.

    NOTE: ADF is robust to serial autocorrelation.
    Args:
        x:
        trend:
        verbose:

    Returns:

    """
    _msg = lambda l: f"REJECT the null hypothesis of a unit root in the residuals at the {l} significance level. " \
                     f"S1 and S2 are cointegrated."

    adf = adfuller(x, regression=trend, autolag="AIC")
    # adf = adfuller(x, regression=trend, autolag="AIC", regresults=True)

    critical_value_test_pass = False
    test_statistic = adf[0]
    p_value = adf[1]
    critical_values = adf[4]
    for k, critical_value in critical_values.items():
        if test_statistic < critical_value and not critical_value_test_pass:
            critical_value_test_pass = True
            if verbose:
                print(_msg(f"{k}"))

    p_value_test_pass = p_value < 0.05

    test_pass = critical_value_test_pass and p_value_test_pass

    if verbose and not test_pass:
        print("Failed to reject the null hypothesis - no cointegration.")

    # if test_pass:
    #     print(f"ADF p_value = {p_value}")

    if critical_value_test_pass and not p_value_test_pass:
        print("-"*80, "CV and not P")

    return test_pass
