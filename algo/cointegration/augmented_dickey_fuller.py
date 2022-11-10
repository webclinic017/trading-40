from typing import Tuple


def print_adf_results(adf: Tuple):
    """
    Args:
        adf: statsmodels.tsa.stattools.adfuller result object.
    """
    _msg = lambda l: f"REJECT the hypothesis that S1 and S2 are NOT cointegrated at the {l} significance level"

    test_pass = False
    test_score = adf[0]
    thresholds = adf[4]
    for k, v in thresholds.items():
        if test_score < v and not test_pass:
            test_pass = True
            print(_msg(f"{k}"))

    if not test_pass:
        print("Failed to reject the null hypothesis - no cointegration.")
