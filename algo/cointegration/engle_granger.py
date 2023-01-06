from statsmodels.tsa.stattools import coint


def engle_granger_bidirectional(x, y, trend: str, p_value: float = 0.05) -> bool:
    # Augmented Engle-Granger two-step cointegration test. Test in both directions.
    coint_t_stat_1, p_value_1 = coint(x, y, trend=trend, autolag="AIC")[:2]
    coint_t_stat_2, p_value_2 = coint(y, x, trend=trend, autolag="AIC")[:2]

    # Did we pass the Engle-Granger Test? - Require: (True, True).
    return (p_value_1 < p_value) and (p_value_2 < p_value)
