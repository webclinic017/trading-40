import numpy as np


def sharpe_ratio_log(df, colname="total"):
    # Log returns are additive - better choice when doing *sqrt(252)
    log_returns = np.log(df[colname]/df[colname].shift())
    log_returns.fillna(0.0, inplace=True)
    log_returns.replace([np.inf, -np.inf], 0.0, inplace=True)

    sharpe = log_returns.mean()/log_returns.std()
    return sharpe
