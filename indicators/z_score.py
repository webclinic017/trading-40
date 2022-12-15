import numpy as np


def zscore(x, mean, std_dev):
    """
    Args:
        x:
        mean: E[x], or a lagged value thereof.
        std_dev: sqrt(Var[x]), or a lagged value thereof.
    """
    return (x - mean) / std_dev


def zscore_signals(df_in, z_entry, z_exit):
    """
    Args:
        x:
        z_entry: when to long the portfolio, i.e. enter market.
        z_exit: when to short the portfolio, i.e. exit market.

    Returns:

    """
    df = df_in.copy()

    # Cast bools -> floats
    df["long"] = 1.0 * (df["zscore"] <= -z_entry)
    df["short"] = 1.0 * (df["zscore"] >= z_entry)
    df["exit"] = 1.0 * (np.abs(df["zscore"]) <= z_exit)

    return df
