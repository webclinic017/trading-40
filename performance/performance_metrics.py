import math
import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame, price_colname: str, output_colname: str = "returns") -> pd.DataFrame:
    """
    Computes simple percentage returns between adjacent timesteps.
    Args:
        df: pandas dataframe to add the computed returns to
        price_colname: the name of the column to use for the price in the returns calculation.
        returns_colname: the name of the column to be added or modified with the output of the returns calculation.
    """
    assert price_colname in df.columns
    df[output_colname] = df[price_colname].pct_change(1)
    return df


def compute_log_returns(df: pd.DataFrame, price_colname: str, output_colname: str = "log_returns") -> pd.DataFrame:
    """
    Computes the natural log (ln) of returns between adjacent timesteps.
    Args:
        df: pandas dataframe to add the computed returns to
        price_colname: the name of the column to use for the price in the log_returns calculation.
        output_colname: the name of the column to be added or modified with the output of the log_returns calculation.
    """
    assert price_colname in df.columns
    df[output_colname] = np.log(df[price_colname]/df[price_colname].shift(1))
    return df


def sharpe_ratio(
        df: pd.DataFrame,
        returns_colname: str,
        risk_free_rate_colname: str,
        output_colname: str = "sharpe_ratio",
) -> pd.DataFrame:
    """
    Computes Sharpe Ratio expanding over time dimension.
    E.g. for daily data, the calculation uses summary statistics up to day i, where i \in {0,t}.

    Leverages the Martingale property: E[X_{n+1} | X_1, ... ,X_n] = X_n to use returns at time t
    in place of expected returns from time t+1 onwards.

    df: dataframe
    output_colname:         name of the column containing the portfolio returns.
    risk_free_rate_colname: name of the column containing the (log) returns of a risk-free baseline asset.
    output_colname:         name of the new column which will contain the computed Sharpe Ratio.
    """
    assert returns_colname in df.columns, "Dataframe must contain computed returns before computing Sharpe Ratio"
    assert risk_free_rate_colname in df.columns, "Dataframe must contain risk-free returns before computing Sharpe Ratio"

    returns = df[returns_colname].expanding().mean()
    risk_free_rate = df[risk_free_rate_colname].expanding().mean()
    std = df[returns_colname].expanding().std()

    df[output_colname] = (returns - risk_free_rate) / std
    return df


def sortino_ratio(
        df: pd.DataFrame,
        returns_colname: str,
        risk_free_rate_colname: str,
        output_colname: str = "sortino_ratio",
) -> pd.DataFrame:
    """
    Computes Sortino Ratio expanding over time dimension.
    E.g. for daily data, the calculation uses summary statistics up to day i, where i \in {0,t}.

    Leverages the Martingale property: E[X_{n+1} | X_1, ... ,X_n] = X_n to use returns at time t
    in place of expected returns from time t+1 onwards.

    Args:
        df: dataframe
        returns_colname:        name of the column containing the portfolio returns.
        risk_free_rate_colname: name of the column containing the returns of a risk-free baseline asset.
        output_colname:         name of the new column which will contain the computed Sortino Ratio.
    """
    assert returns_colname in df.columns, "Dataframe must contain computed returns before computing Sortino Ratio"
    assert risk_free_rate_colname in df.columns, "Dataframe must contain risk free rate of returns before computing Sortino Ratio"

    returns = df[returns_colname].expanding().mean()
    risk_free_rate = df[risk_free_rate_colname].expanding().mean()

    downside_dev_df = df[df[returns_colname] < 0]
    downside_dev = downside_dev_df[returns_colname].std()

    # Scaling can result in no negative values.
    assert not math.isnan(downside_dev)

    df[output_colname] = (returns - risk_free_rate) / downside_dev
    return df
