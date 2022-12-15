import pandas as pd


def simple_moving_average(df: pd.DataFrame, colname: str, span: int) -> pd.DataFrame:
    """
    Args:
        df:
        colname: data column to apply MA over.
        span: duration of time to roll the MA over.

    Returns:
    """
    assert colname in df.columns

    df[f"sma_{span}_{colname}"] = df[colname].rolling(span).mean()
    df.dropna(inplace=True)
    return df


def exponential_moving_average(df: pd.DataFrame, colname: str, span: int) -> pd.DataFrame:
    """
    Args:
        df:
        colname: data column to apply MA over.
        span: duration of time to roll the MA over.

    Returns:
    """
    assert colname in df.columns

    df[f"ema_{span}_{colname}"] = df[colname].ewm(span=span).mean()
    df.dropna(inplace=True)
    return df
