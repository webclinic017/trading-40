import pandas as pd
import yfinance as yf


def get_pairs_data(
        ticker1: str,
        ticker2: str,
        start_date: str,
        end_date,
        interval: str,
        name_asset1: str = "S1",
        name_asset2: str = "S2",
) -> pd.DataFrame:
    """
    Use yfinance API to get historical data for 2 assets with the provided ticker symbols.
    Args:
        ticker1: ticker for asset 1.
        ticker2: ticker for asset 2.
        start_date:
        end_date:
        interval: {"1m", "1h", "1d"}.
        name_asset1: output column name of asset 1.
        name_asset2: output column name of asset 2.

    Returns:
        df: Adj Close for both assets.
    """
    df1 = yf.download(ticker1, start=start_date, end=end_date, interval=interval)
    df2 = yf.download(ticker2, start=start_date, end=end_date, interval=interval)

    # Use common index to align dates.
    df = pd.DataFrame(index=df1.index)

    colname = "Adj Close"
    df[name_asset1] = df1[colname]
    df[name_asset2] = df2[colname]

    df.index = pd.to_datetime(df.index)

    # Drop missing entries
    df.dropna(inplace=True)

    return df
