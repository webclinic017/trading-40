import pandas as pd
import yfinance as yf
from datetime import datetime


def get_data_with_vix(ticker: str, vix: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
    """
    Get data for `ticker` and its corresponding volatility index, often called `VIX`.
    Args:
        ticker:
        num_data:
        interval:

    Returns:

    """
    ticker_df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval
    )
    ticker_df.rename(columns={"Adj Close": "price", "Volume": "volume"}, inplace=True)
    ticker_df = ticker_df[["price", "volume"]]

    # TODO: properly. Temporary hack, but better to align both frames to a common timezone, especially if hourly.
    ticker_df = ticker_df.tz_localize(None)

    vix_df = yf.download(
        vix,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval
    )
    vix_df.rename(columns={"Adj Close": "vix"}, inplace=True)
    vix_df = vix_df[["vix"]]

    df = ticker_df.join(vix_df)
    df.dropna(inplace=True)

    return df
