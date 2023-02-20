import pandas as pd
import yfinance as yf
from etl.load import get_data


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


def get_pairs_data_backtest(ticker0, ticker1, start_date, end_date, interval, convert_timezone, roll_fn0, roll_fn1):
    df0_raw = get_data(ticker=ticker0, start_date=start_date, end_date=end_date, interval=interval, convert_timezone=convert_timezone)
    df1_raw = get_data(ticker=ticker1, start_date=start_date, end_date=end_date, interval=interval, convert_timezone=convert_timezone)

    # Ensure timestamps align.
    lsuffix = "_0"
    rsuffix = "_1"
    data_aligned_df = df0_raw.join(df1_raw, how="left", lsuffix=lsuffix, rsuffix=rsuffix)

    # Drop rows where entries are missing for either asset.
    data_aligned_df.dropna(inplace=True)
    print(f"\tasset0 data: {df0_raw.shape}, \n\tasset1 data: {df1_raw.shape}, \n\tmerged data: {data_aligned_df.shape}")

    df0 = data_aligned_df.filter(like=lsuffix)
    df1 = data_aligned_df.filter(like=rsuffix)

    # Restore original names (for backtrader).
    df0.columns = [col.strip(lsuffix) for col in df0.columns]
    df1.columns = [col.strip(rsuffix) for col in df1.columns]

    df0["roll_date"] = 0
    df0 = df0.groupby(by=[df0.index.month, df0.index.year])
    df0 = df0.apply(roll_fn0, roll_date="roll_date")

    df1["roll_date"] = 0
    df1 = df1.groupby(by=[df1.index.month, df1.index.year])
    df1 = df1.apply(roll_fn1, roll_date="roll_date")

    return df0, df1
