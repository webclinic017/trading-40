import pandas as pd
import yfinance as yf


def get_data(ticker, start_date, end_date, interval, convert_timezone):
    # Download and cache data.
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    assert len(df) > 0
    assert df.isna().sum().sum() == 0.0

    format_map = {
        "1d": "%Y-%m-%d",
        "1h": "%Y-%m-%d %H:%M:%S",
    }

    df.index = pd.to_datetime(df.index, format=format_map[interval])

    # Convert timezones (to UTC).
    if convert_timezone:
        df.index = df.index.tz_convert(None)

    return df
