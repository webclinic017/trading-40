import backtrader as bt
import backtrader.analyzers as btanalyzers
import backtrader.feeds as btfeeds
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta

from algo.strategies.momentum.long_short_uptrend_strategy import LongShortUptrendStrategy


def run(data_path, start_date, end_date):
    cb = bt.Cerebro()

    data = btfeeds.YahooFinanceCSVData(
        dataname=data_path,
        fromdate=datetime.strptime(start_date, "%Y-%m-%d"),
        todate=datetime.strptime(end_date, "%Y-%m-%d"),
    )
    cb.adddata(data)

    cb.addstrategy(LongShortUptrendStrategy)

    cb.addanalyzer(btanalyzers.Returns, _name="returns")
    cb.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe_ratio", timeframe=bt.TimeFrame.Years)

    # Set starting cash balance
    cb.broker.setcash(1000000.0)

    # Percentage commission - broker fees: 0.005 is 0.5%.
    cb.broker.setcommission(0.01)

    # Initial conditions
    print(f"Starting Portfolio Value: {cb.broker.getvalue()}")

    strategies = cb.run()
    strategy = strategies[0]

    print(f"Returns: {strategy.analyzers.returns.get_analysis()}")
    print(f"Sharpe Ratio: {strategy.analyzers.sharpe_ratio.get_analysis()}")

    # Result
    print(f"Final Portfolio Value: {cb.broker.getvalue()}")

    cb.plot()


if __name__ == "__main__":
    # TODO:
    #  [] get_data function.
    #  [] config -> whether to re-run get_data

    # S&P 500
    # ticker = "SPY"

    # Chevron
    ticker = "CVX"

    data_path = f"data/{ticker}.csv"
    price = "price"

    download_data = True

    if download_data:
        interval = "1d"
        end_date = date.today()
        start_date = end_date - timedelta(days=4*252)
        df = yf.download(
            ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval=interval
        )

        # TODO: backtrader seems to only allow dfs with OHLCV.
        # price_col_raw = "Adj Close"
        # df.rename(columns={price_col_raw: price}, inplace=True)

        # Double [[]] keeps `df` as pd.DataFrame, not pd.Series.
        # df = df[[price]].dropna()

        # df = get_data()
        df.to_csv(data_path)

    df = pd.read_csv(data_path)
    start_date = df.loc[0, "Date"]
    end_date = df.loc[len(df) - 1, "Date"]

    # TODO: add_indicators function, e.g. to add SMAs already!
    # Indicates if asset is trending upwards in the long term.
    # span_trend = 200
    #
    # # Indicates momentum in the short term.
    # span_entry = 10
    #
    # df = df \
    #     .pipe(simple_moving_average, colname=price, span=span_trend) \
    #     .pipe(simple_moving_average, colname=price, span=span_entry)

    run(data_path, start_date, end_date)
