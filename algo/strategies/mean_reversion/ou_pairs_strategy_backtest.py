import backtrader as bt
import backtrader.analyzers as btanalyzers
import backtrader.feeds as btfeeds
import yfinance as yf
from datetime import date, timedelta

from algo.strategies.mean_reversion.ou_pairs_strategy import OUPairsTradingStrategy


def run():
    # num_data_full = 24 * 729  # 730 days is the max, includes today.
    # num_data_full = 365
    num_data_full = 24 * 365
    interval = "1h"
    # interval = "1d"

    ticker0 = "BZ=F"
    ticker1 = "CL=F"

    today = date.today()

    # Backtrader needs datetime objects
    end_date = today
    start_date = today - timedelta(hours=num_data_full)
    # start_date = today - timedelta(days=num_data_full)

    # Yahoo needs strings
    end_date_str = end_date.strftime("%Y-%m-%d")
    start_date_str = start_date.strftime("%Y-%m-%d")

    df0 = yf.download(ticker0, start=start_date_str, end=end_date_str, interval=interval)
    df1 = yf.download(ticker1, start=start_date_str, end=end_date_str, interval=interval)

    data_path0 = "data/df1.csv"
    data_path1 = "data/df2.csv"
    df0.to_csv(data_path0)
    df1.to_csv(data_path1)

    cb = bt.Cerebro()

    # Add only the TEST data streams to the engine
    data0 = btfeeds.YahooFinanceCSVData(
        dataname=data_path0,
        fromdate=start_date,
        todate=end_date,
    )
    cb.adddata(data0)

    data1 = btfeeds.YahooFinanceCSVData(
        dataname=data_path1,
        fromdate=start_date,
        todate=end_date,
    )
    cb.adddata(data1)

    # Add the trading strategy to the engine
    cb.addstrategy(OUPairsTradingStrategy)

    cb.addanalyzer(btanalyzers.Returns, _name="returns")
    cb.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe_ratio", annualize=False, timeframe=bt.TimeFrame.Minutes, compression=60)  # Workaround for hourly data.
    cb.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe_ratio_annual", annualize=True, timeframe=bt.TimeFrame.Minutes, compression=60)  # Workaround for hourly data.

    # Set starting cash balance
    cb.broker.setcash(100.0)

    # Percentage commission - broker fees: 0.005 is 0.5%.
    cb.broker.setcommission(0.01)

    initial_portfolio_value = cb.broker.getvalue()
    initial_cash = cb.broker.getcash()

    strategies = cb.run()
    strategy = strategies[0]

    # Results
    print(f"Returns: {strategy.analyzers.returns.get_analysis()}")
    print(f"Sharpe Ratio: {strategy.analyzers.sharpe_ratio.get_analysis()}")
    print(f"Sharpe Ratio Annualised: {strategy.analyzers.sharpe_ratio_annual.get_analysis()}")
    print(f"Starting Portfolio Value: {initial_portfolio_value}")
    print(f"Final Portfolio Value: {cb.broker.getvalue()}")
    print(f"Initial Cash: {initial_cash}")
    print(f"Final Cash: {cb.broker.getcash()}")

    cb.plot()


if __name__ == "__main__":
    run()
