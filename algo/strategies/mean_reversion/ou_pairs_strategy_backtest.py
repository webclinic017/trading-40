import backtrader as bt
import backtrader.analyzers as btanalyzers
import backtrader.feeds as btfeeds
import hydra
import matplotlib.pyplot as plt
import pandas as pd
import pytz
import yfinance as yf
from datetime import date, timedelta
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from algo.strategies.mean_reversion.ou_pairs_strategy import OUPairsTradingStrategy


@hydra.main(config_path="configs", config_name="config_main")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    interval = cfg.data.interval
    num_data_full = cfg.data.num_data_full

    # Backtrader needs datetime objects.
    today = date.today()
    end_date = today
    start = {
        "1m": today - timedelta(minutes=num_data_full),
        "1h": today - timedelta(hours=num_data_full),
        "1d": today - timedelta(days=num_data_full),
    }
    start_date = start[interval]

    # Yahoo needs strings.
    end_date_str = end_date.strftime("%Y-%m-%d")
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Download and cache data.
    df0_raw = yf.download(cfg.pairs.ticker0, start=start_date_str, end=end_date_str, interval=interval)
    df1_raw = yf.download(cfg.pairs.ticker1, start=start_date_str, end=end_date_str, interval=interval)
    assert len(df0_raw) > 0
    assert len(df1_raw) > 0
    assert df0_raw.isna().sum().sum() == 0.0
    assert df1_raw.isna().sum().sum() == 0.0

    # Ensure timestamps align.
    lsuffix = "_0"
    rsuffix = "_1"
    data_aligned_df = df0_raw.join(df1_raw, how="left", lsuffix=lsuffix, rsuffix=rsuffix)

    # Drop rows where entries are missing for either asset.
    data_aligned_df.dropna(inplace=True)
    df0 = data_aligned_df.filter(like=lsuffix)
    df1 = data_aligned_df.filter(like=rsuffix)

    # Restore original names (for backtrader).
    df0.columns = [col.strip(lsuffix) for col in df0.columns]
    df1.columns = [col.strip(rsuffix) for col in df1.columns]

    df0.index = df0.index.tz_convert(None)
    df1.index = df1.index.tz_convert(None)

    data_dir = Path.cwd().joinpath("data")
    Path.mkdir(data_dir)
    data_path0 = data_dir.joinpath("df0.csv")
    data_path1 = data_dir.joinpath("df1.csv")

    # # df0.index.rename("Datetime", inplace=True)
    # df0["Datetime"] = df0.index
    # df0["Time"] = pd.to_datetime(df0["Datetime"]).dt.time
    # df0["Date"] = pd.to_datetime(df0["Datetime"]).dt.date
    # # df0.set_index("Date", drop=True)
    # df0.drop(columns="Datetime", inplace=True)
    #
    # df1["Datetime"] = df1.index
    # df1.loc["Time"] = pd.to_datetime(df1["Datetime"]).dt.time
    # df1.loc["Date"] = pd.to_datetime(df1["Datetime"]).dt.date
    # # df1.set_index("Date", drop=True)
    # df1.drop(columns="Datetime", inplace=True)

    # df0.index.name = "Date"
    # df1.index.name = "Date"
    # df0.to_csv(data_path0, index_label="date")
    # df1.to_csv(data_path1, index_label="date")
    df0.to_csv(data_path0, index_label="datetime")
    df1.to_csv(data_path1, index_label="datetime")

    cb = bt.Cerebro()

    # Add only the TEST data streams to the engine
    timezone = "UTC"
    # data0 = btfeeds.YahooFinanceCSVData(dataname=data_path0, fromdate=start_date, todate=end_date, tz=pytz.timezone(timezone))
    # data1 = btfeeds.YahooFinanceCSVData(dataname=data_path1, fromdate=start_date, todate=end_date, tz=pytz.timezone(timezone))
    # data0 = btfeeds.YahooFinanceCSVData(dataname=data_path0, fromdate=start_date, todate=end_date)
    # data1 = btfeeds.YahooFinanceCSVData(dataname=data_path1, fromdate=start_date, todate=end_date)
    data0 = btfeeds.YahooFinanceCSVData(
        dataname=data_path0,
        fromdate=start_date,
        todate=end_date,
        timeframe=bt.TimeFrame.Minutes,
        # compression=60,  # Minutes -> Hours.
        tz=pytz.timezone(timezone),
        dtformat=("%Y-%m-%d %H:%M:%S"),
        timeformat=("%H:%M:%S"),
    )
    data1 = btfeeds.YahooFinanceCSVData(
        dataname=data_path1,
        fromdate=start_date,
        todate=end_date,
        timeframe=bt.TimeFrame.Minutes,
        # compression=60,  # Minutes -> Hours.
        tz=pytz.timezone(timezone),
        dtformat=("%Y-%m-%d %H:%M:%S"),
        timeformat=("%H:%M:%S"),
    )
    """
    data0 = btfeeds.GenericCSVData(
        dataname=data_path0,
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        # time=6,
        tz=pytz.timezone(timezone),
        dtformat="%Y-%m-%d %H:%M:%S",
        tmformat="%H:%M:%S",
    )
    data1 = btfeeds.GenericCSVData(
        dataname=data_path1,
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        # time=6,
        tz=pytz.timezone(timezone),
        dtformat="%Y-%m-%d %H:%M:%S",
        tmformat="%H:%M:%S",
    )
    """
    # data0 = btfeeds.GenericCSVData(dataname=data_path0, fromdate=start_date, todate=end_date)
    # data1 = btfeeds.GenericCSVData(dataname=data_path1, fromdate=start_date, todate=end_date)

    # data0 = btfeeds.PandasData(dataname=df0)
    # data1 = btfeeds.PandasData(dataname=df1)
    cb.adddata(data0)
    cb.adddata(data1)

    # Add the trading strategy to the engine
    cb.addstrategy(
        OUPairsTradingStrategy,
        z_entry=cfg.pairs.z_entry,
        z_exit=cfg.pairs.z_exit,
        num_train_initial=cfg.data.num_train_initial,
        num_test=cfg.data.num_test,
        use_fixed_train_size=cfg.pairs.use_fixed_train_size,
        dt=cfg.strategy.dt,
        A=cfg.strategy.A,
    )

    cb.addanalyzer(btanalyzers.Returns, _name="returns")
    cb.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe_ratio", annualize=False, timeframe=bt.TimeFrame.Minutes, compression=60)  # Workaround for hourly data.
    cb.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe_ratio_annual", annualize=True, timeframe=bt.TimeFrame.Minutes, compression=60)  # Workaround for hourly data.

    cb.broker.setcash(cfg.broker.cash_initial)
    cb.broker.setcommission(cfg.broker.commission)
    initial_portfolio_value = cb.broker.getvalue()
    initial_cash = cb.broker.getcash()

    strategies = cb.run()
    strategy = strategies[0]

    # Results.
    print(f"Returns: {strategy.analyzers.returns.get_analysis()}")
    print(f"Sharpe Ratio: {strategy.analyzers.sharpe_ratio.get_analysis()}")
    print(f"Sharpe Ratio Annualised: {strategy.analyzers.sharpe_ratio_annual.get_analysis()}")
    print(f"Starting Portfolio Value: {initial_portfolio_value}")
    print(f"Final Portfolio Value: {cb.broker.getvalue()}")
    print(f"Initial Cash: {initial_cash}")
    print(f"Final Cash: {cb.broker.getcash()}")

    df = strategy.df
    df.to_csv("plot_data.csv")

    # Drop the rows of all NaNs, keep individual (row,col) NaN entries.
    # df.drop(columns=["Unnamed: 0"], inplace=True)
    df.dropna(axis=0, how="all", inplace=True)

    fig = plt.figure()
    plt.plot(df.index, df["spread_zscore"], color="black", label="spread_zscore")
    plt.scatter(df.index, df["long"], color="deepskyblue", marker="^", label="long")
    plt.scatter(df.index, df["short"], color="orange", marker="v", label="short")
    plt.scatter(df.index, df["exit_long"], color="blue", marker="X", label="exit_long")
    plt.scatter(df.index, df["exit_short"], color="darkorange", marker="X", label="exit_short")
    xmin = df.index[0]
    xmax = df.index[-1]
    plt.hlines([-cfg.pairs.z_entry, cfg.pairs.z_entry], xmin=xmin, xmax=xmax, color="forestgreen", linestyle="dashed", label="z_entry")
    plt.hlines([-cfg.pairs.z_exit, cfg.pairs.z_exit], xmin=xmin, xmax=xmax, color="red", linestyle="dashed", label="z_exit")
    plt.title("Spread - Entries and Exits")
    plt.xlabel(f"Time Step ({interval})")
    plt.ylabel("Normalised Spread, Z")
    plt.legend()
    plt.savefig("spread_entries_exits.png")
    plt.show()

    cb.plot()


if __name__ == "__main__":
    run()
