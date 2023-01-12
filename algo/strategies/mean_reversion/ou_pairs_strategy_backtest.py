import backtrader as bt
import backtrader.analyzers as btanalyzers
import backtrader.feeds as btfeeds
import hydra
import matplotlib.pyplot as plt
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
    df0 = yf.download(cfg.pairs.ticker0, start=start_date_str, end=end_date_str, interval=interval)
    df1 = yf.download(cfg.pairs.ticker1, start=start_date_str, end=end_date_str, interval=interval)
    assert len(df0) > 0
    assert len(df1) > 0

    data_dir = Path.cwd().joinpath("data")
    Path.mkdir(data_dir)
    data_path0 = data_dir.joinpath("df0.csv")
    data_path1 = data_dir.joinpath("df1.csv")
    df0.to_csv(data_path0)
    df1.to_csv(data_path1)

    cb = bt.Cerebro()

    # Add only the TEST data streams to the engine
    data0 = btfeeds.YahooFinanceCSVData(dataname=data_path0, fromdate=start_date, todate=end_date)
    data1 = btfeeds.YahooFinanceCSVData(dataname=data_path1, fromdate=start_date, todate=end_date)
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

    fig3 = plt.figure()
    plt.plot(df.index, df["spread_zscore"], color="black", label="spread_zscore")
    plt.plot(df.index, df["spread_zscore_all_data"], color="black", linestyle="dotted", label="spread_zscore_all_data")
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
