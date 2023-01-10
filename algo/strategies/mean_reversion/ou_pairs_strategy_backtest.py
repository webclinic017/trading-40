import backtrader as bt
import backtrader.analyzers as btanalyzers
import backtrader.feeds as btfeeds
import hydra
from omegaconf import DictConfig, OmegaConf
import yfinance as yf
from datetime import date, timedelta

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

    data_path0 = "data/df1.csv"
    data_path1 = "data/df2.csv"
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

    cb.plot()


if __name__ == "__main__":
    run()
