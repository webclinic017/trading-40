import backtrader as bt
import backtrader.analyzers as btanalyzers
import hydra
import matplotlib.pyplot as plt
import pandas as pd
import pytz
from datetime import date, datetime
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from algo.strategies.mean_reversion.ou_pairs_strategy import OUPairsTradingStrategy
from etl.load_pairs import get_pairs_data_backtest
from etl.roll_date import get_roll_date_fn, RollDateGenericCSV


@hydra.main(config_path="configs", config_name="config_main")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    interval = cfg.data.interval
    cfg_asset0 = OmegaConf.load(Path(cfg.config_dir).joinpath(cfg.pairs["asset0"]))
    cfg_asset1 = OmegaConf.load(Path(cfg.config_dir).joinpath(cfg.pairs["asset1"]))

    end_date = date.today()
    start_date = cfg.pairs.start_date
    start_date = datetime.strptime(start_date, "%Y-%m-%d")

    ticker0 = cfg_asset0.ticker
    ticker1 = cfg_asset1.ticker
    df0, df1 = get_pairs_data_backtest(
        ticker0=ticker0,
        ticker1=ticker1,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        convert_timezone=cfg.convert_timezone,
        roll_fn0=get_roll_date_fn(ticker0),
        roll_fn1=get_roll_date_fn(ticker1),
    )

    # Save raw data for post-trade analysis.
    data_dir = Path.cwd().joinpath("data")
    Path.mkdir(data_dir)
    data_path0 = data_dir.joinpath(f"{cfg_asset0.ticker}.csv")
    data_path1 = data_dir.joinpath(f"{cfg_asset1.ticker}.csv")
    df0.to_csv(data_path0, index_label="datetime")
    df1.to_csv(data_path1, index_label="datetime")

    formats = {
        "1d": "%Y-%m-%d",
        "1h": "%Y-%m-%d %H:%M:%S",
    }
    timeframes = {
        "1d": bt.TimeFrame.Days,
        "1h": bt.TimeFrame.Minutes,  # Couple with compression=60.
    }

    timezone = "UTC"
    data0 = RollDateGenericCSV(
        dataname=data_path0,
        fromdate=start_date,
        todate=end_date,
        datetime=0,
        high=2,
        low=3,
        open=1,
        close=4,
        volume=6,
        openinterest=-1,
        roll_date=7,
        dtformat=formats[interval],
        timeframe=timeframes[interval],
        # compression=60,  # Minutes -> Hours.
        tz=pytz.timezone(timezone),
        timeformat="%H:%M:%S",
    )
    data1 = RollDateGenericCSV(
        dataname=data_path1,
        fromdate=start_date,
        todate=end_date,
        datetime=0,
        high=2,
        low=3,
        open=1,
        close=4,
        volume=6,
        openinterest=-1,
        roll_date=7,
        dtformat=formats[interval],
        timeframe=timeframes[interval],
        # compression=60,  # Minutes -> Hours.
        tz=pytz.timezone(timezone),
        timeformat="%H:%M:%S",
    )

    cb = bt.Cerebro()
    cb.adddata(data0)
    cb.adddata(data1)

    dt = cfg.strategy.dt
    if isinstance(dt, str):
        dt = eval(dt)

    # Add the trading strategy to the engine.
    cb.addstrategy(
        OUPairsTradingStrategy,
        asset0=cfg_asset0.ticker,
        asset1=cfg_asset1.ticker,
        multiplier0=cfg_asset0.multiplier,
        multiplier1=cfg_asset1.multiplier,
        z_entry=cfg.pairs.z_entry,
        z_exit=cfg.pairs.z_exit,
        risk_per_trade=cfg.risk_per_trade,
        num_train_initial=cfg.data.num_train_initial,
        require_cointegrated=cfg.strategy.require_cointegrated,
        trade_integer_quantities=cfg.strategy.trade_integer_quantities,
        roll_at_expiry=cfg.strategy.roll_at_expiry,
        num_test=cfg.data.num_test,
        use_fixed_train_size=cfg.pairs.use_fixed_train_size,
        dt=dt,
        A=cfg.strategy.A,
    )

    cb.addanalyzer(btanalyzers.Returns, _name="returns")
    cb.addanalyzer(
        btanalyzers.SharpeRatio,
        _name="sharpe_ratio_annual",
        annualize=True,
        timeframe=timeframes[interval],
        # compression=60,  # Workaround for hourly data.
        riskfreerate=cfg.risk_free_rate,
        convertrate=True,
    )

    cb.broker.setcash(cfg.broker.cash_initial)
    margin0 = cfg_asset0.margin if cfg_asset0.margin != "None" else None
    margin1 = cfg_asset1.margin if cfg_asset1.margin != "None" else None
    cb.broker.setcommission(commission=cfg.pairs.commission, margin=margin0, mult=cfg_asset0.multiplier, name=cfg_asset0.ticker)
    cb.broker.setcommission(commission=cfg.pairs.commission, margin=margin1, mult=cfg_asset1.multiplier, name=cfg_asset1.ticker)

    initial_portfolio_value = cb.broker.getvalue()
    initial_cash = cb.broker.getcash()

    strategies = cb.run()
    strategy = strategies[0]

    # Results.
    print(f"Returns: {strategy.analyzers.returns.get_analysis()}")
    print(f"Sharpe Ratio Annualised: {strategy.analyzers.sharpe_ratio_annual.get_analysis()}")
    print(f"Starting Portfolio Value: {initial_portfolio_value:.2f}")
    print(f"Final Portfolio Value: {cb.broker.getvalue():.2f}")
    print(f"Initial Cash: {initial_cash:.2f}")
    print(f"Final Cash: {cb.broker.getcash():.2f}")

    df = strategy.df
    df.to_csv("plot_data.csv")

    # Drop the rows of all NaNs, keep individual (row,col) NaN entries.
    df.dropna(axis=0, how="all", inplace=True)

    cb.plot()


if __name__ == "__main__":
    run()
