import backtrader as bt
import backtrader.analyzers as btanalyzers
import backtrader.feeds as btfeeds
import hydra
import matplotlib.pyplot as plt
import pytz
import yfinance as yf
from datetime import date, datetime
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from algo.strategies.mean_reversion.ou_pairs_strategy import OUPairsTradingStrategy


@hydra.main(config_path="configs", config_name="config_main")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    interval = cfg.data.interval

    asset0_path = Path(cfg.config_dir).joinpath(cfg.pairs["asset0"])
    asset1_path = Path(cfg.config_dir).joinpath(cfg.pairs["asset1"])
    cfg_asset0 = OmegaConf.load(asset0_path)
    cfg_asset1 = OmegaConf.load(asset1_path)

    # Backtrader needs datetime objects.
    today = date.today()
    end_date = today

    # Yahoo needs strings.
    end_date_str = end_date.strftime("%Y-%m-%d")
    start_date_str = cfg.pairs.start_date
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

    # Download and cache data.
    df0_raw = yf.download(cfg_asset0.ticker, start=start_date_str, end=end_date_str, interval=interval)
    df1_raw = yf.download(cfg_asset1.ticker, start=start_date_str, end=end_date_str, interval=interval)
    assert len(df0_raw) > 0
    assert len(df1_raw) > 0
    assert df0_raw.isna().sum().sum() == 0.0
    assert df1_raw.isna().sum().sum() == 0.0

    # TODO!
    # df0_raw["Datetime"] = pd.to_datetime(df0_raw["Datetime"], format="%Y-%m-%d %H:%M:%S")
    # df1_raw["Datetime"] = pd.to_datetime(df1_raw["Datetime"], format="%Y-%m-%d %H:%M:%S")
    # df0_raw["Datetime"] = pd.to_datetime(df0_raw["Datetime"], format="%Y-%m-%d")
    # df1_raw["Datetime"] = pd.to_datetime(df1_raw["Datetime"], format="%Y-%m-%d")

    # Convert timezones (to UTC).
    if cfg.convert_timezone:
        df0_raw.index = df0_raw.index.tz_convert(None)
        df1_raw.index = df1_raw.index.tz_convert(None)

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

    def roll_date_wti_oil(month_df):
        days = month_df.index.day.values

        # Last trade is 4 business days before 25th if 25th is a Bday, else 5 business days before.
        expiry = 25
        last_trade = 21 if expiry in days else 20

        # Account for `last_trade` being on a weekend too.
        index = []
        while len(index) == 0:
            # index = month_df.index[month_df["day_of_month"] == last_trade]
            index = month_df.index[month_df.index.day == last_trade]
            last_trade -= 1

            assert last_trade > 0

        month_df.loc[index.strftime("%Y-%m-%d"), "roll_date"] = True

        return month_df

    df0["roll_date"] = False
    dfg0 = df0.groupby(by=[df0.index.month, df0.index.year])
    dfg0 = dfg0.apply(roll_date_wti_oil)

    df0 = dfg0

    # Save raw data for post-trade analysis.
    data_dir = Path.cwd().joinpath("data")
    Path.mkdir(data_dir)
    data_path0 = data_dir.joinpath("df0.csv")
    data_path1 = data_dir.joinpath("df1.csv")
    df0.to_csv(data_path0, index_label="datetime")
    df1.to_csv(data_path1, index_label="datetime")

    cb = bt.Cerebro()

    timeframes = {
        "1d": bt.TimeFrame.Days,
        "1h": bt.TimeFrame.Minutes,  # Couple with compression=60.
    }

    timezone = "UTC"
    data0 = btfeeds.YahooFinanceCSVData(
        name=cfg_asset0.ticker,
        dataname=data_path0,
        fromdate=start_date,
        todate=end_date,
        timeframe=timeframes[interval],
        # compression=60,  # Minutes -> Hours.
        tz=pytz.timezone(timezone),
        dtformat=("%Y-%m-%d %H:%M:%S"),
        timeformat=("%H:%M:%S"),
    )
    ###
    # from backtrader.feeds import PandasData
    #
    # class RollDatePandas(PandasData):
    #     lines = ("roll_date",)
    #     params = (("roll_date", 7),)
    #
    # data = RollDatePandas(
    #     dataname=dfg0,
    #     datetime=0,
    #     open=1,
    #     high=2,
    #     low=3,
    #     close=4,
    #     roll_date=5,
    # )
    # class RollDateYahooFinanceCSVData(btfeeds.YahooFinanceCSVData):
    #     lines = ("roll_date",)
    #     params = (("roll_date", 8),)
    #
    # data0 = RollDateYahooFinanceCSVData(
    #     name=cfg_asset0.ticker,
    #     dataname=data_path0,
    #     fromdate=start_date,
    #     todate=end_date,
    #     timeframe=timeframes[interval],
    #     # compression=60,  # Minutes -> Hours.
    #     tz=pytz.timezone(timezone),
    #     dtformat=("%Y-%m-%d %H:%M:%S"),
    #     timeformat=("%H:%M:%S"),
    # )

    from backtrader.feeds import GenericCSVData

    class GenericCSV_PE(GenericCSVData):

        # Add a 'pe' line to the inherited ones from the base class
        lines = ('roll_date',)

        # openinterest in GenericCSVData has index 7 ... add 1
        # add the parameter to the parameters inherited from the base class
        params = (('roll_date', 8),)

    data0 = btfeeds.GenericCSVData(
        dataname=data_path0,
        fromdate=start_date,
        todate=end_date,
        # nullvalue=0.0,
        dtformat=('%Y-%m-%d'),
        datetime=0,
        high=1,
        low=2,
        open=3,
        close=4,
        volume=5,
        openinterest=-1,
        roll_date=7,
    )
    # cb.adddata(data)

    ###

    data1 = btfeeds.YahooFinanceCSVData(
        name=cfg_asset1.ticker,
        dataname=data_path1,
        fromdate=start_date,
        todate=end_date,
        timeframe=timeframes[interval],
        # compression=60,  # Minutes -> Hours.
        tz=pytz.timezone(timezone),
        dtformat=("%Y-%m-%d %H:%M:%S"),
        timeformat=("%H:%M:%S"),
    )
    cb.adddata(data0)
    cb.adddata(data1)

    dt = cfg.strategy.dt
    if isinstance(dt, str):
        dt = eval(dt)

    # Add the trading strategy to the engine
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
        timeframe=bt.TimeFrame.Minutes,
        compression=60,  # Workaround for hourly data.
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
