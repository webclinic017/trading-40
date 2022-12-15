import backtrader as bt
import pandas as pd


class LongShortUptrendStrategy(bt.Strategy):

    def __init__(self):
        print(f"Initial Position:\n {self.position}")

        # TODO: PercentageSizer.
        self.size = 1.0

        self.span_trend = 200
        self.span_long = 10
        self.span_short = 10

        sma_signal = bt.indicators.MovingAverageSimple(self.data, period=self.span_long)
        self.crossover = bt.indicators.CrossOver(self.data, sma_signal)
        self.sma_trend = bt.indicators.MovingAverageSimple(self.data, period=self.span_trend)
        # self.order = None

    def next(self):
        # Do nothing if the market is not trending upwards in the long term.
        if self.data.close[0] < self.sma_trend[0]:
            return

        # Assumes uses of algo.momentum.long_short_sizer.FixedReverser
        if self.crossover > 0:
            self.log(f"BUY CREATE: {self.data.close[0]}")
            self.buy()

        elif self.crossover < 0:
            self.log(f"SELL CREATE: {self.data.close[0]}")
            self.sell()

    def log(self, message: str, date_time=None):
        date_time = date_time or self.data.datetime[0]
        date_time = bt.num2date(date_time)
        print('%s, %s' % (date_time.isoformat(), message))
