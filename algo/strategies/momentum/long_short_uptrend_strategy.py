import backtrader as bt
import pandas as pd


class LongShortUptrendStrategy(bt.Strategy):

    def __init__(self):
        print(f"Initial Position:\n {self.position}")

        # TODO: PercentageSizer.
        self.size = 1.0

        self.span_trend = 50
        self.span_long = 3
        self.span_short = 3

        # sma_trend = bt.ind.SMA(period=span_trend)
        # sma_long = bt.ind.SMA(period=span_long)
        # sma_short = bt.ind.SMA(period=span_short)
        # self.long_signal = bt.ind.CrossOver(sma1, sma2)
        self.sma_trend = bt.indicators.MovingAverageSimple(self.data, period=self.span_trend)
        self.sma_long = bt.indicators.MovingAverageSimple(self.data, period=self.span_long)
        self.sma_short = bt.indicators.MovingAverageSimple(self.data, period=self.span_short)

        self.order = None

    def next(self):
        # if len(self.data) < self.span_trend:
        #     return
        # Do nothing if the market is not trending upwards in the long term.
        # if self.data.close[0] < self.sma_trend[0]:
        #     return

        # Do nothing if an order is pending.
        # if self.order is not None:
        #     return
        # print(self.broker.get_cash(), self.data.close[0], self.data.close[0])

        # print("POSITION: ", self.position)

        # Check if in market.
        # if not self.position:
        # if self.position:
        #     return

        # Long if current price is below MA.
        if self.data.close[0] < self.sma_long[0]:
            self.log(f"BUY CREATE: {self.data.close[0]}")

            # Track the newly created order to prevent creating duplicates.
            self.order = self.buy(size=self.size)

        # else:
        # Short if current price is above MA.
        if self.data.close[0] > self.sma_short[0]:
            self.log(f"SELL CREATE: {self.data.close[0]}")

            # Track the newly created order to prevent creating duplicates.
            self.order = self.sell(size=self.size)

    def exit_market(self):
        print(f"\n\n{self.global_count} {self.count} EXITING MARKET")
        self.close(data=self.data, exectype=bt.Order.Market)

    def log(self, message: str, date_time=None):
        # if self.p.printout:
        date_time = date_time or self.data.datetime[0]
        date_time = bt.num2date(date_time)
        print('%s, %s' % (date_time.isoformat(), message))
