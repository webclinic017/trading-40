import backtrader as bt
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime

from algo.cointegration.augmented_dickey_fuller import adf_stationarity
from algo.cointegration.engle_granger import engle_granger_bidirectional
from algo.models.sde.ornstein_uhlenbeck_model_optimisation import OptimiserOU


# Important TODOs:
# [] For now we ignore roll costs.
# [] Smooth the price data? Hourly might be ok.


class OUPairsTradingStrategy(bt.Strategy):

    def __init__(
            self,
            z_entry: float,
            z_exit: float,
            num_train_initial: int,
            num_test: int,
            use_fixed_train_size: bool,
            dt: float,
            A: float,
    ):
        print(f"Initial Position:\n {self.position}")

        # Initial run conditions
        self.model_trained = False  # Check if the model has been trained at least once.
        self.count = 0         # Resets.
        self.global_count = 0  # Never resets.

        self.num_train_initial = num_train_initial  # Number of data points needed to train the initial model.
        self.num_test = num_test  # Number of data points to elapse before retraining the model.

        # Initial OU conditions
        self.alpha = None
        self.beta = None
        self.optimiser = OptimiserOU(A=A, dt=dt)

        # Initial time series
        self.X = np.array([])   # Spread
        self.S0 = []            # Asset 0
        self.S1 = []            # Asset 1

        # Use a rolling vs. expanding training set.
        if use_fixed_train_size:
            self.S0 = deque(self.S0, maxlen=self.num_train_initial)
            self.S1 = deque(self.S1, maxlen=self.num_train_initial)

        # Trading Signals
        self.z_entry = z_entry
        self.z_exit = z_exit

        # Tracking
        self.order_buy = None
        self.order_sell = None
        self.order_close0 = None
        self.order_close1 = None

        columns = ["S0", "S1", "spread", "spread_zscore", "long", "short", "exit_long", "exit_short"]
        output_data = np.full((len(self.data.array), len(columns)), np.nan)
        self.df = pd.DataFrame(data=output_data)
        self.df.columns = columns

    def next(self):
        self.count += 1
        self.global_count += 1

        # Current prices of asset0 and asset1.
        p0 = self.data0.close[0]
        p1 = self.data1.close[0]

        # Do nothing if at least 1 data stream has missing data.
        if p0 is None or p1 is None:
            print(f"NaN: p0={p0} | p1={p1}.")
            return

        # Store most recent asset prices.
        self.S0.append(p0)
        self.S1.append(p1)

        # Train OU-Model for the first time.
        if self.count >= self.num_train_initial and not self.model_trained:
            self.train()

        # Do nothing if there has not been enough data to train the model.
        if not self.model_trained:
            return

        # (Re-)Train OU-Model with updated data as soon as we are not in the market.
        if self.count >= self.num_test \
                and not self.in_market \
                and self.order_buy is None \
                and self.order_sell is None:
            self.train()

        # X will have been populated during training.
        current_spread = self.alpha * p0 - self.beta * p1
        self.X = np.append(self.X, current_spread)

        # Current z_score.
        # z_score = (current_spread - np.mean(self.X)) / np.std(self.X)
        num_train_initial = 2184  # TODO: properly.
        X = self.X[-num_train_initial:]
        z_score = (current_spread - np.mean(X)) / np.std(X)  # TODO: option.

        # Open a long portfolio position on the lower boundary of the entry region.
        if z_score <= -self.z_entry:
            self.long_portfolio(z_score)

        # Open a short portfolio position on the upper boundary of the entry region.
        elif z_score >= self.z_entry:
            self.short_portfolio(z_score)

        # Close any open long portfolio positions on the lower boundary of the exit region.
        elif z_score >= -self.z_exit and self.is_long:
            self.exit_market(z_score, exit_mode="exit_long")

        # Close any open short portfolio positions on the upper boundary of the exit region.
        elif z_score <= self.z_exit and self.is_short:
            self.exit_market(z_score, exit_mode="exit_short")

        self.df.loc[self.global_count, "S0"] = self.S0[0]
        self.df.loc[self.global_count, "S1"] = self.S1[0]
        self.df.loc[self.global_count, "spread"] = self.X[0]
        self.df.loc[self.global_count, "spread_zscore"] = z_score

    def train(self):
        S0 = np.array(self.S0)
        S1 = np.array(self.S1)

        hp, _ = self.optimiser.optimise(asset1=S0, asset2=S1)

        # Update hedge parameters: (alpha, beta) for use until the next re-training.
        self.alpha = hp.alpha
        self.beta = hp.beta

        # Update purchase parameters: (A, B) for use until the next re-training.
        self.A = hp.A
        self.B = hp.B

        # (Re-)Compute historic spread using (new) hedging parameters.
        self.X = self.alpha * S0 - self.beta * S1

        pretrade_checks(S0, S1, self.X)

        # Reset counter for ongoing training.
        self.count = 0

        # Record that the model has been trained at least once.
        self.model_trained = True

    def long_portfolio(self, z_score):
        # Do nothing if already in the market or if orders are already open.
        if self.in_market or self.order_buy is not None or self.order_sell is not None:
            return

        # Buy asset S0 and sell asset S1.
        print(f"{self.global_count} {self.count} LONG PORTFOLIO")
        self.order_buy = self.buy(data=self.data0, size=self.A, exectype=bt.Order.Market)
        self.order_sell = self.sell(data=self.data1, size=self.B, exectype=bt.Order.Market)
        self.df.loc[self.global_count, "long"] = z_score  # self.X[0]

    def short_portfolio(self, z_score):
        # Do nothing if already in the market or if orders are already open.
        if self.in_market or self.order_buy is not None or self.order_sell is not None:
            return

        # Sell asset S0 and buy asset S1.
        print(f"{self.global_count} {self.count} SHORT PORTFOLIO")
        self.order_sell = self.sell(data=self.data0, size=self.A, exectype=bt.Order.Market)
        self.order_buy = self.buy(data=self.data1, size=self.B, exectype=bt.Order.Market)
        self.df.loc[self.global_count, "short"] = z_score  # self.X[0]

    def exit_market(self, z_score, exit_mode):
        # Do nothing if already out of the market.
        if not self.in_market or self.order_close0 is not None or self.order_close1 is not None:
            return

        print(f"{self.global_count} {self.count} EXITING MARKET")
        self.order_close0 = self.close(data=self.data0, exectype=bt.Order.Market)
        self.order_close1 = self.close(data=self.data1, exectype=bt.Order.Market)
        # self.df.loc[self.global_count, "exit"] = z_score  # self.X[0]
        self.df.loc[self.global_count, exit_mode] = z_score  # self.X[0]

    @property
    def in_market(self):
        # return self.positionsbyname["df0"].size != 0 or self.positionsbyname["df1"].size != 0
        return self.is_long or self.is_short

    @property
    def is_long(self):
        return self.positionsbyname["df0"].size > 0.0 and self.positionsbyname["df1"].size < 0.0

    @property
    def is_short(self):
        return self.positionsbyname["df0"].size < 0.0 and self.positionsbyname["df1"].size > 0.0

    @staticmethod
    def log(message: str, date_time: datetime) -> None:
        date_time = bt.num2date(date_time)
        print(f"{date_time.isoformat()}", message)

    def notify_order(self, order):
        if order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            # Wait for further notifications.
            return

        if order.status == order.Completed:
            if order.isbuy():
                msg = f"BUY COMPLETE: {order.executed.price}"
                self.log(msg, order.executed.dt)

                # Allow new orders.
                if self.order_buy is not None and order.ref == self.order_buy.ref:
                    self.order_buy = None

                elif self.order_close0 is not None and order.ref == self.order_close0.ref:
                    self.order_close0 = None

                elif self.order_close1 is not None and order.ref == self.order_close1.ref:
                    self.order_close1 = None

            else:
                msg = f"SELL COMPLETE: {order.executed.price}"
                self.log(msg, order.executed.dt)

                # Allow new orders.
                if self.order_sell is not None and order.ref == self.order_sell.ref:
                    self.order_sell = None

                elif self.order_close0 is not None and order.ref == self.order_close0.ref:
                    self.order_close0 = None

                elif self.order_close1 is not None and order.ref == self.order_close1.ref:
                    self.order_close1 = None

        elif order.status in [order.Expired, order.Canceled, order.Margin]:
            self.log(f"{order.Status[order.status]}", order.executed.dt)


def pretrade_checks(S0, S1, spread) -> None:
    results = {
        # Test for stationarity in the actual spread series generated by the OU Model.
        "adf_c": adf_stationarity(spread, trend="c"),
        # Test for cointegration in the underlying asset price series.
        # "engle_granger_c": engle_granger_bidirectional(S0, S1, trend="c"),
    }

    print(results)
