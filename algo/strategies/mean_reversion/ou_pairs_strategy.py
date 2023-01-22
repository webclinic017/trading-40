import backtrader as bt
import numpy as np
import pandas as pd
from collections import deque
from algo.models.sde.ornstein_uhlenbeck_model_optimisation import OptimiserOU
from algo.strategies.mean_reversion.base_pairs_strategy import PairsTradingStrategy, pretrade_checks


class OUPairsTradingStrategy(PairsTradingStrategy):

    def __init__(
            self,
            asset0: str,
            asset1: str,
            z_entry: float,
            z_exit: float,
            num_train_initial: int,
            num_test: int,
            use_fixed_train_size: bool,
            dt: float,
            A: float,
    ):
        super().__init__(self)

        # Asset names.
        self.asset0 = asset0
        self.asset1 = asset1

        # Initial run conditions
        self.model_trained = False  # Check if the model has been trained at least once.

        # Start at -1 because we pre-crement to 0 before anything else happens/
        self.step = -1         # Resets.
        self.global_step = -1  # Never resets.

        self.num_train_initial = num_train_initial  # Number of data points needed to train the initial model.
        self.num_test = num_test  # Number of data points to elapse before retraining the model.
        self.use_fixed_train_size = use_fixed_train_size  # Number of data points to elapse before retraining the model.

        # Initial OU conditions
        self.alpha = None
        self.beta = None
        self.optimiser = OptimiserOU(A=A, dt=dt)

        # Initial time series
        self.X = np.array([])   # Spread
        self.S0 = []            # Asset 0
        self.S1 = []            # Asset 1

        # Use a rolling vs. expanding training set.
        if self.use_fixed_train_size:
            self.S0 = deque(self.S0, maxlen=self.num_train_initial)
            self.S1 = deque(self.S1, maxlen=self.num_train_initial)

        # Trading Signals
        assert z_entry > z_exit
        self.z_entry = z_entry
        self.z_exit = z_exit

        columns = [
            "datetime_bt",
            "S0",
            "S1",
            "spread",
            "spread_zscore",
            "enter_long",       # Signal to enter long portfolio position.
            "enter_short",      # Signal to enter short portfolio position.
            "exit_long",        # Signal to exit long portfolio position.
            "exit_short",       # Signal to exit short portfolio position.
            "is_long",          # Track if the strategy is currently long the portfolio.
            "is_short"          # Track if the strategy is currently short the portfolio.
        ]
        output_data = np.full((len(self.data.array), len(columns)), np.nan)
        self.df = pd.DataFrame(data=output_data)
        self.df.columns = columns

        # self.is_cointegrated = False

    def next(self):

        # if any(order is not None for order in [self.order_buy, self.order_sell, self.order_close0, self.order_close1]):
        #     print("-"*40)
        #     print(f"order buy: {self.order_buy}")
        #     print(f"order sell: {self.order_sell}")
        #     print(f"order close0: {self.order_close0}")
        #     print(f"order close1: {self.order_close1}")
        #     print("-"*40)

        self.step += 1
        self.global_step += 1

        # Current prices of asset0 and asset1.
        p0 = self.data0.close[0]
        p1 = self.data1.close[0]

        assert p0 is not None
        assert p1 is not None

        # Store most recent asset prices.
        self.S0.append(p0)
        self.S1.append(p1)

        # Train OU-Model for the first time.
        if self.step >= self.num_train_initial and not self.model_trained:
            self.train()

        # Do nothing if there has not been enough data to train the model.
        if not self.model_trained:
            return

        # (Re-)Train OU-Model with updated data as soon as we are not in the market.
        if self.step >= self.num_test \
                and not self.in_market \
                and self.order_buy is None \
                and self.order_sell is None:
            self.train()

        self.df.loc[self.global_step, "datetime_bt"] = bt.num2date(self.datetime[0])
        # Logging price and size like this for now. Consider that it's not right (realised).
        self.df.loc[self.global_step, "S0_price"] = self.positionsbyname[self.asset0].price
        self.df.loc[self.global_step, "S0_size"] = self.positionsbyname[self.asset0].size
        self.df.loc[self.global_step, "S1_price"] = self.positionsbyname[self.asset1].price
        self.df.loc[self.global_step, "S1_size"] = self.positionsbyname[self.asset1].size
        # End.
        self.df.loc[self.global_step, "cash"] = self.broker.get_cash()
        self.df.loc[self.global_step, "S0"] = self.S0[-1]
        self.df.loc[self.global_step, "S1"] = self.S1[-1]
        self.df.loc[self.global_step, "is_long"] = self.is_long
        self.df.loc[self.global_step, "is_short"] = self.is_short

        # X will have been populated during training.
        current_spread = self.alpha * p0 - self.beta * p1
        self.X = np.append(self.X, current_spread)

        # Current z_score.
        X = self.X[-self.num_train_initial:] if self.use_fixed_train_size else self.X
        z_score = (current_spread - np.mean(X)) / np.std(X)

        self.df.loc[self.global_step, "spread"] = current_spread  # Equiv. X[-1]
        self.df.loc[self.global_step, "spread_mean"] = np.mean(X)
        self.df.loc[self.global_step, "spread_std"] = np.std(X)
        self.df.loc[self.global_step, "spread_zscore"] = z_score
        self.df.loc[self.global_step, "A"] = self.A
        self.df.loc[self.global_step, "B"] = self.B
        self.df.loc[self.global_step, "alpha"] = self.alpha
        self.df.loc[self.global_step, "beta"] = self.beta

        # if not self.is_cointegrated:
        #     return

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

    def train(self):
        print("-"*20, "Training")
        S0 = np.array(self.S0)
        S1 = np.array(self.S1)

        hp, _ = self.optimiser.optimise(asset1=S0, asset2=S1)

        # Record OU params - fill forward in plots.
        self.df.loc[self.global_step, "theta"] = hp.ou_params.theta

        # Update hedge parameters: (alpha, beta) for use until the next re-training.
        self.alpha = hp.alpha
        self.beta = hp.beta

        # Update purchase parameters: (A, B) for use until the next re-training.
        self.A = hp.A
        self.B = hp.B

        # (Re-)Compute historic spread using (new) hedging parameters.
        self.X = self.alpha * S0 - self.beta * S1

        pretrade_checks(S0, S1, self.X)
        # self.is_cointegrated = pretrade_checks(S0, S1, self.X)

        # Reset counter for ongoing training.
        self.step = 0

        # Record that the model has been trained at least once.
        self.model_trained = True

        # Record when the model was (re-)trained
        self.df.loc[self.global_step, "train"] = 1

    def long_portfolio(self, z_score):
        # Do nothing if already in the market or if orders are already open.
        if self.in_market or self.is_order_pending:
            return

        # Buy asset S0 and sell asset S1.
        print(f"{self.global_step} {self.step} LONG PORTFOLIO")
        self.order_buy = self.buy(data=self.data0, size=self.A, exectype=bt.Order.Market)
        self.order_sell = self.sell(data=self.data1, size=self.B, exectype=bt.Order.Market)
        self.df.loc[self.global_step, "enter_long"] = z_score

    def short_portfolio(self, z_score):
        # Do nothing if already in the market or if orders are already open.
        if self.in_market or self.is_order_pending:
            return

        # Sell asset S0 and buy asset S1.
        print(f"{self.global_step} {self.step} SHORT PORTFOLIO")
        self.order_sell = self.sell(data=self.data0, size=self.A, exectype=bt.Order.Market)
        self.order_buy = self.buy(data=self.data1, size=self.B, exectype=bt.Order.Market)
        self.df.loc[self.global_step, "enter_short"] = z_score

    def exit_market(self, z_score, exit_mode):
        # Do nothing if already out of the market.
        if not self.in_market or self.is_exit_order_pending:
            return

        print(f"{self.global_step} {self.step} EXITING MARKET")
        self.order_close0 = self.close(data=self.data0, exectype=bt.Order.Market)
        self.order_close1 = self.close(data=self.data1, exectype=bt.Order.Market)
        self.df.loc[self.global_step, exit_mode] = z_score

    def notify_order(self, order):
        if order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            # Wait for further notifications.
            return

        elif order.status in [order.Expired, order.Canceled, order.Margin]:
            self.log(f"{order.Status[order.status]}", order)

        elif order.status == order.Completed:

            if (order.isbuy() and self.is_long) or (order.issell() and self.is_short):
                asset = "S0"

            if (order.isbuy() and self.is_short) or (order.issell() and self.is_long):
                asset = "S1"

            # TODO: breaks on exit when we are neither long nor short.
            # Will want to log this, though. Need to know the per trade PnL.
            # Track each trade entry to exit via IDs?

            # Log the execution stats for post-trade analysis. Note: position info is updated by backtrader.
            """
            self.df.loc[self.global_step, "datetime_bt"] = bt.num2date(order.executed.dt)
            self.df.loc[self.global_step, f"{asset}_price"] = order.executed.price
            self.df.loc[self.global_step, f"{asset}_size"] = order.executed.size
            self.df.loc[self.global_step, f"{asset}_value"] = self.df.loc[self.global_step, f"{asset}_price"] * self.df.loc[self.global_step, f"{asset}_size"]
            """

            if order.isbuy():
                self.log("BUY COMPLETE", order)

                # Allow new orders.
                if self.order_buy is not None and order.ref == self.order_buy.ref:
                    self.order_buy = None

                elif self.order_close0 is not None and order.ref == self.order_close0.ref:
                    self.order_close0 = None

                elif self.order_close1 is not None and order.ref == self.order_close1.ref:
                    self.order_close1 = None

            elif order.issell():
                self.log("SELL COMPLETE", order)

                # Allow new orders.
                if self.order_sell is not None and order.ref == self.order_sell.ref:
                    self.order_sell = None

                elif self.order_close0 is not None and order.ref == self.order_close0.ref:
                    self.order_close0 = None

                elif self.order_close1 is not None and order.ref == self.order_close1.ref:
                    self.order_close1 = None
