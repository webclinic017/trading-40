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
            multiplier0: float,
            multiplier1: float,
            z_entry: float,
            z_exit: float,
            risk_per_trade: float,
            num_train_initial: int,
            num_test: int,
            use_fixed_train_size: bool,
            require_cointegrated: bool,
            trade_integer_quantities: bool,
            roll_at_expiry: bool,
            dt: float,
            A: float,
    ):
        super().__init__(self)

        # Asset names.
        self.asset0 = asset0
        self.asset1 = asset1

        # Derivative contract multipliers.
        self.multiplier0 = multiplier0
        self.multiplier1 = multiplier1

        # Trading Signals.
        assert z_entry > z_exit
        self.z_entry = z_entry
        self.z_exit = z_exit

        # TODO
        self.z_stop_loss = z_entry + 0.01

        # Maximum risk per trade pair (percentage of cash).
        self.risk_per_trade = risk_per_trade

        # Data Parameters.
        self.num_train_initial = num_train_initial  # Number of data points needed to train the initial model.
        self.num_test = num_test  # Number of data points to elapse before retraining the model.
        self.use_fixed_train_size = use_fixed_train_size  # Number of data points to elapse before retraining the model.

        # Initial run conditions.
        self.model_trained = False  # Check if the model has been trained at least once.

        # Start at -1 because we pre-crement to 0 before anything else happens/
        self.step = -1         # Resets.
        self.global_step = -1  # Never resets.

        # Initial OU conditions.
        self.alpha = None
        self.beta = None
        self.optimiser = OptimiserOU(A=A, dt=dt)

        # Force the strategy to only issue trades when the spread is cointegrated.
        self.require_cointegrated = require_cointegrated

        # Whether buy/sell quantities must be integer valued, e.g. futures.
        self.trade_integer_quantities = trade_integer_quantities

        # Handle expiring derivatives contracts.
        self.roll_at_expiry = roll_at_expiry

        # Initial time series.
        self.X = np.array([])   # Spread
        self.S0 = []            # Asset 0
        self.S1 = []            # Asset 1

        # Use a rolling vs. expanding training set.
        if self.use_fixed_train_size:
            self.S0 = deque(self.S0, maxlen=self.num_train_initial)
            self.S1 = deque(self.S1, maxlen=self.num_train_initial)

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

        self.df["roll_0"] = False
        self.df["roll_1"] = False

        self.is_cointegrated = False

    def next(self):
        self.step += 1
        self.global_step += 1

        assert not (self.is_long and self.is_short)
        # print(f"order_buy = {self.order_buy}, order_sell = {self.order_sell}")
        # print(f"cash = {self.broker.get_cash()}")

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

        # Roll open positions if needed.
        if self.roll_at_expiry:
            self.roll()

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
        self.df.loc[self.global_step, "pf_value"] = self.broker.fundshares * self.broker.fundvalue
        nav_0 = self.positionsbyname[self.asset0].price * self.positionsbyname[self.asset0].size
        nav_1 = self.positionsbyname[self.asset1].price * self.positionsbyname[self.asset1].size
        cash = self.broker.get_cash()
        self.df.loc[self.global_step, "NAV"] = nav_0 + nav_1 + cash

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

        if not self.is_cointegrated and self.require_cointegrated:
            return

        # Open a long portfolio position on the lower boundary of the entry region.
        if z_score <= -self.z_entry and not self.in_market:
            # TODO: check conditions
            # if -self.z_stop_loss <= z_score <= -self.z_entry and not self.in_market:
            self.long_portfolio(z_score)

        # Open a short portfolio position on the upper boundary of the entry region.
        # Original:
        elif z_score >= self.z_entry and not self.in_market:
            # TODO: check conditions
            # elif self.z_stop_loss >= z_score >= self.z_entry and not self.in_market:
            self.short_portfolio(z_score)

        # Close any open long portfolio positions on the lower boundary of the exit region.
        elif z_score >= -self.z_exit and self.is_long:
            self.exit_market(z_score, exit_mode="exit_long")

        # Close any open short portfolio positions on the upper boundary of the exit region.
        elif z_score <= self.z_exit and self.is_short:
            self.exit_market(z_score, exit_mode="exit_short")

        # TODO: make it so you don't re-enter the position you just stopped.
        # elif z_score <= -self.z_stop_loss and self.is_long:
        #     # elif z_score <= -self.z_stop_loss and self.in_market:
        #     self.exit_market(z_score, exit_mode="stop_long")
        #
        # elif z_score >= self.z_stop_loss and self.is_short:
        #     # elif z_score >= self.z_stop_loss and self.in_market:
        #     self.exit_market(z_score, exit_mode="stop_short")

    def train(self):
        S0 = np.array(self.S0)
        S1 = np.array(self.S1)

        hp, _ = self.optimiser.optimise(asset1=S0, asset2=S1)

        # Record OU params - fill forward in plots.
        self.df.loc[self.global_step, "theta"] = hp.ou_params.theta
        self.df.loc[self.global_step, "mu"] = hp.ou_params.mu

        # Update hedge parameters: (alpha, beta) for use until the next re-training.
        self.alpha = hp.alpha
        self.beta = hp.beta

        # Update purchase parameters: (A, B) for use until the next re-training.
        self.A = hp.A
        self.B = hp.B

        # (Re-)Compute historic spread using (new) hedging parameters.
        self.X = self.alpha * S0 - self.beta * S1

        self.is_cointegrated = pretrade_checks(S0, S1, self.X)
        self.df.loc[self.global_step, "cointegrated"] = self.is_cointegrated

        # Reset counter for ongoing training.
        self.step = 0

        # Record that the model has been trained at least once.
        self.model_trained = True

        # Record when the model was (re-)trained
        self.df.loc[self.global_step, "train"] = 1

    def long_portfolio(self, z_score):
        # print(f"LONG: in_market = {self.in_market}, is_order_pending = {self.is_order_pending}")
        # Do nothing if already in the market or if orders are already open.
        if self.is_long or self.is_order_pending:
            return

        # Buy asset S0 and sell asset S1.
        print(f"{self.global_step} {self.step} LONG PORTFOLIO")
        quantity0, quantity1 = self.quantities()
        print("-"*100, f" LONG quantities: {quantity0}, {quantity1}")

        if quantity0 == 0.0 and quantity1 == 0.0:
            print("Both assets yielded 0 positions.")
            return

        # fake_size = 1.0
        # self.order_buy = self.buy(data=self.data0, size=fake_size, exectype=bt.Order.Market)
        # self.order_sell = self.sell(data=self.data1, size=fake_size, exectype=bt.Order.Market)
        self.order_buy = self.buy(data=self.data0, size=quantity0, exectype=bt.Order.Market)
        self.order_sell = self.sell(data=self.data1, size=quantity1, exectype=bt.Order.Market)
        self.df.loc[self.global_step, "enter_long"] = z_score

    def short_portfolio(self, z_score):
        # print(f"SHORT: in_market = {self.in_market}, is_order_pending = {self.is_order_pending}")
        # Do nothing if already in the market or if orders are already open.
        if self.is_short or self.is_order_pending:
            return

        # Sell asset S0 and buy asset S1.
        print(f"{self.global_step} {self.step} SHORT PORTFOLIO")
        quantity0, quantity1 = self.quantities()
        print("-"*100, f" SHORT quantities: {quantity0}, {quantity1}")

        if quantity0 == 0.0 and quantity1 == 0.0:
            print("Both assets yielded 0 positions.")
            return

        fake_size = 1.0
        # self.order_sell = self.sell(data=self.data0, size=fake_size, exectype=bt.Order.Market)
        # self.order_buy = self.buy(data=self.data1, size=fake_size, exectype=bt.Order.Market)
        self.order_sell = self.sell(data=self.data0, size=quantity0, exectype=bt.Order.Market)
        self.order_buy = self.buy(data=self.data1, size=quantity1, exectype=bt.Order.Market)
        self.df.loc[self.global_step, "enter_short"] = z_score

    def exit_market(self, z_score, exit_mode):
        print(f"EXIT: in_market = {self.in_market}, is_order_pending = {self.is_exit_order_pending}")

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

        elif order.status in [order.Expired, order.Canceled]:
            self.log(f"{order.Status[order.status]}", order)

        elif order.status == order.Margin:
            # TODO: log and plot margin-call sizes.
            # self.df.loc[self.global_step, "margin"] = order.size
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

    def notify_trade(self, trade):
        trade_msg = f"{bt.num2date(trade.dtopen)} global_step = {self.global_step} Trade ID: {trade.ref} "
        trade_msg += "LONG" if trade.long else "SHORT"
        trade_msg += " OPENED " if trade.isopen else " CLOSED "
        trade_msg += f"size = {trade.size}, price = {trade.price:.2f}, value = {trade.value:.2f}, "
        trade_msg += f"PNL = {trade.pnl:.3f}, PNL_commission = {trade.pnlcomm:.3f}"
        print(trade_msg)

        # if trade.isopen:
        #     print("o")

        if not trade.isopen:
            print(f"\t{bt.num2date(trade.dtclose)} Close - Market prices: p0 = {self.S0[-1]:.2f}, p1 = {self.S1[-1]:.2f}")

            # TODO: tmp (dangerous) hack to find the 2nd asset.
            # No longer works with rolling active.
            # if trade.ref % 2 == 0:
            #     print(f"DEBUG: p1 - trade.price = {self.S1[-1]:.2f} - {trade.price:.2f} = {self.S1[-1] - trade.price:.2f}")
            #     assert False, "fix this"

    def quantities(self):
        # Maximum capital to risk on a trade, defined as a long and a short pair, since both borrowed on margin.
        bet = self.broker.get_cash() * self.risk_per_trade

        # Target ratio of dollars spent on each asset, computed by OU Process.
        r = self.A / self.B

        # Current tick price of each contract (not considering amount of commodity bought by contract).
        contract_p0 = self.S0[-1]
        contract_p1 = self.S1[-1]

        # Cost to buy/sell outright, without margin.
        p0 = contract_p0 * self.multiplier0
        p1 = contract_p1 * self.multiplier1

        # TODO: might be more accurate to set the `price` as the margin per contract.

        # Ideal quantity of each asset to maintain OU-generated ratio, with total spend as near to `bet` as possible.
        n0 = (r/(1+r)) * (bet/p0)
        n1 = (1/(1+r)) * (bet/p1)

        # Realised values if fractional contracts are allowed.
        bet_actual = n0 * p0 + n1 * p1
        ratio_actual = (n0 * p0) / (n1 * p1)

        msg = f"Target \tratio = {r:.3f}, bet = {bet:.3f} \nActual \tratio = {ratio_actual:.3f}, bet = {bet_actual:.2f}, n0 = {n0:.2f}, n1 = {n1:.2f}"
        ou_msg = f"\tA/B = {self.A/self.B:.3f}, alpha/beta = {self.alpha/self.beta:.3}, n0/n1 = {n0/n1:.3f}"

        if self.trade_integer_quantities:
            n0 = round(n0)
            n1 = round(n1)
            bet_actual = n0 * p0 + n1 * p1
            ratio_actual = (n0 * p0) / (n1 * p1) if n1 != 0.0 else np.inf

            msg += f"\nRounded ratio = {ratio_actual:.3f}, bet = {bet_actual:.2f}, n0 = {n0:.2f}, n1 = {n1:.2f}"
            ou_msg += f", n0_round/n1_round = "
            ou_msg += f"{n0/n1:.3f}" if n1 != 0.0 else f"{np.inf}"

        print(msg)
        print(ou_msg)

        if n0 > 0.0 or n1 > 0.0:
            spread_actual = n0 * np.array(self.S0) - n1 * np.array(self.S1)
            _ = pretrade_checks(self.S0, self.S1, spread_actual)

        return n0, n1

    def roll(self):
        if not self.in_market:
            return

        if self.data0.roll_date == 1:
            print("Rolling asset 0...")
            self.df.loc[self.global_step, "roll_0"] = True

            # Cache size of position to reopen.
            size0 = self.positionsbyname[self.asset0].size

            if self.order_close0 is not None:
                self.order_close0 = self.close(data=self.data0, exectype=bt.Order.Market)

            # Do nothing if size is 0.
            if size0 > 0.0:
                self.order_buy = self.buy(data=self.data0, size=size0, exectype=bt.Order.Market)

            elif size0 < 0.0:
                self.order_sell = self.sell(data=self.data0, size=np.abs(size0), exectype=bt.Order.Market)

        if self.data1.roll_date == 1:
            print("Rolling asset 1...")
            self.df.loc[self.global_step, "roll_1"] = True

            # Cache size of position to reopen.
            size1 = self.positionsbyname[self.asset1].size

            if self.order_close1 is not None:
                self.order_close1 = self.close(data=self.data1, exectype=bt.Order.Market)

            # Do nothing if size is 0.
            if size1 > 0.0:
                self.order_sell = self.sell(data=self.data1, size=size1, exectype=bt.Order.Market)

            elif size1 < 0.0:
                self.order_buy = self.buy(data=self.data1, size=np.abs(size1), exectype=bt.Order.Market)

