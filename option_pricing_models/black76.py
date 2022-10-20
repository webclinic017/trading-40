from trading.option_pricing_models.generalised_black_scholes import GeneralisedEuropeanBlackScholes


class Black76Eur(GeneralisedEuropeanBlackScholes):
    """
    Args:
        F_t: price of the underlying
        X_t: underlying strike price
        t:   time to expiration
        vol: implied volatility
        r:   risk-free rate
        q:   (potential) dividend payment
        b:   (potential) cost of carry

    Returns:
        option_value: calculated price of the option
        delta:        1st derivative of value wrt. the price of the underlying (F_t)
        gamma:        2nd derivative of value wrt. the price of the underlying (F_t)
        theta:        1st derivative of value wrt. time to expiration (t)
        vega:         1st derivative of value wrt. implied volatility (vol)
        rho:          1st derivative of value wrt. risk-free rate (r)
    """

    def __init__(self):
        super().__init__(name="Black76")

    def call(self, F_t, X_t, t, r, vol):
        b = self.cost_of_carry
        return super().call(F_t=F_t, X_t=X_t, t=t, r=r, b=b, vol=vol)

    def put(self, F_t, X_t, t, r, vol):
        b = self.cost_of_carry
        return super().put(F_t=F_t, X_t=X_t, t=t, r=r, b=b, vol=vol)

    @property
    def cost_of_carry(self):
        # No cost of carry on the underlying (futures contract).
        return 0
