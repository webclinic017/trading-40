from option_pricing_models.implied_volatility_black76 import implied_volatility_black76_european


if __name__ == "__main__":
    F_t = 50
    X_t = 55
    t = 1/12
    r = 0.04
    option_price_t = 6
    precision = 0.0001
    max_num_iterations = 100

    print(implied_volatility_black76_european("call", F_t, X_t, t, r, option_price_t, precision, max_num_iterations))
