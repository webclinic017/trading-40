from option_pricing_models.black76 import Black76Eur


if __name__ == "__main__":
    F_t = 50
    X_t = 55
    t = 1 / 12
    r = 0.04
    vol = 0.20

    black76 = Black76Eur()
    print(black76.call(F_t=F_t, X_t=X_t, t=t, r=r, vol=vol), black76.put(F_t=F_t, X_t=X_t, t=t, r=r, vol=vol))
