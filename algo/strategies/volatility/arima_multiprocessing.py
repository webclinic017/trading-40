import functools
import itertools
import multiprocessing
import seaborn as sns
import statsmodels.api as sm
import warnings
import yfinance as yf

sns.set_style("darkgrid")


def arima(p, d, q, data):
    # AR(p) I(d), MA(q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            arima_model = sm.tsa.arima.ARIMA(endog=data, order=(p, d, q)).fit()
            return (p, q), arima_model.aic

        except Exception as e:
            print(f"p = {p}, q = {q}")
            print(e)


def arima_parallel(p_candidates, d_candidates, q_candidates, data, num_processes):
    data = data.copy()
    with multiprocessing.Pool(processes=num_processes) as pool:
        arima_pf = functools.partial(arima, data=data)
        res = pool.starmap(arima_pf, (itertools.product(p_candidates, d_candidates, q_candidates)))

    out = dict(res)
    return out


if __name__ == "__main__":

    # Test Data.
    num_processes = 4
    max_lag = 2
    p_candidates = range(max_lag)
    d_candidates = (0,)  # Already differenced to I(0): d=0.
    q_candidates = range(max_lag)

    df = yf.download("SPY", start="2022-01-01", end="2023-01-30", interval="1d")
    df.rename(columns={"Adj Close": "price", "Volume": "volume"}, inplace=True)
    df = df[["price"]]
    df[f"price_diff_{1}"] = df["price"].diff(periods=1)
    df.dropna(inplace=True)

    out = arima_parallel(p_candidates, d_candidates, q_candidates, df["price_diff_1"].values, num_processes)
    print(f"out = {out}")
