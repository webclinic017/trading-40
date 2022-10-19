import pandas as pd
from datetime import datetime
from typing import Optional
from yahoofinancials import YahooFinancials

from trading.performance.performance_metrics import compute_returns


def compute_risk_free_rate(df: pd.DataFrame, method: str, constant: Optional[float] = None) -> pd.DataFrame:
    """
    Modifies dataframe `df` to add a column containing a risk-free rate of return for each time step.

    df:
    method:     function to use to compute risk-free returns
    constant:   only use IFF `method=="constant"`. This is the constant value that will be applied.
                    E.g. for 0% returns, use 0.0; for 1% returns, use 0.01, etc..
    """
    if method == "constant":
        assert constant is not None, "Specify a constant value of risk-free returns."
        df["risk_free_rate_t"] = constant

    elif method == "us_treasury_note_10yr":
        # Get US Treasury Note data
        start_date = datetime.strftime(df.index[0], format="%Y-%m-%d")
        end_date = datetime.strftime(df.index[-1], format="%Y-%m-%d")
        tnx_df = get_data_us_treasury_note_10yr(start_date, end_date)

        # Left join data on date
        df = pd.merge(df, tnx_df, left_index=True, right_index=True)

    else:
        raise ValueError("Unsupported method for risk free rate calculation")

    return df


def get_data_us_treasury_note_10yr(start_date: str, end_date: str, ticker: str = "TNX") -> pd.DataFrame:
    """
    Use the 10-Year US Treasury Note to indicate risk free return
    """
    yahoo_financials = YahooFinancials(ticker)
    data = yahoo_financials.get_historical_price_data(
        start_date=start_date, end_date=end_date, time_interval="daily"
    )
    df = pd.DataFrame(data[ticker]["prices"])

    # Reset date for convenience in joining with tick data.
    df["Date"] = pd.to_datetime(df["formatted_date"])
    df = df.drop("date", axis=1).set_index("Date")

    output_colname = "risk_free_returns"
    df = df \
        .pipe(compute_returns, price_colname="adjclose", output_colname=output_colname) \
        .fillna(0.0)

    return df[[output_colname]]
