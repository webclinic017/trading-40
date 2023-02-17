import warnings

import backtrader.feeds as btfeeds


class RollDateGenericCSV(btfeeds.GenericCSVData):
    lines = ("roll_date",)
    params = (("roll_date", 7),)


def identity(df, *args, **kwargs):
    return df


def roll_date_wti_oil(month_df, roll_date):
    """
    Last trade is 4 business days before 25th if 25th is a Bday, else 5 business days before.
    https://www.cmegroup.com/markets/energy/crude-oil/micro-wti-crude-oil.contractSpecs.html
    """
    days = month_df.index.day.values

    expiry = 25
    last_trade = 21 if expiry in days else 20

    # Account for `last_trade` being on a weekend too.
    index = []
    while len(index) == 0:
        index = month_df.index[month_df.index.day == last_trade]
        last_trade -= 1
        assert last_trade > 0

    month_df.loc[index.strftime("%Y-%m-%d"), roll_date] = 1

    return month_df


def roll_date_rbob(month_df, roll_date):
    """
    2nd last business day.
    https://www.cmegroup.com/markets/energy/refined-products/micro-rbob-gasoline.contractSpecs.html
    """
    month_df.loc[month_df.index[-2], "roll_date"] = 1
    return month_df


def get_roll_date_fn(ticker):
    roll_fn_registry = {
        "CL=F": roll_date_wti_oil,  # Note: this function is specifically for MCL=F (1 day before CL=F).
        "MCL=F": roll_date_wti_oil,
        "RB=F": roll_date_rbob,
        "MRB=F": roll_date_rbob,
    }

    if ticker in roll_fn_registry:
        return roll_fn_registry[ticker]
    else:
        warnings.warn(f"{ticker} not found in roll function registry, using identity.")
        return identity

