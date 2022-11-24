import numpy as np


def compute_positions(df_in):
    df = df_in.copy()

    # Signals to demonstrate when to propagate positions forward:
    # - Stay long if: Z_exit_threshold < Z < Z_entry_threshold
    # - Stay short if: Z_entry_threshold < Z < Z_exit_threshold
    df["long_market"] = 0.0  # Must be float
    df["short_market"] = 0.0  # Must be float

    # Track whether to be long or short while iterating through each timestep
    long_market = 0.0  # Must be float
    short_market = 0.0  # Must be float

    # Calculate when to be in the market via holding a long or short position, and when to exit the market.
    # Hard to vectorise: note how `long_market` and `short_market` values are carried over in each loop iteration.
    long_markets = []
    short_markets = []
    for i, row in enumerate(df.iterrows()):
        if row[1]["long"] == 1.0:
            long_market = 1
        if row[1]["short"] == 1.0:
            short_market = 1
        if row[1]["exit"] == 1.0:
            long_market = 0
            short_market = 0

        # Assign 1/0 to long_market/short_market to indicate when to stay in a position
        long_markets.append(long_market)
        short_markets.append(short_market)

    df["long_market"] = long_markets
    df["short_market"] = short_markets
    df["positions"] = df["long_market"] - df["short_market"]

    # Using _pos to distinguish portfolio from raw.
    df["S1_pos"] = -1.0 * df["S1"] * df["positions"]
    df["S2_pos"] = df["S2"] * df["positions"]
    df["total"] = df["S1_pos"] + df["S2_pos"]

    return df


# TODO: S1_pos, S2_pos - made distinction to preserve data. Careful to propagate.

def compute_returns(df_in):
    """
    Naming convention: `returns := simple returns, returns_cml := cumulative returns`
    """
    df = df_in.copy()

    # Calculate simple percentage returns
    df["returns_pc"] = df["total"].pct_change()

    df["returns_pc"].fillna(0.0, inplace=True)
    df["returns_pc"].replace([np.inf, -np.inf], 0.0, inplace=True)
    df["returns_pc"].replace(-1.0, 0.0, inplace=True)

    # Accumulate returns across each time period
    df["returns_cml"] = (1.0 + df["returns_pc"]).cumprod()

    # S1 and S2 correct here: comparing PF to buy-and-hold.
    df["returns_cml_S1"] = (df["S1"].pct_change() + 1.0).cumprod()
    df["returns_cml_S2"] = (df["S2"].pct_change() + 1.0).cumprod()

    return df
