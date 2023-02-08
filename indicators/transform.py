import pandas as pd
import statsmodels.api as sm
from typing import Dict, List, Tuple
from indicators.periodic_features import weekly_features, yearly_features
from indicators.temporal_features import difference_features, lag_features


def build_features(
        df: pd.DataFrame,
        features_cols: List[int],
        add_lag_features: Dict[str, List[int]],
        add_yearly_features: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Target is next day's price.
    df["price_target"] = df["price"].shift(-1)
    df.dropna(inplace=True)

    if add_lag_features:
        df = df.pipe(lag_features, features=add_lag_features)
        for colname, time_periods in add_lag_features.items():
            features_cols.extend([f"{colname}-{k}" for k in time_periods])

    if add_yearly_features:
        df = df.pipe(yearly_features)
        features_cols.extend(["cos_day_of_year", "sin_day_of_year"])

    features = df[features_cols]
    features = sm.add_constant(features)
    target = df["price_target"]

    return features, target
