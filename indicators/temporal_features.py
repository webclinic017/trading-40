import pandas as pd
from typing import Dict, List


def difference_features(df: pd.DataFrame, colnames: List[str]) -> pd.DataFrame:
    for col in colnames:
        df[col] = df[col].diff(periods=1)

    return df



def lag_features(df: pd.DataFrame, features: Dict[str, List[int]]) -> pd.DataFrame:
    for colname, time_periods in features.items():
        for k in time_periods:
            df[f"{colname}-{k}"] = df[colname].shift(k)

    df.dropna(inplace=True)

    return df
