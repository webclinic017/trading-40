import numpy as np
import pandas as pd


def yearly_features(df) -> pd.DataFrame:
    # Assumes df.index is a `datetime` object.
    num_days_per_year = 366  # 1-indexed
    df["day_of_year"] = df.index.day_of_year
    df["cos_day_of_year"] = np.cos(2*np.pi*df["day_of_year"] / num_days_per_year)
    df["sin_day_of_year"] = np.sin(2*np.pi*df["day_of_year"] / num_days_per_year)
    return df


def weekly_features(df) -> pd.DataFrame:
    # Assumes df.index is a `datetime` object.
    num_days_per_week = 6    # 0-indexed
    df["day_of_week"] = df.index.day_of_week
    df["cos_day_of_week"] = np.cos(2*np.pi*df["day_of_week"] / num_days_per_week)
    df["sin_day_of_week"] = np.sin(2*np.pi*df["day_of_week"] / num_days_per_week)
    return df
