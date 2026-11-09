"""Leakage-aware features computed using each product's prior observations only."""
import numpy as np
import pandas as pd
from .config import LAGS, ROLLING_WINDOWS

EXCLUDED = {"demand_units", "generated_mean_demand", "demand_pattern"}

def chronological_splits(df):
    dates = np.array(sorted(df["date"].dropna().unique()))
    if len(dates) < 3:
        raise ValueError("At least three distinct dates are required for chronological splits")
    train_end = dates[max(0, int(len(dates) * .70) - 1)]
    val_end = dates[max(0, int(len(dates) * .85) - 1)]
    out = df.copy()
    out["split"] = np.select([out.date <= train_end, out.date <= val_end], ["train", "validation"], default="test")
    return out, pd.Timestamp(train_end), pd.Timestamp(val_end)

def add_contextual_features(df):
    out = df.sort_values(["product_id", "date"]).copy()
    grp = out.groupby("product_id", sort=False)["demand_units"]
    for lag in LAGS:
        out[f"demand_lag_{lag}"] = grp.shift(lag)
    prior = grp.shift(1)
    for window in ROLLING_WINDOWS:
        roll = prior.groupby(out["product_id"], sort=False).rolling(window, min_periods=1)
        out[f"demand_rolling_mean_{window}"] = roll.mean().reset_index(level=0, drop=True).sort_index()
        out[f"demand_rolling_std_{window}"] = roll.std().reset_index(level=0, drop=True).sort_index().fillna(0)
    dt = out["date"]
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["weekend_indicator"] = (dt.dt.dayofweek >= 5).astype("int8")
    out["season"] = np.select([dt.dt.month.isin([12, 1, 2]), dt.dt.month.isin([3, 4, 5]), dt.dt.month.isin([6, 7, 8])], ["winter", "summer", "monsoon"], default="autumn")
    out["festival_indicator"] = out["festival_indicator"].fillna(0).astype("int8")
    out["promotion_indicator"] = out["promotion_indicator"].fillna(0).astype("int8")
    out["discount_fraction"] = out["discount_percentage"] / 100.0
    out["promotion_discount_interaction"] = out["promotion_indicator"] * out["discount_fraction"]
    # Explicit predictor contract: target and generator metadata can never enter the feature set.
    feature_columns = [c for c in out.columns if c not in EXCLUDED and c not in {"split"}]
    out.attrs["feature_columns"] = feature_columns
    return out
