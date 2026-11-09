"""Reproducible conditional scenarios with scenario-specific recursive lag updates."""
import numpy as np
import pandas as pd
from .probability import sample_demand

def _future_rows(products, date):
    rows = products.copy()
    rows["date"] = date
    dt = pd.Timestamp(date)
    rows["day_of_week"] = dt.dayofweek; rows["month"] = dt.month
    rows["weekend_indicator"] = int(dt.dayofweek >= 5)
    rows["season"] = "winter" if dt.month in (12,1,2) else "summer" if dt.month in (3,4,5) else "monsoon" if dt.month in (6,7,8) else "autumn"
    rows["festival_indicator"] = 0; rows["promotion_indicator"] = 0
    rows["discount_fraction"] = 0.0; rows["promotion_discount_interaction"] = 0.0
    rows["festival_name"] = "None"; rows["discount_percentage"] = 0.0
    return rows

def generate_scenarios(model, history, horizon=30, n_scenarios=100, seed=42):
    if horizon < 1 or n_scenarios < 1: raise ValueError("horizon and n_scenarios must be positive")
    history = history.sort_values(["product_id", "date"])
    product_rows = history.groupby("product_id", as_index=False).tail(1).drop_duplicates("product_id").copy()
    product_ids = product_rows.product_id.astype(str).tolist()
    histories = {p: list(history.loc[history.product_id.astype(str) == p, "demand_units"].tail(28).astype(int)) for p in product_ids}
    rng = np.random.default_rng(seed); records = []
    first_day = pd.Timestamp(history.date.max()) + pd.Timedelta(days=1)
    for s in range(n_scenarios):
        demand_history = {p: vals.copy() for p, vals in histories.items()}
        for day in range(horizon):
            date = first_day + pd.Timedelta(days=day)
            rows = _future_rows(product_rows.copy(), date)
            means7, std7, means28, std28 = [], [], [], []
            for p in product_ids:
                hist = demand_history[p]
                means7.append(float(np.mean(hist[-7:])) if hist else 0.0)
                std7.append(float(np.std(hist[-7:], ddof=1)) if len(hist) > 1 else 0.0)
                means28.append(float(np.mean(hist[-28:])) if hist else 0.0)
                std28.append(float(np.std(hist[-28:], ddof=1)) if len(hist) > 1 else 0.0)
            rows["demand_rolling_mean_7"] = means7; rows["demand_rolling_std_7"] = std7
            rows["demand_rolling_mean_28"] = means28; rows["demand_rolling_std_28"] = std28
            for lag in (1, 7, 14, 28):
                rows[f"demand_lag_{lag}"] = [hist[-lag] if len(hist) >= lag else (hist[0] if hist else 0) for hist in (demand_history[p] for p in product_ids)]
            means = model.predict_mean(rows)
            draws = sample_demand(rng, means, model.alpha_)
            for p, y in zip(product_ids, draws):
                records.append((s, date, p, int(y), 1.0/n_scenarios))
                demand_history[p].append(int(y))
                demand_history[p] = demand_history[p][-28:]
    return pd.DataFrame(records, columns=["scenario_id", "date", "product_id", "demand_units", "probability"])
