"""Input loading and schema checks."""
import pandas as pd

REQUIRED = {
    "date", "product_id", "product_name", "category", "supplier_id", "demand_units",
    "day_of_week", "month", "weekend_indicator", "festival_name", "festival_indicator",
    "promotion_indicator", "discount_percentage", "synthetic_temperature_c", "unit_cost_inr",
    "selling_price_inr", "holding_cost_per_unit_per_day_inr", "ordering_cost_per_order_inr",
    "initial_inventory_units", "baseline_daily_demand_units", "supplier_average_lead_time_days",
    "supplier_lead_time_std_days", "supplier_reliability", "supplier_max_capacity_units",
    "generated_mean_demand", "demand_pattern",
}
NUMERIC = {
    "demand_units", "month", "weekend_indicator", "festival_indicator", "promotion_indicator",
    "discount_percentage", "synthetic_temperature_c", "unit_cost_inr", "selling_price_inr",
    "holding_cost_per_unit_per_day_inr", "ordering_cost_per_order_inr", "initial_inventory_units",
    "baseline_daily_demand_units", "supplier_average_lead_time_days", "supplier_lead_time_std_days",
    "supplier_reliability", "supplier_max_capacity_units", "generated_mean_demand",
}

def load_dataset(path):
    df = pd.read_csv(path)
    missing_columns = sorted(REQUIRED - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    source = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    type_errors = {"date": int((source["date"].notna() & df["date"].isna()).sum())}
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        type_errors[col] = int((source[col].notna() & df[col].isna()).sum())
    df.attrs["type_coercion_errors"] = {k: v for k, v in type_errors.items() if v}
    return df
