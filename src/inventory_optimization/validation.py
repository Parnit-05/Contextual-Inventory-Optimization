"""Validation checks and compact machine-readable/text reports."""
import json
import pandas as pd

def validate(df):
    dup_key = int(df.duplicated(["date", "product_id"]).sum())
    checks = {
        "row_count": int(len(df)), "column_count": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()), "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v},
        "duplicate_rows": int(df.duplicated().sum()), "duplicate_date_product_keys": dup_key,
        "invalid_dates": int(df.date.isna().sum()),
        "type_coercion_errors": df.attrs.get("type_coercion_errors", {}),
        "invalid_category_values": {
            "category": int((~df.category.isin(["Dairy", "Staples", "Beverages", "Snacks", "Household"])).sum()),
            "demand_pattern": int((~df.demand_pattern.isin(["stable", "seasonal", "trending", "intermittent", "volatile"])).sum()),
            "day_of_week": int((~df.day_of_week.isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])).sum()),
        },
        "invalid_values": {
            "negative_demand": int((df.demand_units < 0).sum()),
            "non_integer_demand": int((df.demand_units.dropna() % 1 != 0).sum()),
            "discount_outside_0_25": int((~df.discount_percentage.between(0, 25)).sum()),
            "reliability_outside_0_1": int((~df.supplier_reliability.between(0, 1)).sum()),
            "invalid_month": int((~df.month.between(1, 12)).sum()),
            "invalid_binary_flags": int(sum((~df[c].isin([0, 1])).sum() for c in ["weekend_indicator", "festival_indicator", "promotion_indicator"])),
            "nonpositive_supplier_capacity": int((df.supplier_max_capacity_units <= 0).sum()),
            "negative_lead_time": int((df.supplier_average_lead_time_days < 0).sum()),
        },
        "data_types": {k: str(v) for k, v in df.dtypes.items()},
        "categorical_values": {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in ["category", "demand_pattern"]},
        "status": "passed_with_warnings" if df.isna().any().any() or dup_key else "passed",
    }
    return checks

def write_report(report, json_path, txt_path):
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = ["# Data validation report", "", f"Status: {report['status']}", f"Rows: {report['row_count']}", f"Columns: {report['column_count']}", f"Missing cells: {report['missing_cells']}", f"Duplicate rows: {report['duplicate_rows']}", f"Duplicate (date, product) keys: {report['duplicate_date_product_keys']}", "", "## Invalid value counts", ""]
    lines.extend(f"- {k}: {v}" for k, v in report["invalid_values"].items())
    lines += ["", "## Type conversion errors", ""]
    if report["type_coercion_errors"]:
        lines.extend(f"- {k}: {v}" for k, v in report["type_coercion_errors"].items())
    else: lines.append("- None")
    lines += ["", "## Missing values by column", ""]
    lines.extend(f"- {k}: {v}" for k, v in report["missing_by_column"].items())
    if not report["missing_by_column"]: lines.append("- None")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
