"""Run Module 1 using the repository's existing raw dataset."""
import argparse
import json
import pandas as pd
from .config import RAW_DATA, DATA_DICTIONARY, PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR
from .data import load_dataset
from .validation import validate, write_report
from .features import add_contextual_features, chronological_splits
from .eda import run_eda

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(RAW_DATA)); args = parser.parse_args()
    for folder in (PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR): folder.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(args.input)
    pd.read_csv(DATA_DICTIONARY).to_csv(REPORTS_DIR / "data_dictionary_copy.csv", index=False)
    report = validate(raw)
    write_report(report, REPORTS_DIR / "data_validation_report.json", REPORTS_DIR / "data_validation_report.md")
    if report["invalid_dates"] or report["invalid_values"]["negative_demand"] or report["duplicate_date_product_keys"]:
        raise ValueError("Input has critical data issues; see outputs/reports/data_validation_report.md")
    # Source calendar fields are checked by report, then recomputed from canonical dates.
    featured = add_contextual_features(raw)
    featured, train_end, val_end = chronological_splits(featured)
    featured = featured.sort_values(["date", "product_id"])
    featured.to_csv(PROCESSED_DIR / "contextual_inventory_features.csv", index=False)
    run_eda(raw, PROCESSED_DIR, FIGURES_DIR)
    summary = {"rows": len(raw), "products": int(raw.product_id.nunique()), "categories": int(raw.category.nunique()), "suppliers": int(raw.supplier_id.nunique()), "date_start": str(raw.date.min().date()), "date_end": str(raw.date.max().date()), "demand_mean": float(raw.demand_units.mean()), "demand_median": float(raw.demand_units.median()), "demand_std": float(raw.demand_units.std()), "demand_total": int(raw.demand_units.sum()), "train_through": str(train_end.date()), "validation_through": str(val_end.date()), "split_rows": {str(k): int(v) for k, v in featured.split.value_counts().items()}, "target": "demand_units", "excluded_predictor_columns": ["demand_units", "generated_mean_demand", "demand_pattern"], "feature_contract": featured.attrs.get("feature_columns", [])}
    (REPORTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORTS_DIR / "dataset_summary.md").write_text("# Dataset summary\n\n" + "\n".join(f"- **{k.replace('_', ' ').title()}:** {v}" for k, v in summary.items() if k != "feature_contract") + "\n", encoding="utf-8")
    print(f"Module 1 complete: {len(raw):,} rows; outputs in {PROCESSED_DIR.parent}")

if __name__ == "__main__": main()
