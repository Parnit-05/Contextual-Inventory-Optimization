"""Fit conditional demand distributions, evaluate chronologically, and sample scenarios."""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR
from .demand_model import ProductMeanBaseline, PooledPoisson, PooledNegativeBinomial
from .probability import evaluate_distribution, interval
from .scenarios import generate_scenarios

def _usable(frame):
    cols = ["demand_lag_1", "demand_lag_7", "demand_lag_14", "demand_lag_28"]
    return frame.dropna(subset=cols).copy()

def _scores(frame, means, alpha):
    y = frame.demand_units.to_numpy(dtype=int); means = np.maximum(np.asarray(means), 1e-8)
    prob = evaluate_distribution(y, means, alpha)
    return {"mae": float(np.mean(np.abs(y-means))), "rmse": float(np.sqrt(np.mean((y-means)**2))), **prob}

def run(horizon=30, scenarios=100, seed=42):
    for folder in (OUTPUT_DIR / "models", OUTPUT_DIR / "evaluations", FIGURES_DIR): folder.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "contextual_inventory_features.csv"
    if not path.exists(): raise FileNotFoundError("Run Module 1 first to create contextual_inventory_features.csv")
    data = pd.read_csv(path, parse_dates=["date"])
    forbidden = {"generated_mean_demand", "demand_pattern"}
    models = [ProductMeanBaseline(), PooledPoisson(), PooledNegativeBinomial()]
    train, val, test = (_usable(data[data.split == name]) for name in ("train", "validation", "test"))
    if min(len(train), len(val), len(test)) == 0: raise ValueError("Each chronological split must contain usable lagged rows")
    baseline, poisson, negative_binomial = models
    baseline.fit(train); poisson.fit(train); negative_binomial.fit(train)
    if forbidden.intersection(poisson.column_names_): raise AssertionError("Forbidden predictors reached model design")
    val_results = {
        "product_mean_poisson": _scores(val, baseline.predict_mean(val), 0.0),
        "contextual_poisson": _scores(val, poisson.predict_mean(val), 0.0),
        "contextual_negative_binomial_nb2": _scores(val, negative_binomial.predict_mean(val), negative_binomial.alpha_),
    }
    selected_name = min(["contextual_poisson", "contextual_negative_binomial_nb2"], key=lambda k: val_results[k]["negative_log_likelihood"])
    selected = poisson if selected_name == "contextual_poisson" else negative_binomial
    test_results = {
        "product_mean_poisson": _scores(test, baseline.predict_mean(test), 0.0),
        "contextual_poisson": _scores(test, poisson.predict_mean(test), 0.0),
        "contextual_negative_binomial_nb2": _scores(test, negative_binomial.predict_mean(test), negative_binomial.alpha_),
    }
    predictions = test[["date", "product_id", "category", "demand_units"]].copy()
    predictions["baseline_mean"] = baseline.predict_mean(test)
    predictions["contextual_mean"] = selected.predict_mean(test)
    predictions["selected_distribution"] = selected_name
    predictions["dispersion_alpha"] = selected.alpha_
    intervals = [interval(mu, selected.alpha_) for mu in predictions["contextual_mean"]]
    predictions["prediction_interval_90_lower"] = [bounds[0] for bounds in intervals]
    predictions["prediction_interval_90_upper"] = [bounds[1] for bounds in intervals]
    predictions.to_csv(OUTPUT_DIR / "evaluations" / "test_predictions.csv", index=False)
    scenarios_df = generate_scenarios(selected, data, horizon=horizon, n_scenarios=scenarios, seed=seed)
    scenarios_df.to_csv(OUTPUT_DIR / "evaluations" / "demand_scenarios.csv", index=False)
    report = {
        "target": "demand_units", "split_method": "existing chronological Module 1 split; leading rows without complete requested lags dropped",
        "split_rows_used": {"train": len(train), "validation": len(val), "test": len(test)},
        "feature_inputs": list(poisson.column_names_), "excluded_predictors": ["demand_units", "generated_mean_demand", "demand_pattern"],
        "validation_metrics": val_results, "test_metrics": test_results, "selected_distribution_by_validation_nll": selected_name,
        "poisson_dispersion_alpha": 0.0, "negative_binomial_nb2_alpha": negative_binomial.alpha_,
        "scenario_config": {"horizon_days": horizon, "scenario_count": scenarios, "seed": seed, "equal_probability": 1.0/scenarios, "products": int(scenarios_df.product_id.nunique()), "rows": len(scenarios_df)},
    }
    (OUTPUT_DIR / "evaluations" / "model_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([{ "model": name, "split": split, **metrics} for split, rows in (("validation", val_results), ("test", test_results)) for name, metrics in rows.items()]).to_csv(OUTPUT_DIR / "evaluations" / "model_evaluation.csv", index=False)
    selected.save(OUTPUT_DIR / "models" / "selected_contextual_model.pkl")
    baseline.save(OUTPUT_DIR / "models" / "product_mean_baseline.pkl")
    poisson.save(OUTPUT_DIR / "models" / "contextual_poisson.pkl")
    negative_binomial.save(OUTPUT_DIR / "models" / "contextual_negative_binomial.pkl")
    fig, ax = plt.subplots(figsize=(8, 5))
    metric_df = pd.DataFrame([{ "model": name, "nll": m["negative_log_likelihood"]} for name, m in val_results.items()])
    ax.bar(metric_df.model, metric_df.nll); ax.set_ylabel("Validation negative log-likelihood (lower is better)"); ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(FIGURES_DIR / "module2_validation_nll.png", dpi=150); plt.close(fig)
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=30); parser.add_argument("--scenarios", type=int, default=100); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); report = run(args.horizon, args.scenarios, args.seed)
    print(f"Module 2 complete: selected {report['selected_distribution_by_validation_nll']}; scenarios={report['scenario_config']['rows']:,} rows")

if __name__ == "__main__": main()
