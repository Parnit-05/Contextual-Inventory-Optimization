"""Graceful, read-only access to generated project artifacts for the dashboard."""
from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT=Path(__file__).resolve().parents[1]

def artifact_path(relative_path, root=PROJECT_ROOT):
    return Path(root)/relative_path

def load_csv(relative_path, root=PROJECT_ROOT):
    path=artifact_path(relative_path,root)
    if not path.is_file(): return None
    try: return pd.read_csv(path)
    except (OSError,UnicodeError,pd.errors.ParserError,ValueError): return None

def load_markdown(relative_path, root=PROJECT_ROOT):
    path=artifact_path(relative_path,root)
    if not path.is_file(): return None
    try: return path.read_text(encoding="utf-8")
    except (OSError,UnicodeError): return None

def load_json(relative_path, root=PROJECT_ROOT):
    path=artifact_path(relative_path,root)
    if not path.is_file(): return None
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError): return None

def load_dashboard_data(root=PROJECT_ROOT):
    csvs={
        "category_demand":"data/processed/category_demand_summary.csv",
        "product_demand":"data/processed/product_demand_summary.csv",
        "optimization":"outputs/optimization/optimization_summary.csv",
        "contextual_orders":"outputs/optimization/context_aware_product_decisions.csv",
        "noncontextual_orders":"outputs/optimization/non_contextual_product_decisions.csv",
        "historical_simulation":"outputs/simulation/policy_comparison_summary.csv",
        "repeated_summary":"outputs/robustness/repeated_policy_summary.csv",
        "repeated_cost_breakdown":"outputs/robustness/repeated_cost_breakdown.csv",
        "paired_comparison":"outputs/robustness/paired_comparison.csv",
        "sensitivity":"outputs/robustness/sensitivity_summary.csv",
        "replications":"outputs/robustness/replication_metrics.csv",
    }
    data={key:load_csv(path,root) for key,path in csvs.items()}
    data["dataset_summary"]=load_json("outputs/reports/dataset_summary.json",root)
    data["research_report"]=load_markdown("outputs/reports/final_research_report.md",root)
    data["figures"]={name:artifact_path(f"outputs/figures/{name}.png",root) for name in
        ("daily_demand_trend","demand_by_category","demand_distribution","monthly_demand","seasonal_demand","weekly_demand","festival_demand","supplier_profile")}
    data["report_figures"]={name:artifact_path(f"outputs/reports/figures/{name}.png",root) for name in
        ("repeated_policy_cost","repeated_policy_fill_rate","sensitivity_cost")}
    return data
