"""Optimize inventory decisions over the existing conditional scenario paths."""
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .config import PROCESSED_DIR, OUTPUT_DIR
from .optimization import InventoryInputs, solve_inventory, evaluate_fixed_policy
from .probability import sample_demand

OPT_DIR = OUTPUT_DIR / "optimization"

def _static_products(data):
    rows = data.sort_values("date").drop_duplicates("product_id", keep="first").sort_values("product_id").reset_index(drop=True)
    products = rows.product_id.astype(str).tolist()
    supplier_ids = rows.supplier_id.astype(str).tolist()
    product_caps = rows.supplier_max_capacity_units.to_numpy(float)
    return rows, products, supplier_ids, product_caps

def _contextual_tensor(path, products):
    frame = pd.read_csv(path, parse_dates=["date"])
    dates = sorted(frame.date.unique())
    scenario_ids = sorted(frame.scenario_id.unique())
    expected = len(scenario_ids) * len(dates) * len(products)
    if len(frame) != expected: raise ValueError(f"Scenario table is not rectangular: got {len(frame)} rows, expected {expected}")
    mass = frame.groupby(["date", "product_id"]).probability.sum()
    if not np.allclose(mass.to_numpy(), 1.0): raise ValueError("Context scenario probabilities fail product-date normalization")
    demands = np.empty((len(scenario_ids), len(dates), len(products)), dtype=float)
    probabilities = []
    for si, sid in enumerate(scenario_ids):
        group = frame[frame.scenario_id == sid]
        pivot = group.pivot(index="date", columns="product_id", values="demand_units").reindex(index=dates, columns=products)
        if pivot.isna().any().any(): raise ValueError(f"Missing demand values for scenario {sid}")
        demands[si] = pivot.to_numpy()
        probs = group.probability.unique()
        if len(probs) != 1: raise ValueError("Each scenario must have one path probability")
        probabilities.append(float(probs[0]))
    probabilities = np.asarray(probabilities)
    if not np.isclose(probabilities.sum(), 1): raise ValueError("Scenario path probabilities must sum to one")
    return demands, probabilities, pd.to_datetime(dates)

def _unconditional_tensor(baseline_path, products, shape, seed):
    with open(baseline_path, "rb") as f: baseline = pickle.load(f)
    rng = np.random.default_rng(seed); S,H = shape; out = np.empty((S,H,len(products)), dtype=float)
    for i, product in enumerate(products):
        mean = baseline.means_.get(product, baseline.global_mean_)
        out[:,:,i] = rng.poisson(mean, size=(S,H))
    return out, np.full(S, 1.0/S)

def _inputs(products, rows, supplier_ids, product_capacity, demands, probs, budget, shared_capacity=None):
    return InventoryInputs(products=products, demands=demands, probabilities=probs,
        initial_inventory=rows.initial_inventory_units.to_numpy(float),
        unit_cost=rows.unit_cost_inr.to_numpy(float), fixed_order_cost=rows.ordering_cost_per_order_inr.to_numpy(float),
        holding_cost=rows.holding_cost_per_unit_per_day_inr.to_numpy(float),
        shortage_penalty=rows.selling_price_inr.to_numpy(float),
        lead_time_days=rows.supplier_average_lead_time_days.to_numpy(float),
        supplier_ids=supplier_ids, product_capacity=product_capacity,
        supplier_capacity=shared_capacity or {}, budget=budget)

def _write_solution(label, data, result, dates, rows):
    probabilities = data.probabilities
    inv, lost = result["ending_inventory"], result["lost_sales"]
    expected_inv = np.einsum("s,sti->ti", probabilities, inv)
    expected_lost = np.einsum("s,sti->ti", probabilities, lost)
    products = data.products
    decisions = pd.DataFrame({"model": label, "product_id": products, "supplier_id": data.supplier_ids,
        "order_quantity_units": result["order_quantity"], "order_placed": result["order_placed"],
        "initial_inventory_units": data.initial_inventory, "lead_time_days": data.lead_time_days,
        "product_order_capacity_units": data.product_capacity,
        "arrival_period": result["arrival_period"], "arrival_date": [dates[min(a-1, len(dates)-1)].date() if a <= len(dates) else None for a in result["arrival_period"]],
        "expected_ending_inventory_units": expected_inv[-1], "expected_shortage_units": expected_lost.sum(axis=0),
        "unit_cost_inr": data.unit_cost, "fixed_order_cost_inr": data.fixed_order_cost,
        "holding_cost_per_unit_day_inr": data.holding_cost, "shortage_penalty_inr": data.shortage_penalty})
    decisions["purchase_cost_inr"] = decisions.order_quantity_units * decisions.unit_cost_inr
    decisions.to_csv(OPT_DIR / f"{label}_product_decisions.csv", index=False)
    states = pd.DataFrame([{"model": label, "period": t+1, "date": dates[t].date(), "product_id": product,
        "expected_ending_inventory_units": expected_inv[t,i], "expected_lost_sales_units": expected_lost[t,i]}
        for t in range(len(dates)) for i,product in enumerate(products)])
    states.to_csv(OPT_DIR / f"{label}_expected_daily_states.csv", index=False)
    return decisions, states

def run(budget=None, seed=43, solver="SCIPY", time_limit=None, shared_capacity=None):
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    feat_path = PROCESSED_DIR / "contextual_inventory_features.csv"
    scenario_path = OUTPUT_DIR / "evaluations" / "demand_scenarios.csv"
    baseline_path = OUTPUT_DIR / "models" / "product_mean_baseline.pkl"
    for path in (feat_path, scenario_path, baseline_path):
        if not path.exists(): raise FileNotFoundError(f"Required Module 1/2 output missing: {path}")
    data = pd.read_csv(feat_path, parse_dates=["date"])
    products_frame, products, suppliers, product_capacity = _static_products(data)
    contextual, probs, dates = _contextual_tensor(scenario_path, products)
    unconditional, uncond_probs = _unconditional_tensor(baseline_path, products, contextual.shape[:2], seed)
    instances = {"context_aware": _inputs(products, products_frame, suppliers, product_capacity, contextual, probs, budget, shared_capacity),
                 "non_contextual": _inputs(products, products_frame, suppliers, product_capacity, unconditional, uncond_probs, budget, shared_capacity)}
    results = {}; summaries=[]
    for label, instance in instances.items():
        result = solve_inventory(instance, solver=solver, time_limit=time_limit)
        results[label] = result
        _write_solution(label, instance, result, dates, products_frame)
        summaries.append({"model": label, "solver_status": result["status"], "solver": result["solver"], "computational_time_seconds": result["solve_seconds"],
            "scenario_count": len(instance.probabilities), "horizon_days": instance.demands.shape[1], "product_count": len(products),
            "shared_budget_inr": budget, "budget_applied": budget is not None, **result["costs"]})
    pd.DataFrame(summaries).to_csv(OPT_DIR / "optimization_summary.csv", index=False)
    cross=[]
    for policy_name, result in results.items():
        for scenario_name, instance in instances.items():
            score = evaluate_fixed_policy(instance, result["order_quantity"])
            cross.append({"policy_from": policy_name, "evaluated_on": scenario_name, **score})
    pd.DataFrame(cross).to_csv(OPT_DIR / "policy_cross_evaluation.csv", index=False)
    return {"summary": summaries, "cross_evaluation": cross}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-inr", type=float, default=None, help="Optional shared purchasing budget (unit purchase cost only)")
    parser.add_argument("--seed", type=int, default=43, help="Seed for non-contextual Poisson scenarios")
    parser.add_argument("--solver", default="SCIPY", help="CVXPY solver; SCIPY/HiGHS supports the mixed-integer order decisions")
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--shared-supplier-capacity", action="append", default=[], metavar="SUPPLIER=UNITS", help="Optional aggregate limit; repeat per supplier")
    args=parser.parse_args()
    shared_capacity={}
    for item in args.shared_supplier_capacity:
        supplier, sep, value = item.partition("=")
        if not sep: parser.error("shared supplier capacity must use SUPPLIER=UNITS")
        shared_capacity[supplier] = float(value)
    result=run(args.budget_inr,args.seed,args.solver,args.time_limit,shared_capacity)
    for row in result["summary"]: print(f"{row['model']}: {row['solver_status']}; expected cost INR {row['expected_total_cost']:.2f}; {row['computational_time_seconds']:.2f}s")

if __name__ == "__main__": main()
