"""Held-out rolling-window simulation of optimized and conventional inventory policies."""
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PROCESSED_DIR, OUTPUT_DIR
from .module3 import _static_products
from .optimization import InventoryInputs, solve_inventory
from .scenarios import generate_scenarios
from .simulation import sample_lead_times, simulate_daily

SIM_DIR = OUTPUT_DIR / "simulation"
FIG_DIR = SIM_DIR / "figures"
WINDOW_DAYS = 30

def _scenario_tensor(frame, products):
    dates = sorted(pd.to_datetime(frame.date.unique()))
    scenario_ids = sorted(frame.scenario_id.unique())
    D = np.empty((len(scenario_ids), len(dates), len(products)), dtype=float)
    p = []
    for s, sid in enumerate(scenario_ids):
        g = frame[frame.scenario_id == sid]
        pivot = g.pivot(index="date", columns="product_id", values="demand_units").reindex(index=dates, columns=products)
        if pivot.isna().any().any(): raise ValueError(f"Incomplete conditional demand path for scenario {sid}")
        D[s] = pivot.to_numpy()
        weights = g.probability.unique()
        if len(weights) != 1: raise ValueError("Inconsistent scenario path probability")
        p.append(float(weights[0]))
    p = np.asarray(p)
    if not np.isclose(p.sum(), 1): raise ValueError("Forecast scenario probabilities do not sum to one")
    return D, p

def _optimizer_inputs(products, rows, stock, D, p, scheduled_receipts=None):
    return InventoryInputs(products=products, demands=D, probabilities=p, initial_inventory=np.asarray(stock, dtype=float),
        unit_cost=rows.unit_cost_inr.to_numpy(float), fixed_order_cost=rows.ordering_cost_per_order_inr.to_numpy(float),
        holding_cost=rows.holding_cost_per_unit_per_day_inr.to_numpy(float), shortage_penalty=rows.selling_price_inr.to_numpy(float),
        lead_time_days=rows.supplier_average_lead_time_days.to_numpy(float), supplier_ids=rows.supplier_id.astype(str).tolist(),
        product_capacity=rows.supplier_max_capacity_units.to_numpy(float), supplier_capacity={}, budget=None,
        scheduled_receipts=scheduled_receipts)

def _daily_test_matrix(test, dates, products):
    if test.duplicated(["date", "product_id"]).any(): raise ValueError("Duplicate test date/product rows")
    pivot = test.pivot(index="date", columns="product_id", values="demand_units").reindex(index=dates, columns=products)
    if pivot.isna().any().any(): raise ValueError("Test demand is not complete for every date and product")
    return pivot.to_numpy(dtype=float)

def _decision_factory(policy, full_data, product_rows, products, demand_model, mean_baseline, n_scenarios, seed, optimization_seconds):
    caps = product_rows.supplier_max_capacity_units.to_numpy(float)
    def decide(window, day_index, stock, current_date, horizon, scheduled_receipts=None):
        history = full_data[full_data.date < current_date].sort_values(["product_id", "date"])
        if policy == "historical_mean_order_up_to_30d":
            quantities=[]
            for i, product in enumerate(products):
                prior = history.loc[history.product_id.astype(str) == product, "demand_units"].tail(28)
                daily_mean = float(prior.mean()) if len(prior) else 0.0
                target = daily_mean * (horizon + float(product_rows.iloc[i].supplier_average_lead_time_days))
                quantities.append(min(int(caps[i]), max(0, int(np.ceil(target-stock[i])))))
            return np.asarray(quantities, dtype=int)
        if policy == "context_aware_optimization":
            forecast = generate_scenarios(demand_model, history, horizon=horizon, n_scenarios=n_scenarios, seed=seed+window)
            D,p = _scenario_tensor(forecast, products)
        elif policy == "non_contextual_optimization":
            rng = np.random.default_rng(seed+10_000+window)
            D = np.empty((n_scenarios, horizon, len(products)), dtype=float)
            for i, product in enumerate(products):
                mu=mean_baseline.means_.get(product, mean_baseline.global_mean_)
                D[:,:,i]=rng.poisson(mu, size=(n_scenarios,horizon))
            p=np.full(n_scenarios,1/n_scenarios)
        else:
            raise ValueError(f"Unknown policy {policy}")
        instance = _optimizer_inputs(products, product_rows, stock, D, p, scheduled_receipts)
        result = solve_inventory(instance)
        optimization_seconds[0] += result["solve_seconds"]
        return result["order_quantity"]
    return decide

def _product_metrics(daily, orders, products):
    rows=[]
    for product in products:
        d=daily[daily.product_id.astype(str)==product]
        o=orders[orders.product_id.astype(str)==product]
        demand=float(d.demand_units.sum()); fulfilled=float(d.fulfilled_units.sum())
        rows.append({"product_id":product,"category":d.category.iloc[0],"supplier_id":d.supplier_id.iloc[0],
            "total_inventory_cost_inr":float(d[["purchase_cost_inr","fixed_order_cost_inr","holding_cost_inr","shortage_cost_inr"]].sum().sum()),
            "purchase_cost_inr":float(d.purchase_cost_inr.sum()),"ordering_cost_inr":float(d.fixed_order_cost_inr.sum()),
            "holding_cost_inr":float(d.holding_cost_inr.sum()),"shortage_cost_inr":float(d.shortage_cost_inr.sum()),
            "total_units_demanded":int(demand),"total_units_fulfilled":int(fulfilled),"total_lost_sales_units":int(d.lost_sales_units.sum()),
            "fill_rate":fulfilled/demand if demand else 1.0,"stockout_days":int((d.lost_sales_units>0).sum()),
            "average_on_hand_inventory_units":float(d.ending_inventory_units.mean()),"average_inventory_value_inr":float(d.groupby("date").inventory_value_inr.sum().mean()),
            "average_order_quantity_units":float(o.order_quantity_units.mean()) if len(o) else 0.0,"number_of_orders":int(len(o))})
    return pd.DataFrame(rows)

def _category_metrics(product_results):
    rows=[]
    for category,g in product_results.groupby("category"):
        demand=g.total_units_demanded.sum(); fulfilled=g.total_units_fulfilled.sum()
        cost_fields=["total_inventory_cost_inr","purchase_cost_inr","ordering_cost_inr","holding_cost_inr","shortage_cost_inr"]
        row={"category":category,"product_count":g.product_id.nunique(),"total_units_demanded":int(demand),"total_units_fulfilled":int(fulfilled),
            "total_lost_sales_units":int(g.total_lost_sales_units.sum()),"fill_rate":fulfilled/demand if demand else 1.0,
            "stockout_product_days":int(g.stockout_days.sum()),"average_on_hand_inventory_units":float(g.average_on_hand_inventory_units.mean())}
        row.update({c:float(g[c].sum()) for c in cost_fields})
        rows.append(row)
    return pd.DataFrame(rows)

def _plot_results(summary, daily):
    FIG_DIR.mkdir(parents=True,exist_ok=True)
    for metric, title, filename in [("total_inventory_cost_inr","Total inventory cost (INR)","policy_cost.png"),
                                    ("fill_rate","Fill rate","policy_fill_rate.png"),
                                    ("stockout_frequency","Stockout product-day frequency","policy_stockout_frequency.png"),
                                    ("average_on_hand_inventory_units","Average on-hand inventory (units)","policy_average_inventory.png")]:
        fig,ax=plt.subplots(figsize=(9,5)); ax.bar(summary.policy,summary[metric]); ax.set_title(title); ax.tick_params(axis="x",rotation=15); fig.tight_layout(); fig.savefig(FIG_DIR/filename,dpi=150); plt.close(fig)
    chart=daily.groupby(["date","policy"],as_index=False).ending_inventory_units.sum()
    fig,ax=plt.subplots(figsize=(11,5))
    for policy,g in chart.groupby("policy"): ax.plot(g.date,g.ending_inventory_units,label=policy)
    ax.set_title("Daily ending inventory across products"); ax.set_ylabel("Units"); ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR/"daily_inventory_by_policy.png",dpi=150); plt.close(fig)

def run(seed=704, scenario_count=25, lead_seed=404, window_days=WINDOW_DAYS):
    SIM_DIR.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True)
    feature_path=PROCESSED_DIR/"contextual_inventory_features.csv"
    model_path=OUTPUT_DIR/"models"/"selected_contextual_model.pkl"
    baseline_path=OUTPUT_DIR/"models"/"product_mean_baseline.pkl"
    if not all(p.exists() for p in (feature_path,model_path,baseline_path)):
        raise FileNotFoundError("Module 1 features and Module 2 model artifacts are required; run Modules 1 and 2 first")
    data=pd.read_csv(feature_path,parse_dates=["date"])
    test=data[data.split=="test"].copy()
    if test.empty: raise ValueError("Module 1 has no held-out test rows")
    product_rows,products,_,_=_static_products(data)
    dates=pd.DatetimeIndex(sorted(test.date.unique()))
    actual=_daily_test_matrix(test,dates,products)
    windows=list(range(0,len(dates),window_days))
    lead_base,failed=sample_lead_times(product_rows,len(windows),seed=lead_seed,std_multiplier=1.0)
    lead_high,_=sample_lead_times(product_rows,len(windows),seed=lead_seed,std_multiplier=1.5)
    with open(model_path,"rb") as f: demand_model=pickle.load(f)
    with open(baseline_path,"rb") as f: mean_baseline=pickle.load(f)
    policy_names=["context_aware_optimization","non_contextual_optimization","historical_mean_order_up_to_30d"]
    daily_all=[]; orders_all=[]; summaries=[]; schedules={}; optimization_times={}
    for policy in policy_names:
        optimization_seconds=[0.0]
        decision=_decision_factory(policy,data,product_rows,products,demand_model,mean_baseline,scenario_count,seed+100*len(summaries),optimization_seconds)
        # A zero-filled initial schedule is replaced online at each window using information available then.
        schedule={w:np.zeros(len(products),dtype=int) for w in range(len(windows))}
        daily,orders,metrics,q_by_window=simulate_daily(dates,actual,product_rows,schedule,lead_base,windows,
            decision_callback=decision,policy=policy)
        schedules[policy]=q_by_window; optimization_times[policy]=optimization_seconds[0]
        metrics["computational_time_seconds"]=optimization_seconds[0]
        summaries.append({"policy":policy,"evaluation_start":str(dates.min().date()),"evaluation_end":str(dates.max().date()),**metrics})
        daily_all.append(daily); orders_all.append(orders)
    daily_all=pd.concat(daily_all,ignore_index=True); orders_all=pd.concat(orders_all,ignore_index=True)
    # Demand and delivery scenarios are shared across all policies by construction.
    for policy,g in daily_all.groupby("policy"):
        check=g.sort_values(["date","product_id"]).demand_units.to_numpy()
        ref=daily_all[daily_all.policy==policy_names[0]].sort_values(["date","product_id"]).demand_units.to_numpy()
        if not np.array_equal(check,ref): raise AssertionError("Policies did not use identical realized demand")
    # Cost/lead sensitivity replays freeze each policy's actual order decisions.
    sensitivity=[]
    for policy in policy_names:
        qplan=schedules[policy]
        for name,leads,hm,sm in [("base",lead_base,1.0,1.0),("lead_time_std_1.5x",lead_high,1.0,1.0),
                                 ("holding_cost_1.5x",lead_base,1.5,1.0),("shortage_penalty_1.5x",lead_base,1.0,1.5)]:
            d,o,m,_=simulate_daily(dates,actual,product_rows,qplan,leads,windows,holding_multiplier=hm,shortage_multiplier=sm,policy=policy)
            sensitivity.append({"policy":policy,"sensitivity_case":name,"order_decisions":"frozen at base plan",**m})
    # Primary daily outputs hold all operational costs and actual outcomes.
    daily_all.to_csv(SIM_DIR/"daily_inventory_simulation.csv",index=False)
    orders_all.to_csv(SIM_DIR/"orders_and_deliveries.csv",index=False)
    summary=pd.DataFrame(summaries)
    summary.to_csv(SIM_DIR/"policy_comparison_summary.csv",index=False)
    summary[["policy","purchase_cost_inr","ordering_cost_inr","holding_cost_inr","shortage_cost_inr","total_inventory_cost_inr"]].to_csv(SIM_DIR/"cost_breakdown.csv",index=False)
    prod_frames=[]; cat_frames=[]
    for policy,g in daily_all.groupby("policy"):
        pmetrics=_product_metrics(g,orders_all[orders_all.policy==policy],products); pmetrics.insert(0,"policy",policy); prod_frames.append(pmetrics)
        cmetrics=_category_metrics(pmetrics); cmetrics.insert(0,"policy",policy); cat_frames.append(cmetrics)
    pd.concat(prod_frames,ignore_index=True).to_csv(SIM_DIR/"product_performance.csv",index=False)
    pd.concat(cat_frames,ignore_index=True).to_csv(SIM_DIR/"category_performance.csv",index=False)
    pd.DataFrame(sensitivity).to_csv(SIM_DIR/"sensitivity_analysis.csv",index=False)
    _plot_results(summary,daily_all)
    methodology=(f"# Module 4 evaluation methodology\n\n"
        f"Backtest uses the Module 1 chronological held-out test dates {dates.min().date()} through {dates.max().date()} ({len(dates)} days, {len(products)} products). All policies use the same realized demand for each product/date, source cost parameters, initial inventory, and common random-number supplier lead-time realizations (seed {lead_seed}). Test outcomes are used only for simulation and reporting, never for orders at that date.\n\n"
        f"Policies make one order decision at the start of each rolling {window_days}-day review window. This preserves Module 3's one-order-per-horizon distinction; it is not a daily reorder policy. Context and non-context policies are re-optimized at each window start from forecasts generated using only demand history through the prior day. Module 2 model artifacts are held fixed (trained on the earlier training split). The conventional baseline orders up to trailing-28-day mean demand times review horizon plus average lead time, capped at the dataset per-product capacity.\n\n"
        f"Supplier lead time is sampled as ceil(max(0, average + standard deviation * Normal(0,1))). A common Bernoulli reliability outcome is applied per product/window; a failed first attempt adds one ceil(average lead) retry cycle, after which delivery is assumed successful. This explicit simplification is simulation only; Module 3 still optimizes deterministic average lead. Sensitivity cases freeze base order decisions while increasing lead-time standard deviation by 1.5x or holding/shortage cost coefficients by 1.5x.\n\n"
        f"Module 2 scenario IDs are coherent recursive paths over the forecast horizon within each product because sampled demand updates that scenario's lag history. Product draws are conditionally independent; no cross-product shock or dependence is represented. Those forecasts are used only to make orders; the primary backtest scores realized held-out demand, not generated scenarios.\n\n"
        f"Fill rate is total fulfilled units divided by total realized demand. Stockout product-days count product-date pairs with positive lost sales; stockout frequency divides by product count times evaluation dates. Average on-hand inventory is the mean ending stock over all product-days; average inventory value is mean daily total ending stock times unit cost. Purchase and fixed order costs are charged when an order is placed, even if delivery falls after the test end. Such pipeline units are reported separately. Costs use source holding cost and selling price as the lost-sales penalty.\n\n"
        "Limitations: results are for one synthetic historical holdout and one fixed random lead-time realization per product/window; they are not universal evidence. The optimizer receives no outstanding pipeline at window boundaries (the tested windows exceed typical lead times, but delayed orders can remain after the final window). Supplier reliability is represented by one delayed retry only. Future promotion/festival covariates in Module 2 forecasts default to zero. Sensitivity replays do not re-optimize decisions.\n")
    (SIM_DIR/"evaluation_methodology.md").write_text(methodology,encoding="utf-8")
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=704); parser.add_argument("--scenario-count",type=int,default=25)
    parser.add_argument("--lead-seed",type=int,default=404); parser.add_argument("--window-days",type=int,default=WINDOW_DAYS)
    args=parser.parse_args()
    result=run(args.seed,args.scenario_count,args.lead_seed,args.window_days)
    print(result[["policy","total_inventory_cost_inr","fill_rate","total_lost_sales_units","stockout_frequency","computational_time_seconds"]].to_string(index=False))

if __name__=="__main__": main()
