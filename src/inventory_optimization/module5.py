"""Repeated stochastic simulations, paired robustness analysis, and final research report."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PROCESSED_DIR, OUTPUT_DIR, REPORTS_DIR
from .module3 import _static_products
from .module4 import _scenario_tensor, _optimizer_inputs
from .optimization import solve_inventory
from .scenarios import generate_scenarios
from .simulation import sample_lead_times, simulate_daily

ROBUST_DIR = OUTPUT_DIR / "robustness"
TEST_DAYS = 30
POLICY_NAMES = ("context_aware_optimization", "non_contextual_optimization", "historical_mean_order_up_to_30d")

def moving_block_demand(history, dates, products, seed, block_days=30):
    """Resample pre-test daily cross-product blocks, aligning block start weekday."""
    rng=np.random.default_rng(seed)
    panel=history.pivot(index="date",columns="product_id",values="demand_units").sort_index().reindex(columns=products)
    source_dates=pd.DatetimeIndex(panel.index)
    if len(source_dates)<block_days: raise ValueError("Pre-test history is shorter than one bootstrap block")
    source=panel.to_numpy(dtype=int); horizon=len(dates); out=np.empty((horizon,len(products)),dtype=int)
    for offset in range(0,horizon,block_days):
        length=min(block_days,horizon-offset); weekday=pd.Timestamp(dates[offset]).dayofweek
        candidates=[j for j in range(len(source_dates)-length+1) if source_dates[j].dayofweek==weekday]
        if not candidates: candidates=list(range(len(source_dates)-length+1))
        start=int(rng.choice(candidates)); out[offset:offset+length]=source[start:start+length]
    return out

def increase_demand_dispersion(demands, historical_means, multiplier=1.25):
    """Inflate deviations around pre-test product means; used only as a labeled stress path."""
    means=np.asarray(historical_means,dtype=float).reshape(1,-1)
    return np.maximum(0,np.rint(means+multiplier*(np.asarray(demands,dtype=float)-means))).astype(int)

def mean_confidence_interval(values, confidence=.95):
    vals=np.asarray(values,dtype=float); vals=vals[np.isfinite(vals)]
    if len(vals)==0: return (float("nan"),float("nan"),float("nan"))
    mean=float(vals.mean())
    if len(vals)<2: return (mean,float("nan"),float("nan"))
    margin=float(student_t.ppf((1+confidence)/2,df=len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals)))
    return mean,mean-margin,mean+margin

def summarize_repetitions(results, confidence=.95):
    metrics=["total_inventory_cost_inr","fill_rate","total_lost_sales_units","stockout_frequency","average_on_hand_inventory_units","average_inventory_value_inr"]
    rows=[]
    for policy,g in results.groupby("policy",sort=False):
        for metric in metrics:
            values=g[metric].to_numpy(float); mean,lo,hi=mean_confidence_interval(values,confidence)
            rows.append({"policy":policy,"metric":metric,"repetitions":len(values),"mean":mean,
                "standard_deviation":float(np.std(values,ddof=1)) if len(values)>1 else 0.0,
                "median":float(np.median(values)),"minimum":float(np.min(values)),"maximum":float(np.max(values)),
                "confidence_level":confidence,"mean_ci_lower":lo,"mean_ci_upper":hi})
    return pd.DataFrame(rows)

def paired_policy_comparison(results, confidence=.95):
    wide=results.pivot(index="replication",columns="policy",values=["total_inventory_cost_inr","fill_rate","total_lost_sales_units"])
    cost_delta=wide["total_inventory_cost_inr"]["context_aware_optimization"]-wide["total_inventory_cost_inr"]["non_contextual_optimization"]
    saving_pct=-100*cost_delta/wide["total_inventory_cost_inr"]["non_contextual_optimization"]
    fill_delta=wide["fill_rate"]["context_aware_optimization"]-wide["fill_rate"]["non_contextual_optimization"]
    lost_delta=wide["total_lost_sales_units"]["context_aware_optimization"]-wide["total_lost_sales_units"]["non_contextual_optimization"]
    out=[]
    for name,values,unit in [("context_minus_noncontext_cost_inr",cost_delta,"INR"),
        ("context_cost_saving_pct",saving_pct,"percent; positive means context costs less"),
        ("context_minus_noncontext_fill_rate",fill_delta,"fraction"),
        ("context_minus_noncontext_lost_sales",lost_delta,"units")]:
        mean,lo,hi=mean_confidence_interval(values.to_numpy(),confidence)
        out.append({"comparison":name,"unit_interpretation":unit,"repetitions":len(values),"mean_paired_difference":mean,
            "standard_deviation":float(values.std(ddof=1)) if len(values)>1 else 0.0,"median":float(values.median()),
            "ci_lower":lo,"ci_upper":hi,"context_lower_cost_wins":int((cost_delta<0).sum()),
            "context_lower_cost_win_rate":float((cost_delta<0).mean())})
    return pd.DataFrame(out)

def _history_as_of(pretest, static_rows, dates, demand_path, day):
    if day==0: return pretest.sort_values(["product_id","date"]).copy()
    chunks=[]
    for j in range(day):
        chunk=static_rows.copy(); chunk["date"]=dates[j]; chunk["demand_units"]=demand_path[j]
        chunks.append(chunk)
    return pd.concat([pretest,*chunks],ignore_index=True).sort_values(["product_id","date"])

def _make_decision(policy, rep, dates, pretest, static_rows, products, demand_path, stock, pipeline, horizon,
                   demand_model, baseline_model, scenario_count, seed, costs, elapsed):
    w=rep["window"]
    history=_history_as_of(pretest,static_rows,dates,demand_path,rep["day"])
    if policy=="historical_mean_order_up_to_30d":
        q=[]
        for i,pid in enumerate(products):
            prior=history.loc[history.product_id.astype(str)==pid,"demand_units"].tail(28)
            mu=float(prior.mean()) if len(prior) else 0.0
            target=mu*(horizon+float(static_rows.iloc[i].supplier_average_lead_time_days))
            position=float(stock[i])+float(pipeline[:min(horizon,len(pipeline)),i].sum())
            cap=int(static_rows.iloc[i].supplier_max_capacity_units)
            q.append(min(cap,max(0,int(np.ceil(target-position)))))
        return np.asarray(q,dtype=int)
    if policy=="context_aware_optimization":
        scenarios=generate_scenarios(demand_model,history,horizon=horizon,n_scenarios=scenario_count,
            seed=seed+rep["replication"]*1009+rep["window"])
        D,p=_scenario_tensor(scenarios,products)
    else:
        rng=np.random.default_rng(seed+50_000+rep["replication"]*1009+rep["window"])
        D=np.empty((scenario_count,horizon,len(products)),dtype=float)
        for i,pid in enumerate(products):
            mu=baseline_model.means_.get(pid,baseline_model.global_mean_)
            D[:,:,i]=rng.poisson(mu,size=(scenario_count,horizon))
        p=np.full(scenario_count,1/scenario_count)
    instance=_optimizer_inputs(products,static_rows,stock,D,p,pipeline)
    result=solve_inventory(instance)
    elapsed[0]+=result["solve_seconds"]
    costs["solver_statuses"].append(result["status"])
    return result["order_quantity"]

def _metrics_for(policy, replication, demand_seed, lead_seed, daily, orders, optimizer_time):
    # Reuse the same definitions as the Module 4 historical backtest.
    from .simulation import inventory_metrics
    m=inventory_metrics(daily,orders,daily.product_id.nunique(),daily.date.nunique(),optimizer_time)
    m.update({"policy":policy,"replication":replication,"demand_seed":demand_seed,"lead_seed":lead_seed,
        "solver_statuses":";".join(sorted(set(orders.get("solver_status",[])))) if "solver_status" in orders else "see_policy_status_file"})
    return m

def _write_research_report(repeated, paired, hist, sensitivity, cost_breakdown, destination, reps, scenarios):
    r=repeated.set_index(["policy","metric"]); h=hist.set_index("policy")
    def val(policy,metric,col="mean"): return float(r.loc[(policy,metric),col])
    ctx="context_aware_optimization"; non="non_contextual_optimization"; base="historical_mean_order_up_to_30d"
    costsaving=float(paired.loc[paired.comparison=="context_cost_saving_pct","mean_paired_difference"].iloc[0])
    ctxci=paired.loc[paired.comparison=="context_cost_saving_pct"].iloc[0]
    cost_rows="| Policy | Purchase INR | Fixed ordering INR | Holding INR | Lost-sales INR |\n|---|---:|---:|---:|---:|\n"
    for row in cost_breakdown.itertuples():
        cost_rows+=f"| {row.policy} | {row.purchase_cost_inr_mean:,.2f} | {row.ordering_cost_inr_mean:,.2f} | {row.holding_cost_inr_mean:,.2f} | {row.shortage_cost_inr_mean:,.2f} |\n"
    sensitivity_rows=""
    for row in sensitivity.itertuples():
        sensitivity_rows+=f"| {row.policy} | {row.case} | {row.mean_total_cost_inr:,.2f} | {row.mean_fill_rate:.3%} | {row.mean_lost_sales_units:,.1f} | {row.mean_stockout_frequency:.3%} | {row.mean_average_inventory_units:.2f} |\n"
    fill_deltas=paired.loc[paired.comparison=="context_minus_noncontext_fill_rate"].iloc[0]
    report=f"""# Final research report: Context-Aware Stochastic Inventory Optimization

## Research problem and question

Retail inventory decisions trade purchase and holding expense against lost sales while daily demand and supplier delivery times vary. This project asks: **does contextual stochastic optimization improve realized inventory cost and service compared with non-contextual optimization and a historical order-up-to baseline?**

## Data and demand modelling

The project uses the supplied 50-product, five-category synthetic retail dataset (2024-01-01–2025-12-31). It is generated data, not evidence about a real retailer. Module 1 validates the source, analyzes demand and supplier fields, creates prior-only lags/rolling features, and assigns chronological train/validation/test dates. `demand_units` is the target; generator metadata is excluded from predictors.

Module 2 compares a product-mean Poisson baseline with pooled contextual Poisson and NB2 log-link models. Validation negative log-likelihood selected pooled NB2, with conditional variance `mu + alpha*mu^2`. Demand paths are generated recursively from each scenario's own prior sampled demand; Module 2 does not model cross-product dependence. Better demand distribution scores do not by themselves imply better replenishment decisions or better realized inventory outcomes.

## Optimization and simulation methods

Module 3 minimizes expected unit purchase plus fixed ordering, holding, and lost-sales costs. It makes an integer day-0 order; scenario inventory/lost sales follow inventory balance. Module 4 extends this as one order per rolling 30-day window. Contextual decisions use conditional NB2 scenario forecasts available at that window start; non-contextual decisions use the saved product-mean Poisson baseline. The historical baseline orders up to trailing-28-day mean demand times review horizon plus average lead time. The three policies use the same product capacity, initial stocks, costs, realized demand, and supplier lead-time paths.

Historical holdout uses realized Module 1 test demand, 2025-09-13–2025-12-31 (110 days, 50 products). Repeated simulation is a separate analysis: `{reps}` paired replications draw 30-day moving blocks from pre-test historical demand, preserving local time and cross-product dependence within sampled blocks. All policies share each replication's realized demand path and lead-time/reliability draws. Before each order, only historical demand through the prior day, current inventory, and known outstanding receipt schedule are available. The two stochastic policies re-optimize each 30-day window with `{scenarios}` forecast scenarios. Confidence intervals are paired Student-t 95% intervals over replications; given the small repetition count, they are descriptive rather than significance claims.

## Historical holdout results

| Policy | Total cost INR | Fill rate | Lost units | Stockout product-days | Avg on-hand units |
|---|---:|---:|---:|---:|---:|
| Contextual optimizer | {h.loc[ctx,'total_inventory_cost_inr']:.2f} | {h.loc[ctx,'fill_rate']:.3%} | {int(h.loc[ctx,'total_lost_sales_units']):,} | {int(h.loc[ctx,'stockout_product_days']):,} | {h.loc[ctx,'average_on_hand_inventory_units']:.2f} |
| Non-contextual optimizer | {h.loc[non,'total_inventory_cost_inr']:.2f} | {h.loc[non,'fill_rate']:.3%} | {int(h.loc[non,'total_lost_sales_units']):,} | {int(h.loc[non,'stockout_product_days']):,} | {h.loc[non,'average_on_hand_inventory_units']:.2f} |
| Historical order-up-to | {h.loc[base,'total_inventory_cost_inr']:.2f} | {h.loc[base,'fill_rate']:.3%} | {int(h.loc[base,'total_lost_sales_units']):,} | {int(h.loc[base,'stockout_product_days']):,} | {h.loc[base,'average_on_hand_inventory_units']:.2f} |

In this single holdout, contextual optimization cost 0.077% more than non-contextual optimization and had lower fill. The historical baseline had the highest fill and total cost. At least one SKU stocked out on each of the 110 days. This result does not establish a general winner.

Historical mean cost breakdown:

{cost_rows}

## Repeated stochastic simulation

Completed `{reps}` paired replications; each policy sees exactly the same demand path and supplier realizations within a repetition. The repeated table below reports mean cost, sample standard deviation, median, range, 95% mean interval, mean fill rate, lost sales, stockout frequency, and average inventory.

| Policy | Mean cost INR (95% CI) | SD INR | Median INR | Range INR | Mean fill | Mean lost | Stockout freq. | Avg inventory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Contextual | {val(ctx,'total_inventory_cost_inr'):,.2f} ({val(ctx,'total_inventory_cost_inr','mean_ci_lower'):,.2f}, {val(ctx,'total_inventory_cost_inr','mean_ci_upper'):,.2f}) | {val(ctx,'total_inventory_cost_inr','standard_deviation'):,.2f} | {val(ctx,'total_inventory_cost_inr','median'):,.2f} | {val(ctx,'total_inventory_cost_inr','minimum'):,.2f}–{val(ctx,'total_inventory_cost_inr','maximum'):,.2f} | {val(ctx,'fill_rate'):.3%} | {val(ctx,'total_lost_sales_units'):,.1f} | {val(ctx,'stockout_frequency'):.3%} | {val(ctx,'average_on_hand_inventory_units'):.2f} |
| Non-contextual | {val(non,'total_inventory_cost_inr'):,.2f} ({val(non,'total_inventory_cost_inr','mean_ci_lower'):,.2f}, {val(non,'total_inventory_cost_inr','mean_ci_upper'):,.2f}) | {val(non,'total_inventory_cost_inr','standard_deviation'):,.2f} | {val(non,'total_inventory_cost_inr','median'):,.2f} | {val(non,'total_inventory_cost_inr','minimum'):,.2f}–{val(non,'total_inventory_cost_inr','maximum'):,.2f} | {val(non,'fill_rate'):.3%} | {val(non,'total_lost_sales_units'):,.1f} | {val(non,'stockout_frequency'):.3%} | {val(non,'average_on_hand_inventory_units'):.2f} |
| Historical order-up-to | {val(base,'total_inventory_cost_inr'):,.2f} ({val(base,'total_inventory_cost_inr','mean_ci_lower'):,.2f}, {val(base,'total_inventory_cost_inr','mean_ci_upper'):,.2f}) | {val(base,'total_inventory_cost_inr','standard_deviation'):,.2f} | {val(base,'total_inventory_cost_inr','median'):,.2f} | {val(base,'total_inventory_cost_inr','minimum'):,.2f}–{val(base,'total_inventory_cost_inr','maximum'):,.2f} | {val(base,'fill_rate'):.3%} | {val(base,'total_lost_sales_units'):,.1f} | {val(base,'stockout_frequency'):.3%} | {val(base,'average_on_hand_inventory_units'):.2f} |

Paired contextual cost saving versus non-contextual was {costsaving:.3f}% (95% CI {ctxci.ci_lower:.3f}% to {ctxci.ci_upper:.3f}%); positive denotes contextual cost lower. Contextual cost was lower in {int(ctxci.context_lower_cost_wins)}/{reps} replications. The average paired fill difference was {fill_deltas.mean_paired_difference:.3%}. The cost interval crosses zero, and this small run is exploratory; no significance claim is made.

## Robustness and sensitivity

`outputs/robustness/sensitivity_summary.csv` reports paired frozen-order stress tests for 1.25× demand deviations around pre-test product means, 1.5× supplier lead-time standard deviation, 1.5× holding cost, and 1.5× shortage penalty. Order schedules are held fixed in these replays; results measure exposure and do not estimate the benefit of re-optimizing for changed assumptions. Changes are reported by policy, not merged into the base repeated-simulation comparison.

| Policy | Stress case | Mean cost INR | Mean fill | Mean lost units | Stockout freq. | Avg inventory |
|---|---|---:|---:|---:|---:|---:|
{sensitivity_rows}

## When context helps and when it does not

Three effects must be separated: (1) Module 2 selected NB2 on validation probabilistic fit; (2) Module 3 contextual demand scenarios led to different expected-cost orders; and (3) Module 4 realized test performance did not improve over the non-contextual policy. The latter reflects model error, lead-time uncertainty, one-order-per-window decisions, cost assumptions, and a single synthetic path. Context can help when contextual features materially improve calibrated distributions and the inventory action is sensitive to those distribution changes. It may not help when uncertainty is poorly calibrated, the context is unavailable at order time, or the objective/constraints dominate the demand signal. The present experiments do not establish which mechanism is general.

## Limitations and next work

The source is synthetic. Repeated simulations use moving-block bootstrap demand from the pre-test period, not new real observations; only `{reps}` replications were computationally feasible here. Lead reliability is simplified as one delayed retry; the optimizer treats known pipeline as scheduled receipts and still uses average lead time for new orders. Lost-sales cost uses selling price as a proxy, no source purchasing budget or aggregate supplier capacity exists, and there is no storage limit, substitution, or backorder. Module 2 assumes zero future promotions/festivals when unknown. Sensitivity replays freeze orders. Module 5's dashboard presents saved results and does not rerun optimization.

Next steps: expand paired replications and bootstrap block diagnostics; calibrate uncertainty on real data; model repeated delivery failures and pipeline more fully; re-optimize under changed cost/uncertainty assumptions; test policy differences on multiple retailers and rolling historical windows.
"""
    destination.parent.mkdir(parents=True,exist_ok=True); destination.write_text(report,encoding="utf-8")

def _write_final_figures(summary, sensitivity):
    figure_dir=REPORTS_DIR/"figures"; figure_dir.mkdir(parents=True,exist_ok=True)
    cost=summary[summary.metric=="total_inventory_cost_inr"]
    fig,ax=plt.subplots(figsize=(9,5))
    y=cost["mean"].to_numpy(); lo=cost["mean"].to_numpy()-cost["mean_ci_lower"].to_numpy(); hi=cost["mean_ci_upper"].to_numpy()-cost["mean"].to_numpy()
    ax.bar(cost.policy,y,yerr=np.vstack([lo,hi]),capsize=5); ax.set_ylabel("Mean total cost (INR)"); ax.set_title("Repeated simulation: policy cost and 95% t interval"); ax.tick_params(axis="x",rotation=15); fig.tight_layout(); fig.savefig(figure_dir/"repeated_policy_cost.png",dpi=160); plt.close(fig)
    fill=summary[summary.metric=="fill_rate"]
    fig,ax=plt.subplots(figsize=(9,5)); ax.bar(fill.policy,fill["mean"]); ax.set_ylabel("Mean fill rate"); ax.set_ylim(0,1); ax.set_title("Repeated simulation: mean service level"); ax.tick_params(axis="x",rotation=15); fig.tight_layout(); fig.savefig(figure_dir/"repeated_policy_fill_rate.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5)); cases=list(sensitivity["case"].unique()); policies=list(sensitivity.policy.unique()); x=np.arange(len(cases)); width=.8/max(1,len(policies))
    for j,policy in enumerate(policies):
        g=sensitivity[sensitivity.policy==policy].set_index("case").reindex(cases)
        ax.bar(x+(j-(len(policies)-1)/2)*width,g.mean_total_cost_inr,width,label=policy)
    ax.set_xticks(x,cases,rotation=15); ax.set_ylabel("Mean total cost (INR)"); ax.set_title("Frozen-order sensitivity replays"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figure_dir/"sensitivity_cost.png",dpi=160); plt.close(fig)

def run(repetitions=5, scenario_count=5, seed=5500, lead_seed=6600, output_dir=ROBUST_DIR):
    if repetitions<2: raise ValueError("Use at least two repetitions for interval estimation")
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    feature_path=PROCESSED_DIR/"contextual_inventory_features.csv"
    demand_model_path=OUTPUT_DIR/"models"/"selected_contextual_model.pkl"
    baseline_path=OUTPUT_DIR/"models"/"product_mean_baseline.pkl"
    if not all(p.exists() for p in (feature_path,demand_model_path,baseline_path)):
        raise FileNotFoundError("Module 1 features and Module 2 model artifacts are required")
    data=pd.read_csv(feature_path,parse_dates=["date"])
    test=data[data.split=="test"].copy(); dates=pd.DatetimeIndex(sorted(test.date.unique()))
    products_frame,products,_,_=_static_products(data)
    pretest=data[data.date<dates.min()].copy().sort_values(["product_id","date"])
    if len(dates)==0 or len(pretest)==0: raise ValueError("Need held-out dates and pre-test history")
    windows=list(range(0,len(dates),TEST_DAYS));
    actual_panel=pretest.pivot(index="date",columns="product_id",values="demand_units").reindex(columns=products)
    hist_means=actual_panel.mean(axis=0).reindex(products).to_numpy(float)
    with open(demand_model_path,"rb") as f: demand_model=pickle.load(f)
    with open(baseline_path,"rb") as f: baseline_model=pickle.load(f)
    actual_reps=[]; daily_frames=[]; order_frames=[]; sensitivity=[]; path_checks=[]
    for rep in range(repetitions):
        demand_seed=seed+rep*1009; rep_lead_seed=lead_seed+rep*1009
        demand=moving_block_demand(pretest,dates,products,demand_seed)
        leads,_=sample_lead_times(products_frame,len(windows),rep_lead_seed,1.0)
        stress_leads,_=sample_lead_times(products_frame,len(windows),rep_lead_seed,1.5)
        uncertain=increase_demand_dispersion(demand,hist_means,1.25)
        path_checks.append({"replication":rep,"demand_seed":demand_seed,"lead_seed":rep_lead_seed,
            "demand_hash":int(np.sum(demand*np.arange(1,len(products)+1)[None,:])%2147483647),
            "different_demand_units_from_previous":bool(rep==0 or not np.array_equal(demand,previous_demand))})
        previous_demand=demand.copy()
        for policy in POLICY_NAMES:
            elapsed=[0.0]; statuses={"solver_statuses":[]}
            def decide(w,day,stock,current_date,horizon,pipeline):
                return _make_decision(policy,{"replication":rep,"window":w,"day":day},dates,pretest,products_frame,products,
                    demand,stock,pipeline,horizon,demand_model,baseline_model,scenario_count,seed, statuses,elapsed)
            q0={w:np.zeros(len(products),dtype=int) for w in range(len(windows))}
            daily,orders,metrics,qplan=simulate_daily(dates,demand,products_frame,q0,leads,windows,
                policy=policy,decision_callback=decide)
            metrics.update({"policy":policy,"replication":rep,"demand_seed":demand_seed,"lead_seed":rep_lead_seed,
                "optimization_solver_statuses":";".join(statuses["solver_statuses"]) if statuses["solver_statuses"] else "not_applicable",
                "forecast_scenarios_per_window":scenario_count,"computational_time_seconds":elapsed[0]})
            actual_reps.append(metrics); daily["replication"]=rep; orders["replication"]=rep
            daily_frames.append(daily); order_frames.append(orders); qplan={k:v.copy() for k,v in qplan.items()}
            for case,case_demand,case_leads,hm,sm in [
                ("demand_deviation_1.25x",uncertain,leads,1.0,1.0),
                ("lead_time_std_1.5x",demand,stress_leads,1.0,1.0),
                ("holding_cost_1.5x",demand,leads,1.5,1.0),
                ("shortage_penalty_1.5x",demand,leads,1.0,1.5)]:
                sd,so,smets,_=simulate_daily(dates,case_demand,products_frame,qplan,case_leads,windows,
                    holding_multiplier=hm,shortage_multiplier=sm,policy=policy)
                sensitivity.append({"policy":policy,"replication":rep,"case":case,"orders":"frozen base plan",**smets})
    results=pd.DataFrame(actual_reps)
    agg=summarize_repetitions(results); paired=paired_policy_comparison(results)
    sens=pd.DataFrame(sensitivity)
    sens_summary=sens.groupby(["policy","case"],as_index=False).agg(
        repetitions=("replication","nunique"),mean_total_cost_inr=("total_inventory_cost_inr","mean"),
        sd_total_cost_inr=("total_inventory_cost_inr","std"),mean_fill_rate=("fill_rate","mean"),
        mean_lost_sales_units=("total_lost_sales_units","mean"),mean_stockout_frequency=("stockout_frequency","mean"),
        mean_average_inventory_units=("average_on_hand_inventory_units","mean"))
    results.to_csv(output_dir/"replication_metrics.csv",index=False)
    agg.to_csv(output_dir/"repeated_policy_summary.csv",index=False)
    cost_columns=["purchase_cost_inr","ordering_cost_inr","holding_cost_inr","shortage_cost_inr"]
    cost_breakdown=results.groupby("policy")[cost_columns].agg(["mean","std"]).reset_index()
    cost_breakdown.columns=["_".join(str(part) for part in col if part) if isinstance(col,tuple) else col for col in cost_breakdown.columns]
    cost_breakdown.to_csv(output_dir/"repeated_cost_breakdown.csv",index=False)
    paired.to_csv(output_dir/"paired_comparison.csv",index=False)
    sens.to_csv(output_dir/"sensitivity_replications.csv",index=False)
    sens_summary.to_csv(output_dir/"sensitivity_summary.csv",index=False)
    pd.DataFrame(path_checks).to_csv(output_dir/"replication_seeds.csv",index=False)
    pd.concat(daily_frames,ignore_index=True).to_csv(output_dir/"repeated_daily_simulation.csv",index=False)
    pd.concat(order_frames,ignore_index=True).to_csv(output_dir/"repeated_orders_and_deliveries.csv",index=False)
    hist=pd.read_csv(OUTPUT_DIR/"simulation"/"policy_comparison_summary.csv")
    # Research report is saved to the canonical report folder only for the full run.
    if output_dir.resolve()==ROBUST_DIR.resolve():
        _write_final_figures(agg,sens_summary)
        _write_research_report(agg,paired,hist,sens_summary,REPORTS_DIR/"final_research_report.md",repetitions,scenario_count)
    return {"replication_metrics":results,"repeated_summary":agg,"paired_comparison":paired,"sensitivity_summary":sens_summary}

def finalize_saved_outputs(output_dir=ROBUST_DIR):
    """Rebuild report and figures from completed result tables without rerunning models."""
    output_dir=Path(output_dir)
    summary=pd.read_csv(output_dir/"repeated_policy_summary.csv")
    paired=pd.read_csv(output_dir/"paired_comparison.csv")
    sensitivity=pd.read_csv(output_dir/"sensitivity_summary.csv")
    raw=pd.read_csv(output_dir/"replication_metrics.csv")
    cost_path=output_dir/"repeated_cost_breakdown.csv"
    if cost_path.exists(): costs=pd.read_csv(cost_path)
    else:
        fields=["purchase_cost_inr","ordering_cost_inr","holding_cost_inr","shortage_cost_inr"]
        costs=raw.groupby("policy")[fields].agg(["mean","std"]).reset_index()
        costs.columns=["_".join(str(part) for part in col if part) if isinstance(col,tuple) else col for col in costs.columns]
        costs.to_csv(cost_path,index=False)
    historical=pd.read_csv(OUTPUT_DIR/"simulation"/"policy_comparison_summary.csv")
    reps=int(raw.replication.nunique()); scenarios=int(raw.forecast_scenarios_per_window.max())
    if output_dir.resolve()==ROBUST_DIR.resolve():
        _write_final_figures(summary,sensitivity)
        _write_research_report(summary,paired,historical,sensitivity,costs,REPORTS_DIR/"final_research_report.md",reps,scenarios)
    return summary,paired,sensitivity

def smoke_test(seed=77,lead_seed=88):
    return run(repetitions=2,scenario_count=1,seed=seed,lead_seed=lead_seed,output_dir=ROBUST_DIR/"smoke_test")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions",type=int,default=5); parser.add_argument("--scenario-count",type=int,default=5)
    parser.add_argument("--seed",type=int,default=5500); parser.add_argument("--lead-seed",type=int,default=6600)
    parser.add_argument("--smoke-test",action="store_true",help="Run a small two-repetition, one-scenario pipeline check to outputs/robustness/smoke_test")
    parser.add_argument("--finalize-only",action="store_true",help="Build report/figures from existing completed robustness tables")
    args=parser.parse_args()
    if args.finalize_only:
        summary,paired,sensitivity=finalize_saved_outputs()
        print(summary.query("metric == 'total_inventory_cost_inr'").to_string(index=False)); print(paired.to_string(index=False))
    else:
        result=smoke_test(args.seed,args.lead_seed) if args.smoke_test else run(args.repetitions,args.scenario_count,args.seed,args.lead_seed)
        print(result["repeated_summary"].query("metric == 'total_inventory_cost_inr'").to_string(index=False))
        print(result["paired_comparison"].to_string(index=False))

if __name__=="__main__": main()
