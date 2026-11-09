# Final research report: Context-Aware Stochastic Inventory Optimization

## Research problem and question

Retail inventory decisions trade purchase and holding expense against lost sales while daily demand and supplier delivery times vary. This project asks: **does contextual stochastic optimization improve realized inventory cost and service compared with non-contextual optimization and a historical order-up-to baseline?**

## Data and demand modelling

The project uses the supplied 50-product, five-category synthetic retail dataset (2024-01-01–2025-12-31). It is generated data, not evidence about a real retailer. Module 1 validates the source, analyzes demand and supplier fields, creates prior-only lags/rolling features, and assigns chronological train/validation/test dates. `demand_units` is the target; generator metadata is excluded from predictors.

Module 2 compares a product-mean Poisson baseline with pooled contextual Poisson and NB2 log-link models. Validation negative log-likelihood selected pooled NB2, with conditional variance `mu + alpha*mu^2`. Demand paths are generated recursively from each scenario's own prior sampled demand; Module 2 does not model cross-product dependence. Better demand distribution scores do not by themselves imply better replenishment decisions or better realized inventory outcomes.

## Optimization and simulation methods

Module 3 minimizes expected unit purchase plus fixed ordering, holding, and lost-sales costs. It makes an integer day-0 order; scenario inventory/lost sales follow inventory balance. Module 4 extends this as one order per rolling 30-day window. Contextual decisions use conditional NB2 scenario forecasts available at that window start; non-contextual decisions use the saved product-mean Poisson baseline. The historical baseline orders up to trailing-28-day mean demand times review horizon plus average lead time. The three policies use the same product capacity, initial stocks, costs, realized demand, and supplier lead-time paths.

Historical holdout uses realized Module 1 test demand, 2025-09-13–2025-12-31 (110 days, 50 products). Repeated simulation is a separate analysis: `3` paired replications draw 30-day moving blocks from pre-test historical demand, preserving local time and cross-product dependence within sampled blocks. All policies share each replication's realized demand path and lead-time/reliability draws. Before each order, only historical demand through the prior day, current inventory, and known outstanding receipt schedule are available. The two stochastic policies re-optimize each 30-day window with `2` forecast scenarios. Confidence intervals are paired Student-t 95% intervals over replications; given the small repetition count, they are descriptive rather than significance claims.

## Historical holdout results

| Policy | Total cost INR | Fill rate | Lost units | Stockout product-days | Avg on-hand units |
|---|---:|---:|---:|---:|---:|
| Contextual optimizer | 43232958.43 | 45.078% | 156,797 | 2,512 | 154.22 |
| Non-contextual optimizer | 43199584.79 | 46.767% | 151,975 | 2,382 | 175.49 |
| Historical order-up-to | 43770668.47 | 47.679% | 149,370 | 2,253 | 198.98 |

In this single holdout, contextual optimization cost 0.077% more than non-contextual optimization and had lower fill. The historical baseline had the highest fill and total cost. At least one SKU stocked out on each of the 110 days. This result does not establish a general winner.

Historical mean cost breakdown:

| Policy | Purchase INR | Fixed ordering INR | Holding INR | Lost-sales INR |
|---|---:|---:|---:|---:|
| context_aware_optimization | 15,484,323.33 | 22,468.00 | 407,980.73 | 29,571,826.03 |
| historical_mean_order_up_to_30d | 16,942,487.67 | 21,965.67 | 509,004.57 | 28,437,307.34 |
| non_contextual_optimization | 16,146,361.33 | 22,427.33 | 445,885.79 | 28,809,394.90 |


## Repeated stochastic simulation

Completed `3` paired replications; each policy sees exactly the same demand path and supplier realizations within a repetition. The repeated table below reports mean cost, sample standard deviation, median, range, 95% mean interval, mean fill rate, lost sales, stockout frequency, and average inventory.

| Policy | Mean cost INR (95% CI) | SD INR | Median INR | Range INR | Mean fill | Mean lost | Stockout freq. | Avg inventory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Contextual | 45,486,598.10 (42,134,804.84, 48,838,391.35) | 1,349,278.36 | 44,882,244.47 | 44,545,143.29–47,032,406.53 | 43.624% | 168,698.3 | 46.782% | 153.33 |
| Non-contextual | 45,424,069.35 (42,223,124.35, 48,625,014.35) | 1,288,553.76 | 44,867,009.06 | 44,507,763.24–46,897,435.75 | 44.890% | 164,905.0 | 44.994% | 167.16 |
| Historical order-up-to | 45,910,765.25 (42,407,602.31, 49,413,928.18) | 1,410,212.85 | 45,329,736.39 | 44,883,894.23–47,518,665.11 | 45.920% | 161,842.7 | 43.339% | 195.40 |

Paired contextual cost saving versus non-contextual was -0.135% (95% CI -0.469% to 0.199%); positive denotes contextual cost lower. Contextual cost was lower in 0/3 replications. The mean paired fill difference was -1.267 percentage points. The cost interval crosses zero, and this small run is exploratory; no significance claim is made.

## Robustness and sensitivity

`outputs/robustness/sensitivity_summary.csv` reports paired frozen-order stress tests for 1.25× demand deviations around pre-test product means, 1.5× supplier lead-time standard deviation, 1.5× holding cost, and 1.5× shortage penalty. Order schedules are held fixed in these replays; results measure exposure and do not estimate the benefit of re-optimizing for changed assumptions. Changes are reported by policy, not merged into the base repeated-simulation comparison.

| Policy | Case | Mean cost INR | Mean fill | Lost units | Stockout frequency | Avg inventory units |
|---|---|---:|---:|---:|---:|---:|
| context_aware_optimization | demand_deviation_1.25x | 46,288,943.79 | 42.880% | 173,740.7 | 46.685% | 152.69 |
| context_aware_optimization | holding_cost_1.5x | 45,690,588.46 | 43.624% | 168,698.3 | 46.782% | 153.33 |
| context_aware_optimization | lead_time_std_1.5x | 45,544,829.81 | 43.525% | 168,993.0 | 46.927% | 154.72 |
| context_aware_optimization | shortage_penalty_1.5x | 60,272,511.11 | 43.624% | 168,698.3 | 46.782% | 153.33 |
| historical_mean_order_up_to_30d | demand_deviation_1.25x | 46,667,625.21 | 45.221% | 166,643.3 | 43.224% | 194.09 |
| historical_mean_order_up_to_30d | holding_cost_1.5x | 46,165,267.53 | 45.920% | 161,842.7 | 43.339% | 195.40 |
| historical_mean_order_up_to_30d | lead_time_std_1.5x | 45,952,283.74 | 45.842% | 162,074.3 | 43.467% | 196.85 |
| historical_mean_order_up_to_30d | shortage_penalty_1.5x | 60,129,418.92 | 45.920% | 161,842.7 | 43.339% | 195.40 |
| non_contextual_optimization | demand_deviation_1.25x | 46,222,886.48 | 44.132% | 169,928.3 | 45.042% | 166.42 |
| non_contextual_optimization | holding_cost_1.5x | 45,647,012.24 | 44.890% | 164,905.0 | 44.994% | 167.16 |
| non_contextual_optimization | lead_time_std_1.5x | 45,480,771.73 | 44.785% | 165,221.7 | 45.188% | 168.49 |
| non_contextual_optimization | shortage_penalty_1.5x | 59,828,766.80 | 44.890% | 164,905.0 | 44.994% | 167.16 |


## When context helps and when it does not

Three effects must be separated: (1) Module 2 selected NB2 on validation probabilistic fit; (2) Module 3 contextual demand scenarios led to different expected-cost orders; and (3) Module 4 realized test performance did not improve over the non-contextual policy. The latter reflects model error, lead-time uncertainty, one-order-per-window decisions, cost assumptions, and a single synthetic path. Context can help when contextual features materially improve calibrated distributions and the inventory action is sensitive to those distribution changes. It may not help when uncertainty is poorly calibrated, the context is unavailable at order time, or the objective/constraints dominate the demand signal. The present experiments do not establish which mechanism is general.

## Limitations and next work

The source is synthetic. Repeated simulations use moving-block bootstrap demand from the pre-test period, not new real observations; only `3` replications were computationally feasible here. Lead reliability is simplified as one delayed retry; the optimizer treats known pipeline as scheduled receipts and still uses average lead time for new orders. Lost-sales cost uses selling price as a proxy, no source purchasing budget or aggregate supplier capacity exists, and there is no storage limit, substitution, or backorder. Module 2 assumes zero future promotions/festivals when unknown. Sensitivity replays freeze orders. Module 5's dashboard presents saved results and does not rerun optimization.

Next steps: expand paired replications and bootstrap block diagnostics; calibrate uncertainty on real data; model repeated delivery failures and pipeline more fully; re-optimize under changed cost/uncertainty assumptions; test policy differences on multiple retailers and rolling historical windows.
