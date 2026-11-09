# Project status

## Module 1 — data preparation and exploratory analysis

**Status:** Complete and executed successfully.

- Reads the provided synthetic inventory dataset and data dictionary without modifying either raw CSV.
- Validates schema, missingness, duplicates, date/key integrity, types, categorical values, and domain constraints.
- Produces descriptive summaries, product/category and contextual demand analyses, supplier profiles, and figures.
- Builds calendar, festival, promotion, discount, lag, and rolling demand features. Lags and rolling statistics use historical observations only.
- Assigns chronological train/validation/test labels by date.
- Keeps `demand_units` as the target and excludes `generated_mean_demand` and `demand_pattern` from the predictor contract.

Module 1 remains the data-preparation and exploratory-analysis layer; later modules consume its processed feature output.

## Module 2 — conditional demand distributions and scenarios

**Status:** Complete and executed successfully.

- Implemented product-mean Poisson baseline, pooled contextual Poisson GLM, and pooled contextual NB2 GLM. NB2 uses variance `mu + alpha*mu^2`, with training residual moment dispersion estimation and NB2 IRLS refitting.
- Used the existing chronological split; dropped only rows without complete lag inputs. No predictors use `generated_mean_demand` or `demand_pattern`.
- Validation NLL selected the contextual NB2 distribution. Test metrics (held-out) for NB2: MAE 19.469, RMSE 28.692, NLL 4.947, 90% interval coverage 0.852, mean width 84.421, CRPS 14.571. Coverage below nominal is a calibration limitation.
- Executed baseline/contextual Poisson/NB2 comparisons on validation and test. The selected model is a pooled product-effect model.
- Generated 100 reproducible 30-day scenarios for 50 products (150,000 rows), seed 42. Test confirmed integer nonnegative demand and per-product/date probability mass of 1.
- Scenario future promotions, discounts, and festivals are set to zero; cross-product dependence is not modelled. Evaluation uses one chronological holdout rather than rolling-origin CV.
- Focused scenario test passed: expected dimensions, reproducibility, nonnegative integer demand, and normalized probabilities.

Model assumptions, mathematical distributions, feature inputs, evaluation method, scenario method, and limitations are documented in `outputs/evaluations/README.md`.

## Module 4 — inventory simulation and evaluation

**Status:** Implemented and run successfully on the chronological held-out test set.

- Compared context-aware and non-contextual Module 3 policies plus a trailing-28-day mean order-up-to baseline. Each policy places one order decision per rolling 30-day window; the two stochastic policies are re-optimized at each window start using information through the prior day. This is not a daily dynamic reorder model.
- Backtest dates: 2025-09-13 through 2025-12-31 (110 days, 50 products). Policies used identical realized test demand, initial stocks at the start, product costs, and common sampled supplier lead-time paths.
- Contextual optimizer: cost INR 43,232,958.43; fill rate 45.08%; lost sales 156,797; stockout frequency 45.67% of product-days.
- Non-contextual optimizer: cost INR 43,199,584.79; fill rate 46.77%; lost sales 151,975; stockout frequency 43.31%. Contextual cost was 0.077% higher on this holdout, so the experiment does not show a cost improvement over non-contextual optimization.
- Historical-mean order-up-to: cost INR 43,770,668.47; fill rate 47.68%; lost sales 149,370; stockout frequency 40.96%. It had the best service and highest cost of the three.
- All 110 calendar days had at least one product stockout. Average on-hand inventory: 154.22, 175.49, and 198.98 units for context, non-context, and historical baseline respectively. Optimizer time across rolling windows was 25.34 s contextual and 22.42 s non-contextual; the conventional policy uses no solver.
- Frozen-order sensitivity results are saved for 1.5x lead-time standard deviation and 1.5x holding/shortage costs. These are one seeded stress path and coefficient re-costing, not repeated-sample inference. No orders were outstanding at test end in the primary run.
- Supplier lead time sampled from the provided average/std; a reliability failure adds one average lead-time retry cycle. Common lead-time draws were applied across policies. Forecast scenario IDs are recursive time paths within products but assume conditional independence across products. Only realized test demand is used for primary scoring.
- Frozen-order sensitivity replays tested 1.5x lead-time standard deviation and 1.5x holding/shortage cost. These do not re-optimize decisions; demand-uncertainty sensitivity was not run.
- Focused simulation tests passed for balance, arrival timing, lost sales, costs, fill rate, lead-time reliability/reproducibility, and common conditions.

Outputs are saved under `outputs/simulation/`. See `outputs/simulation/evaluation_methodology.md` for leakage controls and limitations.

## Module 3 — stochastic inventory optimization

**Status:** Implemented and run successfully for both demand scenario sets.

- Formulated as a mixed-integer two-stage stochastic program: one integer order decision per product before demand; scenario-specific inventory/lost-sales state recourse linked by inventory balance. No later order decisions are allowed, so no clairvoyant replenishment decisions can occur.
- Used the existing contextual 100-scenario, 30-day Module 2 output. The non-contextual case samples from the saved Module 2 per-product Poisson mean baseline. Both cases use the same objective, initial inventory, deterministic average lead times, per-product order capacity, holding cost, fixed/unit ordering costs, and selling-price lost-sales proxy.
- Actual final run (no budget supplied because none is present in the source data): contextual objective INR 10,286,658.43 (purchase INR 3,833,673; fixed orders INR 5,617; expected holding INR 128,788.47; expected shortage INR 6,318,579.96; solver optimal, 39.50 s). Non-contextual objective INR 10,902,642.93 (purchase INR 4,065,363; fixed INR 5,617; expected holding INR 133,735.62; expected shortage INR 6,697,927.31; solver optimal, 26.65 s).
- The source capacity field differs by product even for a shared supplier ID. It is enforced as per-product order capacity; no unsupported supplier-wide cap is inferred. Optional explicit shared supplier caps and shared purchase budget are supported.
- Supplier average lead time is rounded up and applied as a deterministic day-0 order arrival. Lead-time variability and supplier reliability are not optimized; these remain for Module 4 simulation.
- Focused tests passed: hand-verifiable single product (order 5, expected cost INR 5.25), inventory balance/nonnegative recourse, budget, supplier capacity, and probability validation.
- Outputs are in `outputs/optimization/`. Mathematical definitions and timing assumptions are in `MATHEMATICAL_FORMULATION.md`.

## Module 5 — robustness, final evaluation, and dashboard

**Status:** Implemented and run successfully (3 paired repetitions; 2 forecast scenarios per 30-day review; demand seed 5500, lead-time seed 6600).

- Reused Modules 1–4 data, models, scenario generator, optimizer, and daily simulator. Repeated paths use weekday-aligned 30-day moving blocks from pre-test demand; each policy shares realized demand and lead-time paths within a replication. Historical features at each decision date stop at the prior day. Known pipeline receipts are carried between review windows.
- Compared contextual and non-contextual optimization with the trailing-mean order-up-to baseline. Mean costs were INR 45,486,598; INR 45,424,069; and INR 45,910,765, respectively. Paired contextual cost savings versus non-contextual averaged -0.135% (95% CI -0.469% to 0.199%); contextual cost was lower in 0/3 repetitions. Mean fill rates were 43.62%, 44.89%, and 45.92%. This small experiment is descriptive, with no general winner established.
- Ran frozen-order stress replays for 1.25x demand deviations, 1.5x lead-time standard deviation, holding cost, and shortage penalty. These quantify exposure without re-optimizing under changed assumptions. Results and cost breakdowns are saved in `outputs/robustness/`.
- Wrote the final report and three figures to `outputs/reports/`; implemented a saved-results Streamlit dashboard in `app/`. It does not train models or run optimization at startup.
- Added optional scheduled-receipt support to the optimizer/simulator input path for repeated rolling reviews; the default preserves prior Module 4 behavior. This ensures decisions account for already ordered units without observing future demand.
- Focused tests passed for repeated-path/policy pairing, common conditions, nonanticipative review history, stress outputs, dashboard loading, and scheduled receipt handling. Module 2–4 regression tests also pass.

Limitations: only three paired replications were run; the demand paths are bootstrap samples from synthetic pre-test data. Sensitivity replays freeze base orders. Lead-time reliability remains a simplified delayed retry; supplier capacity is per product, and no shared budget or aggregate capacity is present. The dashboard displays saved outputs only. See `outputs/reports/final_research_report.md` for methodology and detailed results.
