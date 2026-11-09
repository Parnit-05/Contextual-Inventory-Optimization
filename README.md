# Context-Aware Stochastic Inventory Optimization

## Module 1: data preparation and exploratory analysis

Run from the repository root with `python -m src.inventory_optimization.pipeline` after installing `requirements.txt`. The pipeline reads the existing CSV under `data/raw/`, validates it, creates leakage-aware contextual features, generates summaries and figures, and writes outputs under `data/processed/` and `outputs/`.

Demand lags and rolling statistics use only earlier rows within each product; rolling windows shift demand by one observation first. The chronological split is based on unique dates (70% train, 15% validation, 15% test). `demand_units` is the target. `generated_mean_demand` and `demand_pattern` are excluded from predictors.

## Module 2: conditional demand and scenarios

Run `python -m src.inventory_optimization.module2 --horizon 30 --scenarios 100 --seed 42` after Module 1. It compares a product-mean Poisson baseline, pooled contextual Poisson regression, and pooled NB2 regression; validation NLL selects the contextual distribution. It saves pickled models, validation/test metrics, test predictions with 90% intervals, a validation NLL chart, and equally weighted recursive demand scenarios. See [Module 2 evaluation notes](outputs/evaluations/README.md) for model assumptions, formulas, feature contract, evaluation, and limitations.

Scenario rows use `scenario_id`, `date`, `product_id`, `demand_units`, and `probability`, with probability mass summing to one for each product-date. Forecast-time promotions, discounts, and festivals default to zero because their future values are not supplied. Modules 4–5 evaluate delivery uncertainty and saved model outputs; the dashboard is described below.

## Module 3: stochastic inventory optimization

Read [MATHEMATICAL_FORMULATION.md](MATHEMATICAL_FORMULATION.md) for the objective, decisions, recourse, balance, capacity, timing, and assumptions. Run `python -m src.inventory_optimization.module3` after Module 2. It solves comparable contextual and product-mean Poisson scenarios using a day-0 SKU order, deterministic average lead times, and scenario-based lost-sales recourse. An optional shared purchase budget is supplied with `--budget-inr`; additional supplier-wide capacity limits can be passed as repeated `--shared-supplier-capacity SUPPLIER=UNITS`. Without those inputs, only the per-product capacities from the dataset apply.

Module 3 writes decisions, expected daily inventory/shortage, objective breakdown, solve status/time, and cross-evaluated policies to `outputs/optimization/`. Lead-time variability and supplier reliability are not optimized, but are evaluated in simulation in Module 4.

## Module 4: inventory simulation and evaluation

Run `python -m src.inventory_optimization.module4 --scenario-count 25 --window-days 30` after Modules 1–3. It uses the held-out test period, makes/re-optimizes one order per 30-day window before that window’s demand, and compares the contextual optimizer, non-contextual optimizer, and a trailing-mean order-up-to baseline. All policies see the same realized demand and supplier lead-time draws. Lead-time variability and one delayed-retry reliability rule are simulated; they are not optimized.

Results and plots are saved under `outputs/simulation/`. [Evaluation methodology](outputs/simulation/evaluation_methodology.md) describes information timing, delivery assumptions, scenario dependence, metric definitions, and limitations. The test is one synthetic historical holdout; the reported results are not universal performance claims.

## Module 5: repeated robustness, research report and dashboard

Run a two-repetition smoke check with `python -m src.inventory_optimization.module5 --smoke-test`. Then run the repeated experiment with `python -m src.inventory_optimization.module5 --repetitions 3 --scenario-count 2`. This resamples pre-test historical demand in weekday-aligned blocks, uses common random demand and lead-time paths across policies, carries pipeline receipts across review windows, and writes results to `outputs/robustness/`. The final research report and figures are in `outputs/reports/`.

Install dependencies with `python -m pip install -r requirements.txt`. Launch the saved-results dashboard from the repository root with `python -m streamlit run app/dashboard.py`. The dashboard reads outputs and does not fit models or run optimization on startup. Missing results are shown with an informational message.
