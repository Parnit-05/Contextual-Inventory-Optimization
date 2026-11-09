# Module 2: conditional demand distributions

## Models

- **Product mean baseline:** per-product training mean, modelled as Poisson. It ignores contextual inputs and supplies a statistical reference.
- **Contextual Poisson:** pooled, ridge-regularized Poisson log-link regression fitted by IRLS. Product, category, day, month, and season use reference-coded indicators; continuous demand history, promotion, price, supplier, and temperature fields are standardized using training rows.
- **Contextual NB2:** pooled log-link regression with variance `mu + alpha * mu^2`. Dispersion `alpha` is estimated by training residual moments from the Poisson fit; coefficients are refitted with NB2 IRLS weights. Validation negative log-likelihood selects Poisson or NB2.

The response is a nonnegative count. For Poisson, `P(Y=y)=exp(-mu) mu^y/y!`. For NB2, size `r=1/alpha`, probability `p=r/(r+mu)`, and `Var(Y)=mu+alpha*mu^2`; alpha zero reduces to Poisson. This is a pooled model with product indicator effects, allowing products to share coefficients while retaining product-specific levels.

## Inputs and leakage controls

Inputs are the Module 1 historical demand lags (1, 7, 14, 28), shifted rolling means and standard deviations (7, 28), weekend/festival/promotion flags, discount, synthetic temperature, prices/costs, baseline demand, supplier lead-time/reliability/capacity, plus product/category and calendar indicator fields. Target, `generated_mean_demand`, and `demand_pattern` are excluded explicitly.

The existing Module 1 chronological train/validation/test labels are used unchanged. Rows without all four lags are dropped. Each evaluated row uses lag fields built only from observations before its date. Validation chooses the contextual distribution by NLL; test metrics are reported after selection. Evaluation includes MAE, RMSE, discrete NLL, empirical 90% interval coverage/width, and discrete CRPS.

## Scenario generation

By default, `python -m src.inventory_optimization.module2` generates 100 scenarios over 30 days with seed 42. Each scenario is simulated forward per product: daily conditional counts update that scenario's own lag and rolling history. NB2 samples use a gamma-Poisson mixture; Poisson samples use direct Poisson draws. Scenario probability is `1 / scenario_count`; each product-date probability mass sums to one. Future calendar fields are derived from dates. Unknown future promotion, discount, and festival inputs are set to zero.

The long scenario table has `scenario_id`, `date`, `product_id`, integer `demand_units`, and `probability`, ready to join to Module 3 product cost/capacity parameters.

## Limitations

The generated source data is synthetic; temperature and baseline demand are synthetic context. The pooled linear log-mean cannot capture all nonlinearities or cross-product dependencies. NB2 dispersion is a practical method-of-moments estimate, not a full joint maximum-likelihood fit. Interval coverage is below nominal 90% in the held-out test set. Multi-step scenarios assume conditional independence across products and rely on zero promotion/festival assumptions for unknown future context. The supplied chronological evaluation is one fixed holdout, not rolling cross-validation.
