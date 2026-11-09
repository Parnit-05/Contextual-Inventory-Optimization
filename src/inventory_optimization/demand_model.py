"""Pooled Poisson log-linear demand model and empirical product baseline."""
import math
import pickle
import numpy as np

EXCLUDED = {"demand_units", "generated_mean_demand", "demand_pattern", "split"}
NUMERIC_FEATURES = [
    "demand_lag_1", "demand_lag_7", "demand_lag_14", "demand_lag_28",
    "demand_rolling_mean_7", "demand_rolling_std_7", "demand_rolling_mean_28", "demand_rolling_std_28",
    "weekend_indicator", "festival_indicator", "promotion_indicator", "discount_fraction",
    "promotion_discount_interaction", "synthetic_temperature_c", "unit_cost_inr", "selling_price_inr",
    "holding_cost_per_unit_per_day_inr", "ordering_cost_per_order_inr", "baseline_daily_demand_units",
    "supplier_average_lead_time_days", "supplier_lead_time_std_days", "supplier_reliability", "supplier_max_capacity_units",
]
CATEGORICAL_FEATURES = ["product_id", "category", "day_of_week", "month", "season"]

def _design_frame(frame, state=None):
    """Build deterministic numeric design matrix; no target/generator metadata accepted."""
    if any(c in NUMERIC_FEATURES + CATEGORICAL_FEATURES for c in EXCLUDED):
        raise AssertionError("Forbidden target or metadata in feature contract")
    if state is None:
        levels = {c: sorted(frame[c].fillna("__missing__").astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES}
        numeric = {c: (float(frame[c].mean()), max(float(frame[c].std()), 1e-8)) for c in NUMERIC_FEATURES}
        state = {"levels": levels, "numeric": numeric}
    cols = [np.ones(len(frame), dtype=float)]
    names = ["intercept"]
    for c in NUMERIC_FEATURES:
        mean, scale = state["numeric"][c]
        vals = frame[c].astype(float).fillna(mean).to_numpy()
        cols.append((vals - mean) / scale); names.append(c)
    for c in CATEGORICAL_FEATURES:
        vals = frame[c].fillna("__missing__").astype(str)
        # Drop first level for an identifiable reference category.
        for level in state["levels"][c][1:]:
            cols.append((vals.to_numpy() == level).astype(float)); names.append(f"{c}={level}")
    return np.column_stack(cols), state, names

class ProductMeanBaseline:
    """Non-contextual Poisson distribution with training mean for each product."""
    def fit(self, frame):
        means = frame.groupby("product_id").demand_units.mean()
        self.means_ = {str(k): max(float(v), 1e-8) for k, v in means.items()}
        self.global_mean_ = max(float(frame.demand_units.mean()), 1e-8)
        return self

    def predict_mean(self, frame):
        return np.array([self.means_.get(str(p), self.global_mean_) for p in frame.product_id], dtype=float)

    def save(self, path):
        with open(path, "wb") as f: pickle.dump(self, f)

class PooledPoisson:
    """Ridge-regularized Poisson GLM fitted with IRLS using NumPy only."""
    def __init__(self, ridge=1e-4, max_iter=60, tolerance=1e-7):
        self.ridge, self.max_iter, self.tolerance = ridge, max_iter, tolerance

    def fit(self, frame):
        X, self.encoder_, self.column_names_ = _design_frame(frame)
        y = frame.demand_units.to_numpy(dtype=float)
        beta = np.zeros(X.shape[1]); beta[0] = math.log(max(y.mean(), 1e-6))
        penalty = np.eye(X.shape[1]) * self.ridge; penalty[0, 0] = 0
        for _ in range(self.max_iter):
            eta = np.clip(X @ beta, -15, 15); mu = np.exp(eta)
            z = eta + (y - mu) / np.maximum(mu, 1e-10)
            lhs = X.T @ (mu[:, None] * X) + penalty
            rhs = X.T @ (mu * z)
            try: next_beta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError: next_beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            if np.max(np.abs(next_beta - beta)) < self.tolerance:
                beta = next_beta; break
            beta = next_beta
        self.coef_ = beta
        mu = self.predict_mean(frame)
        numerator = np.sum((y - mu) ** 2 - mu)
        denominator = max(np.sum(mu ** 2), 1e-8)
        self.alpha_ = float(np.clip(numerator / denominator, 0.0, 10.0))
        return self

    def predict_mean(self, frame):
        X, _, _ = _design_frame(frame, self.encoder_)
        return np.exp(np.clip(X @ self.coef_, -15, 15))

    def save(self, path):
        with open(path, "wb") as f: pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f: return pickle.load(f)

class PooledNegativeBinomial(PooledPoisson):
    """Pooled NB2 log-link regression; variance is mu + alpha*mu^2."""
    def fit(self, frame):
        # Initialize coefficients with Poisson IRLS, estimate NB2 dispersion from
        # training residuals, then refit with the NB2 variance weights.
        super().fit(frame)
        X, _, _ = _design_frame(frame, self.encoder_)
        y = frame.demand_units.to_numpy(dtype=float)
        alpha = self.alpha_
        if alpha > 1e-10:
            beta = self.coef_.copy()
            penalty = np.eye(X.shape[1]) * self.ridge; penalty[0, 0] = 0
            for _ in range(self.max_iter):
                eta = np.clip(X @ beta, -15, 15); mu = np.exp(eta)
                weight = mu / (1.0 + alpha * mu)
                z = eta + (y - mu) / np.maximum(mu, 1e-10)
                lhs = X.T @ (weight[:, None] * X) + penalty
                rhs = X.T @ (weight * z)
                try: new_beta = np.linalg.solve(lhs, rhs)
                except np.linalg.LinAlgError: new_beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
                if np.max(np.abs(new_beta - beta)) < self.tolerance:
                    beta = new_beta; break
                beta = new_beta
            self.coef_ = beta
            self.alpha_ = alpha
        else:
            self.alpha_ = 0.0
        return self
