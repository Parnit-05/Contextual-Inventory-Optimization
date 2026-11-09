import unittest
import numpy as np
import pandas as pd
from src.inventory_optimization.scenarios import generate_scenarios

class DummyModel:
    alpha_ = 0.0
    def predict_mean(self, frame): return np.full(len(frame), 4.0)

class ScenarioTests(unittest.TestCase):
    def test_shape_types_probabilities_and_reproducibility(self):
        dates = pd.date_range("2024-01-01", periods=30)
        history = pd.DataFrame({"date": list(dates)*2, "product_id": ["A"]*30+["B"]*30,
            "demand_units": [3]*60, "category": ["x"]*60, "supplier_id": ["s"]*60,
            "supplier_average_lead_time_days": [2]*60, "supplier_lead_time_std_days": [0.2]*60,
            "supplier_reliability": [0.9]*60, "supplier_max_capacity_units": [100]*60,
            "baseline_daily_demand_units": [3]*60, "unit_cost_inr": [1]*60, "selling_price_inr": [2]*60,
            "holding_cost_per_unit_per_day_inr": [0.1]*60, "ordering_cost_per_order_inr": [1]*60,
            "synthetic_temperature_c": [20]*60, "demand_lag_1": [3]*60, "demand_lag_7": [3]*60,
            "demand_lag_14": [3]*60, "demand_lag_28": [3]*60,
            "demand_rolling_mean_7": [3]*60, "demand_rolling_std_7": [0]*60,
            "demand_rolling_mean_28": [3]*60, "demand_rolling_std_28": [0]*60,
            "weekend_indicator": [0]*60, "festival_indicator": [0]*60, "promotion_indicator": [0]*60,
            "discount_fraction": [0]*60, "promotion_discount_interaction": [0]*60})
        a = generate_scenarios(DummyModel(), history, horizon=4, n_scenarios=5, seed=9)
        b = generate_scenarios(DummyModel(), history, horizon=4, n_scenarios=5, seed=9)
        self.assertEqual(a.shape, (40, 5)); self.assertTrue(a.equals(b))
        self.assertTrue(np.issubdtype(a.demand_units.dtype, np.integer)); self.assertTrue((a.demand_units >= 0).all())
        self.assertTrue(np.allclose(a.probability, .2))
        self.assertTrue(np.allclose(a.groupby(["date", "product_id"]).probability.sum(), 1.0))

if __name__ == "__main__": unittest.main()
