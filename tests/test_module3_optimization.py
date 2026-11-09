import unittest
import numpy as np
from src.inventory_optimization.optimization import InventoryInputs, solve_inventory, evaluate_fixed_policy

def inputs(demand, **overrides):
    base = dict(products=["P1"], demands=np.asarray(demand, dtype=float), probabilities=np.full(len(demand), 1/len(demand)),
        initial_inventory=np.array([0.]), unit_cost=np.array([1.]), fixed_order_cost=np.array([0.]),
        holding_cost=np.array([0.1]), shortage_penalty=np.array([10.]), lead_time_days=np.array([0.]),
        supplier_ids=["S1"], product_capacity=np.array([20.]), supplier_capacity={}, budget=None)
    base.update(overrides)
    return InventoryInputs(**base)

class InventoryOptimizationTests(unittest.TestCase):
    def test_hand_verifiable_two_period_order_and_balance(self):
        demand = np.array([[[5.],[0.]], [[0.],[5.]]])
        data = inputs(demand)
        result = solve_inventory(data)
        self.assertEqual(result["status"], "optimal")
        self.assertEqual(result["order_quantity"].tolist(), [5])
        self.assertAlmostEqual(result["objective_value"], 5.25, places=4)
        self.assertTrue((result["ending_inventory"] >= -1e-8).all())
        self.assertTrue((result["lost_sales"] >= -1e-8).all())
        manual = evaluate_fixed_policy(data, np.array([5]))
        self.assertAlmostEqual(manual["expected_total_cost"], 5.25, places=8)

    def test_budget_and_shared_supplier_capacity(self):
        demand = np.full((2, 1, 2), 10.)
        data = InventoryInputs(products=["A","B"], demands=demand, probabilities=np.array([.5,.5]),
            initial_inventory=np.zeros(2), unit_cost=np.ones(2), fixed_order_cost=np.zeros(2),
            holding_cost=np.zeros(2), shortage_penalty=np.full(2,10.), lead_time_days=np.zeros(2),
            supplier_ids=["S1","S1"], product_capacity=np.array([20.,20.]), supplier_capacity={"S1":5.}, budget=3.)
        result = solve_inventory(data)
        self.assertLessEqual(result["order_quantity"].sum(), 3)
        self.assertLessEqual(result["order_quantity"].sum(), 5)
        self.assertEqual(result["status"], "optimal")

    def test_probability_validation(self):
        data = inputs(np.array([[[1.]], [[2.]]]))
        data.probabilities = np.array([.4,.4])
        with self.assertRaises(ValueError): solve_inventory(data)

if __name__ == "__main__": unittest.main()
