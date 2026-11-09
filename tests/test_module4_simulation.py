import unittest
import numpy as np
import pandas as pd
from src.inventory_optimization.simulation import sample_lead_times, simulate_daily

class SimulationTests(unittest.TestCase):
    def products(self, reliability=1.0):
        return pd.DataFrame({"product_id":["P1"],"supplier_id":["S1"],"category":["Dairy"],
            "initial_inventory_units":[2.],"unit_cost_inr":[5.],"ordering_cost_per_order_inr":[7.],
            "holding_cost_per_unit_per_day_inr":[.5],"selling_price_inr":[10.],
            "supplier_average_lead_time_days":[1.],"supplier_lead_time_std_days":[0.],
            "supplier_reliability":[reliability]})

    def test_delivery_balance_lost_sales_and_costs(self):
        dates=pd.date_range("2025-01-01",periods=2)
        products=self.products()
        daily,orders,metrics,_=simulate_daily(dates,np.array([[4.],[2.]]),products,
            {0:np.array([3])},np.array([[1]]),[0],policy="test")
        self.assertEqual(daily.received_units.tolist(),[0.,3.])
        self.assertEqual(daily.lost_sales_units.tolist(),[2.,0.])
        self.assertEqual(daily.ending_inventory_units.tolist(),[0.,1.])
        self.assertEqual(metrics["total_units_demanded"],6)
        self.assertEqual(metrics["total_units_fulfilled"],4)
        self.assertAlmostEqual(metrics["fill_rate"],4/6)
        self.assertAlmostEqual(metrics["purchase_cost_inr"],15)
        self.assertAlmostEqual(metrics["ordering_cost_inr"],7)
        self.assertAlmostEqual(metrics["holding_cost_inr"],.5)
        self.assertAlmostEqual(metrics["shortage_cost_inr"],20)
        self.assertAlmostEqual(metrics["total_inventory_cost_inr"],42.5)
        self.assertEqual(metrics["outstanding_pipeline_units_at_test_end"],0)
        self.assertEqual(orders.delivery_date.iloc[0],dates[1])

    def test_common_path_and_lead_reproducibility(self):
        products=self.products(reliability=.7)
        a,_=sample_lead_times(products,3,seed=12)
        b,_=sample_lead_times(products,3,seed=12)
        self.assertTrue(np.array_equal(a,b))
        dates=pd.date_range("2025-02-01",periods=3); demand=np.array([[1.],[2.],[0.]])
        res=[]
        for policy in ("A","B"):
            daily,_,_,_=simulate_daily(dates,demand,products,{0:np.array([2])},np.array([[2]]),[0],policy=policy)
            res.append(daily)
        self.assertTrue(np.array_equal(res[0].demand_units,res[1].demand_units))
        self.assertTrue(np.array_equal(res[0].received_units,res[1].received_units))
        self.assertTrue((res[0].ending_inventory_units>=0).all())

    def test_supplier_failure_adds_retry_lead(self):
        lead,failed=sample_lead_times(self.products(reliability=0.0),1,seed=2)
        self.assertTrue(failed[0,0]); self.assertEqual(lead[0,0],2)

if __name__=="__main__": unittest.main()
