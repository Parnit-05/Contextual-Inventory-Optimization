import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from app.data_loader import load_csv, load_dashboard_data, load_json
from src.inventory_optimization.module5 import moving_block_demand, mean_confidence_interval, summarize_repetitions, paired_policy_comparison
from src.inventory_optimization.simulation import simulate_daily
from src.inventory_optimization.optimization import InventoryInputs, solve_inventory

class Module5Tests(unittest.TestCase):
    def test_bootstrap_reproducibility_and_seed_variation(self):
        dates=pd.date_range("2025-01-01",periods=70)
        history=pd.DataFrame({"date":np.repeat(dates,2),"product_id":["A","B"]*len(dates),
            "demand_units":np.repeat(np.arange(len(dates)),2)+np.tile([0,100],len(dates))})
        target=pd.date_range("2025-04-01",periods=21)
        a=moving_block_demand(history,target,["A","B"],seed=99,block_days=7)
        b=moving_block_demand(history,target,["A","B"],seed=99,block_days=7)
        c=moving_block_demand(history,target,["A","B"],seed=100,block_days=7)
        self.assertTrue(np.array_equal(a,b)); self.assertFalse(np.array_equal(a,c))

    def test_mean_confidence_interval_and_paired_aggregation(self):
        mean,lo,hi=mean_confidence_interval([1,2,3])
        self.assertAlmostEqual(mean,2); self.assertLess(lo,mean); self.assertGreater(hi,mean)
        rows=[]
        for rep,(ctx,non,fill) in enumerate([(90,100,.8),(110,100,.7),(100,100,.75)]):
            for policy,cost,rate in [("context_aware_optimization",ctx,fill),("non_contextual_optimization",non,.75),
                ("historical_mean_order_up_to_30d",120,.85)]:
                rows.append({"replication":rep,"policy":policy,"total_inventory_cost_inr":cost,"fill_rate":rate,"total_lost_sales_units":1,"stockout_frequency":.1,"average_on_hand_inventory_units":2,"average_inventory_value_inr":3})
        frame=pd.DataFrame(rows); agg=summarize_repetitions(frame); paired=paired_policy_comparison(frame)
        self.assertEqual(len(agg),18); self.assertEqual(len(paired),4)
        saving=paired.loc[paired.comparison=="context_cost_saving_pct","mean_paired_difference"].iloc[0]
        self.assertAlmostEqual(saving,0)

    def test_pipeline_is_visible_at_review_and_received_on_delivery_day(self):
        products=pd.DataFrame({"product_id":["P1"],"supplier_id":["S1"],"category":["Dairy"],
            "initial_inventory_units":[0.],"unit_cost_inr":[1.],"ordering_cost_per_order_inr":[0.],
            "holding_cost_per_unit_per_day_inr":[0.],"selling_price_inr":[10.],
            "supplier_average_lead_time_days":[3.],"supplier_lead_time_std_days":[0.],"supplier_reliability":[1.]})
        seen={}
        def choose(window,day,stock,date,horizon,pipeline):
            if window==1: seen["pipeline"]=pipeline.copy()
            return np.array([2 if window==0 else 0])
        daily,orders,metrics,q=simulate_daily(pd.date_range("2025-01-01",periods=4),np.zeros((4,1)),products,
            {0:np.zeros(1,dtype=int),1:np.zeros(1,dtype=int)},np.array([[3],[0]]),[0,2],decision_callback=choose)
        self.assertEqual(seen["pipeline"][1,0],2)
        self.assertEqual(daily.received_units.tolist(),[0.,0.,0.,2.])
        self.assertEqual(metrics["outstanding_pipeline_units_at_test_end"],0)

    def test_optimizer_accounts_for_known_pipeline_receipt(self):
        data=InventoryInputs(products=["P1"],demands=np.array([[[0.],[5.]]]),probabilities=np.array([1.]),
            initial_inventory=np.array([0.]),unit_cost=np.array([1.]),fixed_order_cost=np.array([0.]),
            holding_cost=np.array([0.]),shortage_penalty=np.array([10.]),lead_time_days=np.array([0.]),
            supplier_ids=["S1"],product_capacity=np.array([10.]),supplier_capacity={},budget=None,
            scheduled_receipts=np.array([[0.],[5.]]))
        result=solve_inventory(data)
        self.assertEqual(result["order_quantity"].tolist(),[0])

    def test_dashboard_loaders_handle_missing_and_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"outputs").mkdir(); (root/"outputs"/"x.csv").write_text("a\n1\n",encoding="utf-8")
            (root/"outputs"/"x.json").write_text('{"ok": true}',encoding="utf-8")
            self.assertEqual(load_csv("outputs/x.csv",root).iloc[0,0],1)
            self.assertIsNone(load_csv("missing.csv",root)); self.assertEqual(load_json("outputs/x.json",root),{"ok":True})
            self.assertIsNone(load_dashboard_data(root)["repeated_summary"])

if __name__=="__main__": unittest.main()
