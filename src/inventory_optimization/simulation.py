"""Daily lost-sales inventory simulation and consistent policy metrics."""
import numpy as np
import pandas as pd

def sample_lead_times(products, n_windows, seed=404, std_multiplier=1.0):
    """Shared common-random-number lead realizations; failed first attempt adds one mean lead."""
    rng = np.random.default_rng(seed)
    n = len(products)
    z = rng.standard_normal((n_windows, n))
    failed = rng.random((n_windows, n)) > products.supplier_reliability.to_numpy(float)[None, :]
    avg = products.supplier_average_lead_time_days.to_numpy(float)[None, :]
    std = products.supplier_lead_time_std_days.to_numpy(float)[None, :]
    sampled = np.ceil(np.maximum(0.0, avg + std_multiplier * std * z)).astype(int)
    retry = np.ceil(avg).astype(int)
    sampled = sampled + failed.astype(int) * retry
    return sampled, failed

def inventory_metrics(daily, orders, n_products, n_dates, total_optimization_seconds=0.0):
    demand = float(daily.demand_units.sum()); fulfilled = float(daily.fulfilled_units.sum())
    lost = float(daily.lost_sales_units.sum())
    stockout_product_days = int((daily.lost_sales_units > 0).sum())
    date_stockouts = int(daily.loc[daily.lost_sales_units > 0, "date"].nunique())
    positive = orders[orders.order_quantity_units > 0]
    total_costs = {
        "purchase_cost_inr": float(daily.purchase_cost_inr.sum()),
        "ordering_cost_inr": float(daily.fixed_order_cost_inr.sum()),
        "holding_cost_inr": float(daily.holding_cost_inr.sum()),
        "shortage_cost_inr": float(daily.shortage_cost_inr.sum()),
    }
    return {
        "total_inventory_cost_inr": sum(total_costs.values()), **total_costs,
        "fill_rate": fulfilled / demand if demand else 1.0,
        "total_units_demanded": int(demand), "total_units_fulfilled": int(fulfilled), "total_lost_sales_units": int(lost),
        "stockout_product_days": stockout_product_days, "stockout_calendar_days": date_stockouts,
        "stockout_frequency": stockout_product_days / (n_products*n_dates),
        "average_on_hand_inventory_units": float(daily.ending_inventory_units.mean()),
        "average_inventory_value_inr": float(daily.groupby("date").inventory_value_inr.sum().mean()),
        "average_order_quantity_units": float(positive.order_quantity_units.mean()) if len(positive) else 0.0,
        "number_of_orders": int(len(positive)), "computational_time_seconds": float(total_optimization_seconds),
    }

def simulate_daily(test_dates, demand_matrix, products, order_schedule, lead_days, windows,
                   initial_inventory=None, holding_multiplier=1.0, shortage_multiplier=1.0,
                   demand_multiplier=1.0, policy="policy", decision_callback=None):
    """Simulate one order per review window; receipts occur at start of delivery day."""
    dates = pd.DatetimeIndex(test_dates); D = np.asarray(demand_matrix, dtype=float) * demand_multiplier
    S, N = D.shape
    if len(dates) != S or len(products) != N: raise ValueError("Demand matrix dimensions do not match dates/products")
    if np.any(D < 0): raise ValueError("Demand must be nonnegative")
    stock = (products.initial_inventory_units.to_numpy(float).copy() if initial_inventory is None
             else np.asarray(initial_inventory, dtype=float).copy())
    if len(stock) != N: raise ValueError("Initial inventory must have one value per product")
    windows = list(windows); window_at = {int(start): w for w, start in enumerate(windows)}
    arrivals = {}; daily_records = []; order_records = []
    q_by_window = {int(w): np.asarray(q, dtype=int) for w,q in order_schedule.items()}
    for day, date in enumerate(dates):
        beginning = stock.copy(); receipts = np.zeros(N); q_today = np.zeros(N, dtype=int)
        purchase_today = np.zeros(N); fixed_today = np.zeros(N); realized_lead = np.full(N, np.nan)
        # Previously outstanding orders arrive before that day's demand.
        for i, quantity in arrivals.pop(day, []):
            stock[i] += quantity; receipts[i] += quantity
        if day in window_at:
            w = window_at[day]
            if decision_callback is not None:
                next_start = windows[w+1] if w+1 < len(windows) else S
                horizon = next_start-day
                pipeline = np.zeros((horizon, N), dtype=float)
                for arrival_day, shipments in arrivals.items():
                    offset = arrival_day-day
                    if 0 <= offset < horizon:
                        for product_index, quantity in shipments: pipeline[offset, product_index] += quantity
                q_by_window[w] = np.asarray(decision_callback(w, day, stock.copy(), date, horizon, pipeline), dtype=int)
            if w not in q_by_window: raise ValueError(f"Missing order decisions for evaluation window {w}")
            q_today = q_by_window[w].copy()
            for i, quantity in enumerate(q_today):
                if quantity <= 0: continue
                avg = float(products.iloc[i].supplier_average_lead_time_days)
                delay = int(lead_days[w, i]); delivery_day = day + delay
                realized_lead[i] = delay
                purchase_today[i] = quantity * float(products.iloc[i].unit_cost_inr)
                fixed_today[i] = float(products.iloc[i].ordering_cost_per_order_inr)
                order_records.append({"policy": policy, "window": w, "order_date": date, "product_id": products.iloc[i].product_id,
                    "supplier_id": products.iloc[i].supplier_id, "order_quantity_units": int(quantity),
                    "lead_time_days": delay, "average_lead_time_days": avg,
                    "delivery_date": dates[delivery_day] if delivery_day < S else pd.NaT,
                    "delivery_within_test": delivery_day < S, "purchase_cost_inr": purchase_today[i],
                    "fixed_order_cost_inr": fixed_today[i]})
                if delivery_day < S:
                    arrivals.setdefault(delivery_day, []).append((i, int(quantity)))
                # Zero-day lead is available on order day, after the decision and before demand.
                if delivery_day == day:
                    stock[i] += quantity; receipts[i] += quantity
        fulfilled = np.minimum(stock, D[day])
        lost = D[day] - fulfilled
        stock = stock - fulfilled
        hold_cost = stock * products.holding_cost_per_unit_per_day_inr.to_numpy(float) * holding_multiplier
        shortage_cost = lost * products.selling_price_inr.to_numpy(float) * shortage_multiplier
        inv_value = stock * products.unit_cost_inr.to_numpy(float)
        for i in range(N):
            daily_records.append({"policy": policy, "date": date, "product_id": products.iloc[i].product_id,
                "supplier_id": products.iloc[i].supplier_id, "category": products.iloc[i].category, "beginning_inventory_units": beginning[i],
                "received_units": receipts[i], "order_quantity_units": q_today[i], "realized_lead_time_days": realized_lead[i],
                "demand_units": D[day,i], "fulfilled_units": fulfilled[i], "lost_sales_units": lost[i],
                "ending_inventory_units": stock[i], "purchase_cost_inr": purchase_today[i],
                "fixed_order_cost_inr": fixed_today[i], "holding_cost_inr": hold_cost[i],
                "shortage_cost_inr": shortage_cost[i], "inventory_value_inr": inv_value[i]})
    daily = pd.DataFrame(daily_records); orders = pd.DataFrame(order_records)
    if orders.empty:
        orders = pd.DataFrame(columns=["policy", "window", "order_date", "product_id", "supplier_id", "order_quantity_units", "lead_time_days", "average_lead_time_days", "delivery_date", "delivery_within_test", "purchase_cost_inr", "fixed_order_cost_inr"])
    metrics = inventory_metrics(daily, orders, N, S)
    # Derive end-of-test pipeline from the auditable order/delivery ledger.
    metrics["outstanding_pipeline_units_at_test_end"] = int(orders.loc[~orders.delivery_within_test, "order_quantity_units"].sum())
    # Daily stock conservation, including due receipts, is checked for every product.
    balance = daily.beginning_inventory_units + daily.received_units - daily.fulfilled_units - daily.ending_inventory_units
    if not np.allclose(balance, 0.0, atol=1e-8): raise AssertionError("Daily inventory balance failed")
    return daily, orders, metrics, q_by_window
