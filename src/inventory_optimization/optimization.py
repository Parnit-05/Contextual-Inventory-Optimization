"""Two-stage scenario inventory model with one pre-horizon order and state recourse."""
from dataclasses import dataclass
import time
import numpy as np
import cvxpy as cp

@dataclass
class InventoryInputs:
    products: list
    demands: np.ndarray          # scenario x period x product
    probabilities: np.ndarray   # scenario
    initial_inventory: np.ndarray
    unit_cost: np.ndarray
    fixed_order_cost: np.ndarray
    holding_cost: np.ndarray
    shortage_penalty: np.ndarray
    lead_time_days: np.ndarray
    supplier_ids: list
    product_capacity: np.ndarray
    supplier_capacity: dict | None = None  # optional additional aggregate limits
    budget: float | None = None
    scheduled_receipts: np.ndarray | None = None  # known future arrivals, period x product

def validate_inputs(data):
    D = np.asarray(data.demands)
    p = np.asarray(data.probabilities, dtype=float)
    n = len(data.products)
    if D.ndim != 3 or D.shape[2] != n or D.shape[0] != len(p): raise ValueError("demands must have shape (scenarios, periods, products)")
    if D.shape[0] == 0 or D.shape[1] == 0 or n == 0: raise ValueError("scenario, period, and product sets must be nonempty")
    if np.any(D < 0) or not np.all(np.isfinite(D)): raise ValueError("demand must be finite and nonnegative")
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0): raise ValueError("scenario probabilities must be nonnegative and sum to one")
    for field in ("initial_inventory", "unit_cost", "fixed_order_cost", "holding_cost", "shortage_penalty", "lead_time_days", "product_capacity"):
        values = np.asarray(getattr(data, field), dtype=float)
        if len(values) != n or not np.all(np.isfinite(values)): raise ValueError(f"{field} must have one finite value per product")
    if min(np.min(data.initial_inventory), np.min(data.unit_cost), np.min(data.fixed_order_cost), np.min(data.holding_cost), np.min(data.shortage_penalty), np.min(data.lead_time_days)) < 0 or np.min(data.product_capacity) <= 0:
        raise ValueError("inventory, costs, and lead times must be nonnegative")
    if len(data.supplier_ids) != n: raise ValueError("supplier_ids must have one value per product")
    if data.budget is not None and data.budget < 0: raise ValueError("budget must be nonnegative")
    if data.scheduled_receipts is not None:
        receipts = np.asarray(data.scheduled_receipts, dtype=float)
        if receipts.shape != (D.shape[1], n) or not np.all(np.isfinite(receipts)) or np.any(receipts < 0):
            raise ValueError("scheduled_receipts must be a nonnegative (periods, products) matrix")

def solve_inventory(data, solver="SCIPY", time_limit=None):
    """Solve and return decisions, recourse, expected cost breakdown, status and time."""
    validate_inputs(data)
    D = np.asarray(data.demands, dtype=float); p = np.asarray(data.probabilities, dtype=float)
    S, H, N = D.shape
    scheduled = np.zeros((H, N)) if data.scheduled_receipts is None else np.asarray(data.scheduled_receipts, dtype=float)
    q = cp.Variable(N, integer=True, name="order_quantity")
    z = cp.Variable(N, boolean=True, name="order_placed")
    inv = cp.Variable((S, H, N), nonneg=True, name="ending_inventory")
    lost = cp.Variable((S, H, N), nonneg=True, name="lost_sales")
    constraints = [q >= 0]
    lead_period = np.maximum(0, np.ceil(np.asarray(data.lead_time_days)).astype(int))
    # day index 1 is the first demand day; lead L arrives at start of day L+1.
    arrival_t = np.maximum(0, lead_period)  # zero-based period index
    for i in range(N):
        cap = float(data.product_capacity[i])
        constraints.append(q[i] <= cap * z[i])
    for supplier, cap in (data.supplier_capacity or {}).items():
        ix = [i for i, g in enumerate(data.supplier_ids) if g == supplier]
        if not ix: raise ValueError(f"Shared capacity supplied for unknown supplier {supplier}")
        constraints.append(cp.sum(q[ix]) <= float(cap))
    if data.budget is not None:
        constraints.append(cp.sum(cp.multiply(np.asarray(data.unit_cost), q)) <= float(data.budget))
    for t in range(H):
        prior = np.asarray(data.initial_inventory) if t == 0 else inv[:, t-1, :]
        receipt = scheduled[t, :] + cp.hstack([q[i] if arrival_t[i] == t else 0.0 for i in range(N)])
        constraints.append(inv[:, t, :] == prior + receipt - D[:, t, :] + lost[:, t, :])
    fixed_order = cp.sum(cp.multiply(np.asarray(data.fixed_order_cost), z))
    purchase = cp.sum(cp.multiply(np.asarray(data.unit_cost), q))
    prob_weight = p.reshape(S, 1, 1)
    holding_expr = cp.sum(cp.multiply(inv, prob_weight * np.asarray(data.holding_cost).reshape(1, 1, N)))
    shortage_expr = cp.sum(cp.multiply(lost, prob_weight * np.asarray(data.shortage_penalty).reshape(1, 1, N)))
    problem = cp.Problem(cp.Minimize(purchase + fixed_order + holding_expr + shortage_expr), constraints)
    options = {}
    if time_limit is not None: options["scipy_options"] = {"time_limit": float(time_limit)}
    start = time.perf_counter()
    problem.solve(solver=solver, **options)
    elapsed = time.perf_counter() - start
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Inventory optimization did not solve to an acceptable status: {problem.status}")
    qv = np.rint(q.value).astype(int); zv = np.rint(z.value).astype(int)
    iv = np.maximum(inv.value, 0); lv = np.maximum(lost.value, 0)
    costs = {
        "purchase_cost": float(np.dot(data.unit_cost, qv)),
        "fixed_order_cost": float(np.dot(data.fixed_order_cost, zv)),
        "expected_holding_cost": float(np.sum(p[:, None, None] * iv * np.asarray(data.holding_cost)[None, None, :])),
        "expected_shortage_cost": float(np.sum(p[:, None, None] * lv * np.asarray(data.shortage_penalty)[None, None, :])),
    }
    costs["expected_total_cost"] = sum(costs.values())
    if not np.isclose(costs["expected_total_cost"], problem.value, rtol=1e-4, atol=1e-3):
        raise AssertionError(f"Independent cost calculation {costs['expected_total_cost']} does not match solver objective {problem.value}")
    # Verify conservation and operational limits from the returned solution.
    for t in range(H):
        prior = np.asarray(data.initial_inventory) if t == 0 else iv[:, t-1, :]
        receipt = np.zeros(N)
        for i in range(N):
            if arrival_t[i] == t: receipt[i] = qv[i]
        receipt += scheduled[t, :]
        residual = iv[:, t, :] - (prior + receipt - D[:, t, :] + lv[:, t, :])
        if np.max(np.abs(residual)) > 1e-3: raise AssertionError("Inventory balance check failed")
    if data.budget is not None and np.dot(data.unit_cost, qv) > data.budget + 1e-4: raise AssertionError("Budget feasibility check failed")
    for supplier, cap in (data.supplier_capacity or {}).items():
        if sum(qv[i] for i,g in enumerate(data.supplier_ids) if g == supplier) > cap + 1e-4: raise AssertionError("Shared supplier capacity check failed")
    return {"status": problem.status, "solver": solver, "solve_seconds": elapsed, "objective_value": float(problem.value),
            "order_quantity": qv, "order_placed": zv, "ending_inventory": iv, "lost_sales": lv, "costs": costs,
            "arrival_period": arrival_t + 1}

def evaluate_fixed_policy(data, order_quantity):
    """Evaluate a fixed first-stage order on a scenario set without re-optimizing."""
    D = np.asarray(data.demands, dtype=float); p = np.asarray(data.probabilities)
    S,H,N = D.shape; q = np.asarray(order_quantity, dtype=float)
    lead = np.ceil(np.asarray(data.lead_time_days)).astype(int)
    inv = np.zeros_like(D); lost = np.zeros_like(D)
    for s in range(S):
        stock = np.asarray(data.initial_inventory, dtype=float).copy()
        for t in range(H):
            for i in range(N):
                if max(0, lead[i]) == t: stock[i] += q[i]
                available = stock[i]
                lost[s,t,i] = max(0.0, D[s,t,i] - available)
                stock[i] = max(0.0, available - D[s,t,i])
                inv[s,t,i] = stock[i]
    order_cost = float(np.dot(data.unit_cost, q) + np.dot(data.fixed_order_cost, (q > 0).astype(int)))
    holding = float(np.sum(p[:,None,None] * inv * np.asarray(data.holding_cost)[None,None,:]))
    shortage = float(np.sum(p[:,None,None] * lost * np.asarray(data.shortage_penalty)[None,None,:]))
    return {"purchase_and_order_cost": order_cost, "expected_holding_cost": holding, "expected_shortage_cost": shortage,
            "expected_total_cost": order_cost+holding+shortage, "expected_total_ending_inventory_units": float((p[:,None,None]*inv[:,-1:,:]).sum()),
            "expected_shortage_units": float(np.sum(p[:,None,None]*lost))}
