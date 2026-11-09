# Module 3: stochastic inventory optimization

## Sets and indices

- Products `i in I` (single-product model: `|I|=1`).
- Demand scenarios `s in S`, with probability `p_s >= 0` and `sum_s p_s = 1`.
- Daily periods `t in T={1,...,H}`.
- Suppliers `g in G`; each product is assigned to one supplier `g(i)`.

## Parameters

- `D[s,t,i]`: demand scenario, product, and day.
- `p_s`: scenario probability. Module 2 scenarios use equal weights.
- `I0[i]`: initial on-hand inventory.
- `L[i]`: deterministic delivery lead time, set to `ceil(supplier_average_lead_time_days)` days.
- `a[t,i]`: 1 if the day-0 order is available at the start of period `t`, else 0. With day 1 the first demand period, `a[t,i]=1` at `t=max(1,L[i]+1)` only. If that delivery falls after the horizon, no arrival occurs during the horizon.
- `P[t,i]`: known outstanding pipeline units due at the start of period `t` (zero for the original single-horizon Module 3 run). The rolling simulation optimizer supplies this vector at each review.
- `c[i]`: unit purchase cost; `K[i]`: fixed ordering cost per product order.
- `h[i]`: holding cost per unit per day.
- `b[i]`: lost-sales penalty per unit; in the initial run, set to product selling price as an explicit lost-revenue proxy.
- `C[i]`: source-dataset maximum quantity for a product/supplier order. This value varies by SKU even within a supplier, so it is treated as a per-product order limit.
- `C_shared[g]`: optional additional aggregate supplier limit across products. The raw dataset does not provide a consistent aggregate value; this constraint is disabled unless explicitly supplied.
- `B`: optional shared purchase budget. The source dataset has no budget, so it is disabled unless supplied by the user. It covers unit purchase spend, not fixed ordering charges.

## Decisions and recourse

First-stage decisions, chosen before any horizon demand is observed:

- `q[i] >= 0`, integer units ordered at day 0.
- `z[i] in {0,1}`, indicating whether product `i` is ordered.

Scenario recourse variables after demand is realized:

- `I[s,t,i] >= 0`: end-of-day inventory.
- `u[s,t,i] >= 0`: end-of-day lost sales (unmet demand).

There are no replenishment decisions after day 0. Inventory and lost sales are state/accounting recourse determined by the order and the realized demand history. Thus no future demand is used to choose an order, and no future order decision can violate nonanticipativity.

## Objective

Minimize expected total cost:

`sum_i (c[i] q[i] + K[i] z[i]) + sum_s p_s sum_t sum_i (h[i] I[s,t,i] + b[i] u[s,t,i])`.

The first term is purchase plus fixed ordering cost. The second is expected holding plus shortage/lost-sales cost.

## Constraints

Order activation and capacity:

`0 <= q[i] <= C[i] z[i]`, `q[i] integer`, `z[i] binary`.

Shared supplier capacity:

Optional shared supplier capacity, when provided: `sum_{i:g(i)=g} q[i] <= C_shared[g]`.

Inventory balance, for all `s,t,i`:

`I[s,t,i] = I[s,t-1,i] + P[t,i] + a[t,i] q[i] - D[s,t,i] + u[s,t,i]`,

where `I[s,0,i]=I0[i]`, `I[s,t,i]>=0`, `u[s,t,i]>=0`. Positive holding and lost-sales costs make inventory and lost sales mutually exclusive at an optimum. Lost sales satisfy demand without backlogging.

Optional shared purchasing budget:

`sum_i c[i] q[i] <= B`.

Scenario probabilities are normalized before optimization and appear directly in the expected-cost objective. Identical scenario histories share the same first-stage order; recourse state is linked by the balance equations.

## Contextual and non-contextual instances

- Contextual instance: uses the existing Module 2 conditional scenario paths.
- Non-contextual instance: uses the Module 2 product-mean Poisson baseline to sample product-specific unconditional demand, independent of date and context.

Both instances use the same products, costs, lead-time convention, capacities, optional budget, solver, and objective. They are optimized on their respective scenario distributions. Policies are also cross-evaluated on both sets of scenarios to make their expected costs comparable.

## Assumptions and scope

Orders are placed once at the start of each optimization horizon, immediately after current on-hand stock and known outstanding receipts are observed. `P[t,i]` carries deliveries already scheduled from earlier review windows; it is zero for the original single-horizon Module 3 run. A deterministic average lead time is used for new orders; lead-time variability and supplier reliability are simulated in Modules 4–5, not optimized. Dataset capacity is enforced per product; an additional shared supplier cap can be passed explicitly, but is not inferred because the source values vary within supplier. There is no warehouse capacity, backorder, substitution, salvage value, or order quantity minimum. The per-product fixed cost assumes a separate order charge for each SKU ordered. Lost-sales penalty defaults to selling price, an explicit proxy because the dataset has no shortage-cost field. Future promotion/festival assumptions are inherited from Module 2 scenarios (zero when unknown). Integer order quantities and binary activation are solved as a mixed-integer linear program.
