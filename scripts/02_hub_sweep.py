"""Stage 2: solve the baseline and sweep the number of hubs (1..8)."""
import _setup  # noqa: F401
import pandas as pd
from vnnet.model import build_instance, solve, kpis
from vnnet.params import BASE
from vnnet.paths import TABLES, load_nodes

inst = build_instance(load_nodes(), BASE)

best = solve(inst)
print("Unconstrained optimum:", kpis(best, inst))
best.assign.to_csv(TABLES / "baseline_assignment.csv", index=False)

rows = []
for p in range(1, 9):
    k = kpis(solve(inst, n_hubs=p), inst)
    rows.append({"hubs_requested": p, **k})
    print(p, k.get("status"), k.get("total_cost_usd_day"), k.get("hubs"))
pd.DataFrame(rows).to_csv(TABLES / "hub_sweep.csv", index=False)
