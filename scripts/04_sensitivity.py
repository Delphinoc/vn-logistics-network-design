"""Stage 4: one-at-a-time sensitivity (+/-25%) of the optimal network to each assumption."""
import _setup  # noqa: F401
import pandas as pd
from vnnet.model import build_instance, solve, kpis
from vnnet.params import BASE
from vnnet.paths import TABLES, load_nodes

nodes = load_nodes()
LEVERS = {
    "Hub fixed cost": "hub_fixed_usd_day",
    "Outbound line-haul rate": "outbound_usd_parcel_km",
    "Inbound trunk rate": "trunk_usd_parcel_km",
    "Parcels per capita": "parcels_per_capita_day",
    "Hub capacity": "hub_capacity_parcels_day",
    "Next-day service radius": "max_outbound_km",
}
base = kpis(solve(build_instance(nodes, BASE)), build_instance(nodes, BASE))
rows = []
for label, attr in LEVERS.items():
    for f in (0.75, 1.25):
        p = BASE.with_(**{attr: getattr(BASE, attr) * f})
        inst = build_instance(nodes, p)
        k = kpis(solve(inst), inst)
        rows.append({"assumption": label, "change": f"{int(round((f - 1) * 100)):+d}%", **k,
                     "cost_per_parcel_vs_base_pct": round(100 * (k["cost_per_parcel_usd"] / base["cost_per_parcel_usd"] - 1), 1)
                     if k["status"] == "Optimal" else None})
        print(label, f, k.get("n_hubs"), k.get("cost_per_parcel_usd"), k.get("hubs"))
pd.DataFrame(rows).to_csv(TABLES / "sensitivity.csv", index=False)
