"""Stage 3: scenario analysis.

For each scenario we compare two responses:
  * keep_network - keep today's (baseline-optimal) hubs, re-route parcels and
                   hand any parcels the network cannot serve to a 3PL
  * redesign     - re-optimise which hubs to open
The gap between them is the value of adapting the network.
"""
import _setup  # noqa: F401
import pandas as pd
from vnnet.model import build_instance, solve, kpis
from vnnet.params import BASE
from vnnet.paths import TABLES, load_nodes

nodes = load_nodes()
base_inst = build_instance(nodes, BASE)
base_sol = solve(base_inst)
base_hubs = base_sol.open_hubs
all_hubs = base_inst.hubs["name"].tolist()

SCENARIOS = {
    "Baseline": {},
    "Peak campaign demand +50%": {"demand_multiplier": 1.5},
    "Fuel price +30%": {"fuel_multiplier": 1.3},
    "Labour cost +20%": {"labour_multiplier": 1.2},
    "Labour shortage: hub capacity -30%": {"capacity_multiplier": 0.7},
    "Combined: demand +30%, fuel +20%, labour +10%": {"demand_multiplier": 1.3, "fuel_multiplier": 1.2, "labour_multiplier": 1.1},
}

rows = []
for name, kw in SCENARIOS.items():
    inst = build_instance(nodes, BASE.with_(**kw))
    keep = solve(inst, force_open=base_hubs, force_closed=[h for h in all_hubs if h not in base_hubs],
                 allow_overflow=True)
    redo = solve(inst)
    for resp, sol in (("keep_network", keep), ("redesign", redo)):
        rows.append({"scenario": name, "response": resp, **kpis(sol, inst)})
        print(name, resp, sol.status, round(sol.total_cost) if sol.status == "Optimal" else "")

# Disruption: the largest-volume baseline hub goes offline (flood, fire, outage).
loads = dict(zip(base_inst.hubs["name"], base_sol.flows.sum(axis=0)))
down = max(base_hubs, key=lambda h: loads[h])
name = f"Disruption: {down} hub offline"
others = [h for h in base_hubs if h != down]
keep = solve(base_inst, force_open=others, force_closed=[h for h in all_hubs if h not in others],
             allow_overflow=True)
redo = solve(base_inst, force_closed=[down])
for resp, sol in (("keep_network", keep), ("redesign", redo)):
    rows.append({"scenario": name, "response": resp, **kpis(sol, base_inst)})
    print(name, resp, sol.status, round(sol.total_cost) if sol.status == "Optimal" else "")

out = pd.DataFrame(rows)
base_cost = out.loc[(out.scenario == "Baseline") & (out.response == "redesign"), "total_cost_usd_day"].iloc[0]
base_cpp = out.loc[(out.scenario == "Baseline") & (out.response == "redesign"), "cost_per_parcel_usd"].iloc[0]
out["total_cost_vs_baseline_pct"] = (100 * (out["total_cost_usd_day"] / base_cost - 1)).round(1)
out["cost_per_parcel_vs_baseline_pct"] = (100 * (out["cost_per_parcel_usd"] / base_cpp - 1)).round(1)
out.to_csv(TABLES / "scenarios.csv", index=False)
