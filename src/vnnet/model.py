"""Capacitated hub-location model (MILP) solved with PuLP (bundled CBC solver).

Decision: which candidate cities to open as regional sortation hubs, and which
hub serves each city's delivery demand.

    min  sum_j F * y_j  +  sum_i sum_j d_i * c_ij * x_ij

    s.t. sum_j x_ij = 1                    every city's demand is served
         x_ij <= y_j                       only open hubs serve
         sum_i d_i * x_ij <= Cap * y_j     hub capacity (parcels/day)
         x_ij = 0 if road_km(i,j) > Dmax   next-day service radius (optional)
         sum_j y_j = p                     fixed hub count (optional, for sweeps)

With allow_overflow=True, each city may also hand parcels u_i to a 3PL at a
per-parcel price (sum_j x_ij + u_i = 1), so a fixed network stays feasible
under demand surges and the cost of absorbing them can be measured.

c_ij (USD/parcel) = inbound trunk cost from seller origins to hub j
                  + outbound line-haul from hub j to city i
                  + hub handling labour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import warnings

import pulp

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pulp")

from .data import distance_matrix
from .params import Params


@dataclass
class Instance:
    demand: pd.DataFrame          # node_id, name, lat, lon, parcels
    hubs: pd.DataFrame            # candidate hubs: node_id, name, lat, lon
    road_km: np.ndarray           # demand x hubs road km
    cost: np.ndarray              # demand x hubs USD per parcel
    trunk_cost: np.ndarray        # per hub USD per parcel (inbound)
    params: Params


def build_instance(nodes: pd.DataFrame, p: Params) -> Instance:
    demand = nodes[["node_id", "name", "lat", "lon", "population"]].copy()
    demand["parcels"] = demand["population"] * p.parcels_per_capita_day * p.demand_multiplier
    hubs = nodes.loc[nodes["is_candidate_hub"], ["node_id", "name", "lat", "lon"]].reset_index(drop=True)

    road_km = distance_matrix(demand, hubs) * p.circuity

    origins = nodes.set_index("name").loc[[o for o, _ in p.origin_shares], ["lat", "lon"]]
    shares = np.array([s for _, s in p.origin_shares])
    assert abs(shares.sum() - 1) < 1e-9, "origin shares must sum to 1"
    origin_hub_km = distance_matrix(origins.reset_index(), hubs) * p.circuity
    trunk_cost = (shares[:, None] * origin_hub_km).sum(axis=0) * p.trunk_usd_parcel_km * p.fuel_factor

    outbound = road_km * p.outbound_usd_parcel_km * p.fuel_factor
    cost = trunk_cost[None, :] + outbound + p.handling_effective
    return Instance(demand.reset_index(drop=True), hubs, road_km, cost, trunk_cost, p)


@dataclass
class Solution:
    status: str
    total_cost: float
    fixed_cost: float
    trunk_cost: float
    outbound_cost: float
    handling_cost: float
    overflow_cost: float
    overflow_parcels: float
    open_hubs: list
    assign: pd.DataFrame          # demand rows with assigned hub (largest share)
    flows: np.ndarray             # demand x hubs parcels

    @property
    def n_hubs(self):
        return len(self.open_hubs)


def solve(inst: Instance, n_hubs: int | None = None, enforce_service: bool = True,
          force_open: list | None = None, force_closed: list | None = None,
          allow_overflow: bool = False, time_limit: int = 120) -> Solution:
    p = inst.params
    I, J = range(len(inst.demand)), range(len(inst.hubs))
    d = inst.demand["parcels"].to_numpy()
    names = inst.hubs["name"].tolist()

    allowed = (inst.road_km <= p.max_outbound_km) if enforce_service else np.ones_like(inst.road_km, bool)

    m = pulp.LpProblem("vn_hub_location", pulp.LpMinimize)
    y = {j: pulp.LpVariable(f"y_{j}", cat="Binary") for j in J}
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", 0, 1) for i in I for j in J if allowed[i, j]}
    u = {i: pulp.LpVariable(f"u_{i}", 0, 1) for i in I} if allow_overflow else {}

    m += (pulp.lpSum(p.hub_fixed_effective * y[j] for j in J)
          + pulp.lpSum(d[i] * inst.cost[i, j] * v for (i, j), v in x.items())
          + pulp.lpSum(d[i] * p.overflow_3pl_usd_parcel * u[i] for i in u))

    by_i = {i: [] for i in I}
    by_j = {j: [] for j in J}
    for (i, j), v in x.items():
        by_i[i].append(v)
        by_j[j].append((i, v))
        m += v <= y[j]
    for i in I:
        if not by_i[i] and i not in u:
            return _infeasible(inst)
        m += pulp.lpSum(by_i[i]) + (u[i] if i in u else 0) == 1
    for j in J:
        m += pulp.lpSum(d[i] * v for i, v in by_j[j]) <= p.hub_capacity_effective * y[j]
    if n_hubs is not None:
        m += pulp.lpSum(y.values()) == n_hubs
    for name in force_open or []:
        m += y[names.index(name)] == 1
    for name in force_closed or []:
        m += y[names.index(name)] == 0

    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    status = pulp.LpStatus[m.status]
    if status != "Optimal":
        return _infeasible(inst, status)

    flows = np.zeros_like(inst.cost)
    for (i, j), v in x.items():
        flows[i, j] = d[i] * (v.value() or 0.0)
    open_j = [j for j in J if y[j].value() > 0.5]

    trunk = float((flows.sum(axis=0) * inst.trunk_cost).sum())
    outbound = float((flows * inst.road_km).sum() * p.outbound_usd_parcel_km * p.fuel_factor)
    handling = float(flows.sum() * p.handling_effective)
    fixed = p.hub_fixed_effective * len(open_j)
    over_parcels = float(sum(d[i] * (v.value() or 0.0) for i, v in u.items()))
    over_cost = over_parcels * p.overflow_3pl_usd_parcel

    a = inst.demand.copy()
    main = flows.argmax(axis=1)
    a["overflow_parcels"] = [d[i] * (u[i].value() or 0.0) if i in u else 0.0 for i in I]
    a["hub"] = [names[j] for j in main]
    a["road_km"] = inst.road_km[np.arange(len(a)), main]
    a["split"] = (flows > 1e-6).sum(axis=1) > 1
    return Solution(status, fixed + trunk + outbound + handling + over_cost, fixed, trunk, outbound,
                    handling, over_cost, over_parcels, [names[j] for j in open_j], a, flows)


def _infeasible(inst, status="Infeasible"):
    nan = np.nan
    return Solution(status, nan, nan, nan, nan, nan, nan, nan, [], inst.demand.copy(), None)


def kpis(sol: Solution, inst: Instance) -> dict:
    """Cost and service KPIs for a solved network."""
    p = inst.params
    if sol.status != "Optimal":
        return {"status": sol.status}
    own = sol.flows.sum()
    total_parcels = own + sol.overflow_parcels
    km_w = (sol.flows * inst.road_km).sum() / own
    within = lambda km: float((sol.flows * (inst.road_km <= km)).sum() / total_parcels)  # 3PL counted as not within
    hub_load = sol.flows.sum(axis=0)
    open_mask = hub_load > 1e-6
    util = hub_load[open_mask] / p.hub_capacity_effective
    return {
        "status": sol.status,
        "n_hubs": sol.n_hubs,
        "hubs": ", ".join(sol.open_hubs),
        "parcels_day": round(total_parcels),
        "total_cost_usd_day": round(sol.total_cost),
        "cost_per_parcel_usd": round(sol.total_cost / total_parcels, 4),
        "fixed_usd_day": round(sol.fixed_cost),
        "trunk_usd_day": round(sol.trunk_cost),
        "outbound_usd_day": round(sol.outbound_cost),
        "handling_usd_day": round(sol.handling_cost),
        "overflow_3pl_usd_day": round(sol.overflow_cost),
        "pct_parcels_to_3pl": round(100 * sol.overflow_parcels / total_parcels, 1),
        "avg_outbound_road_km": round(km_w, 1),
        "est_outbound_hours": round(km_w / p.truck_kmh, 1),
        "pct_demand_within_same_day_km": round(100 * within(p.same_day_km), 1),
        "pct_demand_within_next_day_km": round(100 * within(p.max_outbound_km), 1),
        "max_hub_utilisation_pct": round(100 * util.max(), 1) if len(util) else np.nan,
    }
