"""Model tests on tiny hand-checkable instances."""
import numpy as np
import pandas as pd
import pytest

from vnnet.model import build_instance, kpis, solve
from vnnet.params import Params


def tiny_nodes():
    # Two clusters ~1,100 km apart; origins are the two big cities.
    return pd.DataFrame({
        "node_id": range(4),
        "name": ["Ho Chi Minh City", "SouthTown", "Hanoi", "NorthTown"],
        "lat": [10.8, 10.4, 21.0, 21.4],
        "lon": [106.6, 106.3, 105.8, 106.2],
        "population": [10_000_000, 1_000_000, 8_000_000, 1_000_000],
        "is_candidate_hub": [True, True, True, True],
    })


P = Params(hub_fixed_usd_day=1_000, hub_capacity_parcels_day=1e9)


def test_all_demand_served_exactly_once():
    inst = build_instance(tiny_nodes(), P)
    sol = solve(inst)
    assert sol.status == "Optimal"
    np.testing.assert_allclose(sol.flows.sum(axis=1), inst.demand["parcels"], rtol=1e-6)


def test_service_radius_forces_one_hub_per_cluster():
    inst = build_instance(tiny_nodes(), P)
    sol = solve(inst)
    assert sol.n_hubs == 2
    assert (sol.assign["road_km"] <= P.max_outbound_km).all()


def test_single_hub_infeasible_with_radius_feasible_without():
    inst = build_instance(tiny_nodes(), P)
    assert solve(inst, n_hubs=1).status != "Optimal"
    assert solve(inst, n_hubs=1, enforce_service=False).status == "Optimal"


def test_capacity_respected_and_forces_extra_hub():
    p = P.with_(hub_capacity_parcels_day=150_000)  # demand is 400k parcels/day
    inst = build_instance(tiny_nodes(), p)
    sol = solve(inst)
    assert sol.flows.sum(axis=0).max() <= 150_000 + 1e-3
    assert sol.n_hubs >= 3


def test_cost_components_add_up():
    inst = build_instance(tiny_nodes(), P)
    s = solve(inst)
    assert s.total_cost == pytest.approx(s.fixed_cost + s.trunk_cost + s.outbound_cost + s.handling_cost + s.overflow_cost)


def test_overflow_keeps_fixed_network_feasible():
    p = P.with_(hub_capacity_parcels_day=150_000)
    inst = build_instance(tiny_nodes(), p)
    sol = solve(inst, force_open=["Hanoi"], force_closed=["Ho Chi Minh City", "SouthTown", "NorthTown"],
                allow_overflow=True)
    assert sol.status == "Optimal"
    assert sol.overflow_parcels == pytest.approx(inst.demand["parcels"].sum() - 150_000, rel=1e-6)


def test_fuel_shock_raises_cost():
    a = solve(build_instance(tiny_nodes(), P)).total_cost
    b = solve(build_instance(tiny_nodes(), P.with_(fuel_multiplier=1.3))).total_cost
    assert b > a


def test_kpis_percentages_in_range():
    inst = build_instance(tiny_nodes(), P)
    k = kpis(solve(inst), inst)
    assert 0 <= k["pct_demand_within_same_day_km"] <= k["pct_demand_within_next_day_km"] <= 100
