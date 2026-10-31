import numpy as np
import pandas as pd
import pytest

from vnnet.data import build_nodes, dedupe_metro_districts, haversine_km


def test_haversine_hcmc_hanoi():
    # Straight-line HCMC-Hanoi is roughly 1,140 km.
    d = haversine_km(10.823, 106.630, 21.025, 105.841)
    assert 1100 < d < 1180


def test_haversine_zero_and_symmetric():
    assert haversine_km(16.0, 108.2, 16.0, 108.2) == pytest.approx(0)
    assert haversine_km(10, 106, 21, 105) == pytest.approx(haversine_km(21, 105, 10, 106))


def test_dedupe_drops_district_inside_parent_city():
    df = pd.DataFrame({"name": ["Big", "District", "FarTown"], "lat": [10.0, 10.05, 12.0],
                       "lon": [106.0, 106.0, 108.0], "population": [1_000_000, 300_000, 200_000]})
    kept, dropped = dedupe_metro_districts(df, radius_km=12)
    assert kept["name"].tolist() == ["Big", "FarTown"]
    assert dropped.loc[0, "merged_into"] == "Big"


@pytest.fixture(scope="module")
def nodes():
    return build_nodes()[0]


def test_nodes_are_mainland_and_unique(nodes):
    assert not nodes["name"].isin(["Phu Quoc", "Phú Quốc"]).any()
    assert nodes["node_id"].is_unique
    assert {"Ho Chi Minh City", "Hanoi"} <= set(nodes["name"])


def test_no_two_nodes_closer_than_dedupe_radius(nodes):
    lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(d, np.inf)
    assert d.min() > 12
