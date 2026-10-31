"""Build demand nodes and candidate hubs for mainland Vietnam from GeoNames."""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0088

# Island cities with no road connection to the mainland network.
ISLAND_EXCLUDE = {"Phu Quoc", "Phú Quốc", "Hàm Ninh"}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def distance_matrix(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """Straight-line km between every row of a (rows) and b (cols)."""
    return haversine_km(a["lat"].to_numpy()[:, None], a["lon"].to_numpy()[:, None],
                        b["lat"].to_numpy()[None, :], b["lon"].to_numpy()[None, :])


def load_raw_cities(min_pop: int = 15_000) -> pd.DataFrame:
    import geonamescache
    gc = geonamescache.GeonamesCache(min_city_population=min_pop)
    rows = [
        {"geonameid": c["geonameid"], "name": c["name"], "lat": c["latitude"],
         "lon": c["longitude"], "population": c["population"], "admin1": c["admin1code"]}
        for c in gc.get_cities().values() if c["countrycode"] == "VN"
    ]
    return pd.DataFrame(rows).sort_values("population", ascending=False).reset_index(drop=True)


def dedupe_metro_districts(df: pd.DataFrame, radius_km: float = 12.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GeoNames lists urban districts (e.g. Bình Thạnh in HCMC) as separate
    'cities' whose population is already inside the parent city's figure.
    Walk cities from largest to smallest and drop any city within radius_km of
    a larger kept city, so population is not double counted.

    Returns (kept, dropped) with a 'merged_into' column on dropped.
    """
    df = df.sort_values("population", ascending=False).reset_index(drop=True)
    kept_idx, merged_into = [], {}
    for i, r in df.iterrows():
        if kept_idx:
            k = df.loc[kept_idx]
            d = haversine_km(r.lat, r.lon, k["lat"].to_numpy(), k["lon"].to_numpy())
            j = int(np.argmin(d))
            if d[j] <= radius_km:
                merged_into[i] = k.iloc[j]["name"]
                continue
        kept_idx.append(i)
    kept = df.loc[kept_idx].reset_index(drop=True)
    dropped = df.loc[list(merged_into)].assign(merged_into=[merged_into[i] for i in merged_into])
    return kept, dropped.reset_index(drop=True)


def build_nodes(min_pop: int = 15_000, dedupe_km: float = 12.0,
                hub_min_pop: int = 200_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_raw_cities(min_pop)
    raw = raw[~raw["name"].isin(ISLAND_EXCLUDE)]
    nodes, dropped = dedupe_metro_districts(raw, dedupe_km)
    nodes["is_candidate_hub"] = nodes["population"] >= hub_min_pop
    nodes.insert(0, "node_id", range(len(nodes)))
    return nodes, dropped
