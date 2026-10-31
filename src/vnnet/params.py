"""Model parameters.

Every cost and volume figure here is an ILLUSTRATIVE ASSUMPTION, not an observed
Shopee/SPX or market figure. They are set to plausible orders of magnitude so the
model's trade-offs (fixed hub cost vs transport cost vs service radius) are
meaningful. Results should be read as relative comparisons between network
designs and scenarios, not as forecasts of real costs.
"""
from dataclasses import dataclass, field, replace

USD = float


@dataclass(frozen=True)
class Params:
    # --- Demand ---------------------------------------------------------
    # Parcels per resident per day, used to turn city population into parcel
    # demand (population is a proxy; real demand skews to urban, richer areas).
    parcels_per_capita_day: float = 0.02
    demand_multiplier: float = 1.0          # scenario lever (e.g. 11.11 peak)

    # Share of parcels that originate (seller pickup / first mile) in each
    # metro. E-commerce supply is concentrated in the two largest metros.
    origin_shares: tuple = (("Ho Chi Minh City", 0.6), ("Hanoi", 0.4))

    # --- Geography --------------------------------------------------------
    circuity: float = 1.3                   # road km = straight-line km x circuity
    max_outbound_km: float = 450.0          # next-day service radius, hub -> city (road km)
    same_day_km: float = 150.0              # reporting threshold only
    truck_kmh: float = 50.0                 # average line-haul speed, for lead-time estimate

    # --- Costs (USD) ------------------------------------------------------
    hub_fixed_usd_day: USD = 20_000.0       # rent, equipment amortisation, base staffing
    hub_fixed_labour_share: float = 0.4     # part of fixed cost that is labour
    handling_usd_parcel: USD = 0.04         # variable sort labour per parcel at hub
    trunk_usd_parcel_km: USD = 0.00012      # origin -> hub, full 30t trucks
    outbound_usd_parcel_km: USD = 0.0004    # hub -> city delivery station, smaller trucks
    fuel_share_transport: float = 0.35      # share of transport cost that is fuel

    # Overflow parcels handed to a third-party logistics (3PL) partner when the
    # own network cannot serve them (capacity or service radius). Only used when
    # solve(..., allow_overflow=True).
    overflow_3pl_usd_parcel: USD = 0.60

    # --- Capacity ----------------------------------------------------------
    hub_capacity_parcels_day: float = 600_000.0

    # --- Scenario multipliers -------------------------------------------
    fuel_multiplier: float = 1.0
    labour_multiplier: float = 1.0
    capacity_multiplier: float = 1.0

    @property
    def fuel_factor(self) -> float:
        """Multiplier on transport cost from a fuel price change."""
        s = self.fuel_share_transport
        return (1 - s) + s * self.fuel_multiplier

    @property
    def hub_fixed_effective(self) -> float:
        l = self.hub_fixed_labour_share
        return self.hub_fixed_usd_day * ((1 - l) + l * self.labour_multiplier)

    @property
    def handling_effective(self) -> float:
        return self.handling_usd_parcel * self.labour_multiplier

    @property
    def hub_capacity_effective(self) -> float:
        return self.hub_capacity_parcels_day * self.capacity_multiplier

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


BASE = Params()
