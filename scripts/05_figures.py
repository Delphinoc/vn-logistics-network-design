"""Stage 5: charts for the README / deck."""
import _setup  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vnnet.model import build_instance, solve
from vnnet.params import BASE
from vnnet.paths import FIGURES, TABLES, load_nodes

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#b5b3ad"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def network_map(ax, inst, sol, title):
    a = sol.assign
    hubs = inst.hubs.set_index("name")
    colour = {h: SERIES[k] for k, h in enumerate(sorted(sol.open_hubs, key=lambda h: -hubs.loc[h, "lat"]))}
    over = a.get("overflow_parcels", pd.Series(0, index=a.index)) > 0.5 * a["parcels"]
    for _, r in a[~over].iterrows():
        h = hubs.loc[r.hub]
        ax.plot([h.lon, r.lon], [h.lat, r.lat], color=colour[r.hub], lw=0.6, alpha=0.35, zorder=1)
    s = 8 + 400 * a["parcels"] / a["parcels"].max()
    ax.scatter(a.loc[~over, "lon"], a.loc[~over, "lat"], s=s[~over], c=[colour[h] for h in a.loc[~over, "hub"]],
               edgecolor=SURFACE, linewidth=0.8, zorder=2)
    if over.any():
        ax.scatter(a.loc[over, "lon"], a.loc[over, "lat"], s=s[over], c=MUTED, edgecolor=SURFACE,
                   linewidth=0.8, zorder=2, label="Handed to 3PL")
    for h, c in colour.items():
        ax.scatter(hubs.loc[h, "lon"], hubs.loc[h, "lat"], marker="*", s=420, c=c, edgecolor=INK,
                   linewidth=1, zorder=3, label=f"Hub: {h}")
        ax.annotate(h, (hubs.loc[h, "lon"], hubs.loc[h, "lat"]), xytext=(10, -4),
                    textcoords="offset points", fontsize=9, fontweight="bold", color=INK, zorder=4)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=False, fontsize=8)


nodes = load_nodes()
inst = build_instance(nodes, BASE)
base = solve(inst)

# fig01: baseline network
fig, ax = plt.subplots(figsize=(6.5, 9))
network_map(ax, inst, base, "Cost-optimal network: 3 regional hubs")
ax.text(0, -0.08, "Dot size = estimated parcels/day (population proxy). Lines = hub-to-city outbound line-haul.",
        transform=ax.transAxes, fontsize=8, color=INK2)
fig.tight_layout()
fig.savefig(FIGURES / "fig01_baseline_network.png", dpi=160)
plt.close(fig)

# fig02: hub-count trade-off (two panels, no dual axis)
sw = pd.read_csv(TABLES / "hub_sweep.csv")
ok = sw[sw.status == "Optimal"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ok.hubs_requested, ok.cost_per_parcel_usd, color=SERIES[0], lw=2, marker="o", ms=8)
axes[0].set_title("Network cost per parcel")
axes[0].set_ylabel("USD per parcel")
axes[1].plot(ok.hubs_requested, ok.avg_outbound_road_km, color=SERIES[0], lw=2, marker="o", ms=8)
axes[1].set_title("Avg outbound distance (lead-time proxy)")
axes[1].set_ylabel("Road km, hub to city (parcel-weighted)")
for ax in axes:
    ax.set_xlabel("Number of hubs")
    ax.set_xlim(0.5, 8.5)
    ax.set_xticks(range(1, 9))
    ax.axvspan(0.5, 2.5, color=GRID, alpha=0.6, lw=0)
    ax.text(1.5, 0.5, "Infeasible:\nbreaks next-day\nradius", transform=ax.get_xaxis_transform(),
            ha="center", va="center", fontsize=8, color=INK2)
best = ok.loc[ok.cost_per_parcel_usd.idxmin()]
axes[0].annotate(f"Optimum: {int(best.hubs_requested)} hubs\n${best.cost_per_parcel_usd:.3f}/parcel",
                 (best.hubs_requested, best.cost_per_parcel_usd), xytext=(25, 55), textcoords="offset points",
                 fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2))
fig.tight_layout()
fig.savefig(FIGURES / "fig02_hub_count_tradeoff.png", dpi=160)
plt.close(fig)

# fig03: scenarios, keep network vs redesign
sc = pd.read_csv(TABLES / "scenarios.csv")
sc = sc[sc.scenario != "Baseline"]
order = list(dict.fromkeys(sc.scenario))
fig, ax = plt.subplots(figsize=(10, 5))
yy = np.arange(len(order))
for k, (resp, lab) in enumerate((("keep_network", "Keep current 3 hubs (+3PL overflow)"), ("redesign", "Re-optimise network"))):
    v = sc[sc.response == resp].set_index("scenario").loc[order, "cost_per_parcel_vs_baseline_pct"]
    ax.barh(yy + (k - 0.5) * 0.38, v, height=0.36, color=SERIES[k], label=lab)
    for y_, val in zip(yy, v):
        ax.text(val + (1 if val >= 0 else -1), y_ + (k - 0.5) * 0.38, f"{val:+.1f}%", va="center",
                ha="left" if val >= 0 else "right", fontsize=8, color=INK2)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(yy)
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_xlabel("Cost per parcel vs baseline (%)")
ax.set_title("Scenario analysis: what shocks cost, and how much redesign recovers", pad=28)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
ax.set_xlim(min(-12, sc.cost_per_parcel_vs_baseline_pct.min() - 8), sc.cost_per_parcel_vs_baseline_pct.max() + 12)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(FIGURES / "fig03_scenarios.png", dpi=160)
plt.close(fig)

# fig04: sensitivity tornado
se = pd.read_csv(TABLES / "sensitivity.csv")
piv = se.pivot(index="assumption", columns="change", values="cost_per_parcel_vs_base_pct")
piv = piv.loc[(piv.abs().max(axis=1)).sort_values(ascending=False).index]
hub_n = se.pivot(index="assumption", columns="change", values="n_hubs").loc[piv.index]
fig, ax = plt.subplots(figsize=(9, 5.2))
yy = np.arange(len(piv))
for k, ch in enumerate(("-25%", "+25%")):
    off = (k - 0.5) * 0.38
    ax.barh(yy + off, piv[ch], height=0.36, color=SERIES[k], label=f"Assumption {ch}")
    for y_, val, n in zip(yy, piv[ch], hub_n[ch]):
        ax.text(val + (0.4 if val >= 0 else -0.4), y_ + off, f"{val:+.1f}% ({int(n)} hubs)", va="center",
                ha="left" if val >= 0 else "right", fontsize=8, color=INK2)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(yy)
ax.set_yticklabels(piv.index)
ax.set_xlabel("Cost per parcel vs baseline (%), with optimal hub count in brackets")
ax.invert_yaxis()
ax.set_title("Sensitivity: which assumptions move the answer", pad=28)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
ax.grid(axis="y", visible=False)
lim = piv.abs().max().max() + 9
ax.set_xlim(-lim, lim)
fig.tight_layout()
fig.savefig(FIGURES / "fig04_sensitivity.png", dpi=160)
plt.close(fig)

# fig05: disruption, keep vs redesign
down = sc[sc.scenario.str.startswith("Disruption")].scenario.iloc[0].split(": ")[1].replace(" hub offline", "")
others = [h for h in base.open_hubs if h != down]
all_h = inst.hubs["name"].tolist()
keep = solve(inst, force_open=others, force_closed=[h for h in all_h if h not in others], allow_overflow=True)
redo = solve(inst, force_closed=[down])
fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
network_map(axes[0], inst, keep, f"{down} offline: keep remaining hubs")
network_map(axes[1], inst, redo, f"{down} offline: re-optimise")
fig.tight_layout()
fig.savefig(FIGURES / "fig05_disruption.png", dpi=160)
plt.close(fig)
print("Saved figures to", FIGURES)
