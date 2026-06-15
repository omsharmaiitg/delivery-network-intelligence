"""
================================================================================
06 | VISUALIZATIONS  -  Graph-Based Network Intelligence
================================================================================
Produces the figure set for the deliverable:
    fig1_network_map.png        geographic map: top bottleneck hubs + worst corridors
    fig2_bottleneck_hubs.png    hub ranking by share of network delay
    fig3_model_comparison.png   baseline vs graph-enhanced (MAE + HIT@15%)
    fig4_ftl_carting.png        route-type decision boundary + time/cost trade-off
    fig5_osrm_miscalibration.png the structural ~2x OSRM gap
    fig6_feature_importance.png what the graph model leans on
================================================================================
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "font.family": "DejaVu Sans",
})
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data")
T = os.path.join(ROOT, "outputs", "tables")
F = os.path.join(ROOT, "outputs", "figures")

INK = "#1d2733"; ACCENT = "#c1440e"; BLUE = "#2c5f8a"; GREEN = "#2e7d52"
GREY = "#9aa7b4"; SAND = "#f4efe6"

# approximate lat/long for the hub cities (deg)
CITY_LL = {
    "Gurgaon": (28.46, 77.03), "Delhi": (28.61, 77.21), "Sonipat": (28.99, 77.02),
    "Chandigarh": (30.73, 76.78), "Ludhiana": (30.90, 75.86), "Jaipur": (26.91, 75.79),
    "Bangalore": (12.97, 77.59), "Bengaluru": (12.97, 77.59), "Hyderabad": (17.38, 78.49),
    "Bhiwandi": (19.30, 73.06), "Mumbai": (19.08, 72.88), "Pune": (18.52, 73.86),
    "Kolkata": (22.57, 88.36), "Bhubaneshwar": (20.30, 85.82), "Guwahati": (26.14, 91.74),
    "Bhopal": (23.26, 77.41), "Kanpur": (26.45, 80.33), "Ahmedabad": (23.02, 72.57),
    "MAA": (13.08, 80.27), "Surat": (21.17, 72.83), "Ranchi": (23.34, 85.31),
    "Muzaffrpur": (26.12, 85.39), "Purnia": (25.78, 87.47), "Vijayawada": (16.51, 80.65),
    "Sasaram": (24.95, 84.03), "Akola": (20.71, 77.00), "Visakhapatnam": (17.69, 83.22),
    "Goa": (15.30, 74.12), "Solapur": (17.66, 75.91),
}


def city_of(name):
    return str(name).split("_")[0].strip()


# --------------------------------------------------------------------------- #
def fig_network_map(met, edges):
    hubs = met.sort_values("breach_contrib_pct", ascending=False).head(22).copy()
    hubs["cc"] = hubs["city"].map(city_of)
    hubs = hubs[hubs["cc"].isin(CITY_LL)].drop_duplicates("cc")

    chronic = pd.read_csv(os.path.join(T, "chronic_corridors.csv")).head(22)

    fig, ax = plt.subplots(figsize=(9.5, 10))
    ax.set_facecolor(SAND)
    # corridors first (behind nodes)
    maxex = chronic["total_excess_min"].max()
    for _, r in chronic.iterrows():
        a, b = city_of(r["src_name"]), city_of(r["dst_name"])
        if a in CITY_LL and b in CITY_LL:
            (y1, x1), (y2, x2) = CITY_LL[a], CITY_LL[b]
            lw = 0.6 + 4.2 * (r["total_excess_min"] / maxex)
            ax.plot([x1, x2], [y1, y2], color=ACCENT,
                    alpha=0.30 + 0.5 * (r["total_excess_min"] / maxex),
                    lw=lw, zorder=1, solid_capstyle="round")
    # nodes
    smax = hubs["breach_contrib_pct"].max()
    for _, r in hubs.iterrows():
        lat, lon = CITY_LL[r["cc"]]
        size = 120 + 2300 * (r["breach_contrib_pct"] / smax)
        ax.scatter(lon, lat, s=size, c=ACCENT, alpha=0.85,
                   edgecolors="white", linewidths=1.4, zorder=3)
        ax.annotate(f"{r['cc']}\n{r['breach_contrib_pct']:.1f}%",
                    (lon, lat), fontsize=8.0, ha="center", va="center",
                    color="white", weight="bold", zorder=4)
    ax.set_xlim(67, 96); ax.set_ylim(7, 33)
    ax.set_title("Delhivery network — bottleneck hubs & worst delay corridors\n"
                 "(node size & label = share of total network delay; "
                 "red line width = corridor excess delay)", fontsize=12)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.grid(alpha=0.15)
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT,
                  markersize=12, label="Bottleneck hub"),
           Line2D([0], [0], color=ACCENT, lw=3, label="Chronic-delay corridor")]
    ax.legend(handles=leg, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig1_network_map.png"), bbox_inches="tight")
    plt.close(fig)


def fig_bottleneck_hubs(met):
    top = met.sort_values("breach_contrib_pct", ascending=False).head(12).iloc[::-1]
    labels = [f"{city_of(n)} ({s})" for n, s in zip(top["name"], top["state"])]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(labels, top["breach_contrib_pct"], color=BLUE, alpha=0.9)
    # color the worst 3 in accent
    for i in range(len(bars) - 3, len(bars)):
        bars[i].set_color(ACCENT)
    for i, (v, b) in enumerate(zip(top["betweenness"], top["breach_contrib_pct"])):
        ax.text(b + 0.12, i, f"{b:.1f}%  (btw {v:.02f})", va="center", fontsize=8.5)
    ax.set_xlabel("Share of total network excess-delay minutes (%)")
    ax.set_title("Top bottleneck hubs by SLA-breach contribution")
    ax.margins(x=0.18)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig2_bottleneck_hubs.png"), bbox_inches="tight")
    plt.close(fig)


def fig_model_comparison():
    comp = pd.read_csv(os.path.join(T, "model_comparison.csv"))
    order = ["OSRM (current)", "XGB baseline (trip features)", "XGB graph-enhanced"]
    comp = comp[comp["model"].isin(order)].set_index("model").loc[order].reset_index()
    short = ["OSRM\n(today)", "XGB\nbaseline", "XGB\ngraph-enhanced"]
    colors = [GREY, BLUE, ACCENT]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.bar(short, comp["MAE"], color=colors, alpha=0.9)
    for i, v in enumerate(comp["MAE"]):
        ax1.text(i, v + 2, f"{v:.1f}", ha="center", fontsize=10, weight="bold")
    ax1.set_ylabel("MAE (minutes)  ↓ lower is better")
    ax1.set_title("ETA error: graph model cuts MAE")
    ax2.bar(short, comp["HIT15"], color=colors, alpha=0.9)
    for i, v in enumerate(comp["HIT15"]):
        ax2.text(i, v + 1.2, f"{v:.0f}%", ha="center", fontsize=10, weight="bold")
    ax2.set_ylabel("% trips within 15% of actual  ↑ higher is better")
    ax2.set_title("Business metric: on-target ETA rate")
    fig.suptitle("Baseline vs graph-enhanced ETA  (held-out test set)",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig3_model_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def fig_ftl_carting():
    dec = pd.read_csv(os.path.join(T, "ftl_carting_decision.csv"))
    summ = pd.read_csv(os.path.join(T, "ftl_carting_summary.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # left: % FTL recommended by distance band
    ax1.bar(summ["dist_band"], summ["pct_recommend_ftl"], color=ACCENT, alpha=0.85)
    for i, v in enumerate(summ["pct_recommend_ftl"]):
        ax1.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9.5, weight="bold")
    ax1.set_ylabel("% of corridors where FTL is recommended")
    ax1.set_xlabel("Corridor distance band")
    ax1.set_title("Route-type decision boundary")
    ax1.set_ylim(0, 108)
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")
    # right: time saved vs cost premium scatter
    s = dec.sample(min(900, len(dec)), random_state=1)
    sc = ax2.scatter(s["median_dist_km"], s["time_saved_ftl"],
                     c=(s["recommend"] == "FTL").map({True: 1, False: 0}),
                     cmap="coolwarm", alpha=0.6, s=18)
    ax2.axhline(0, color=GREY, lw=1, ls="--")
    ax2.set_xlabel("Corridor distance (km)")
    ax2.set_ylabel("Predicted time saved by FTL (min)")
    ax2.set_title("FTL time advantage grows with distance")
    ax2.set_xscale("log")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#b40426",
                      markersize=9, label="FTL recommended"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="#3b4cc0",
                      markersize=9, label="Carting recommended")]
    ax2.legend(handles=handles, loc="upper left", framealpha=0.9)
    fig.suptitle("FTL vs Carting — data-backed route-type framework",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig4_ftl_carting.png"), bbox_inches="tight")
    plt.close(fig)


def fig_osrm_miscalibration(legs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    r = legs["delay_ratio"].clip(0, 6)
    ax1.hist(r, bins=60, color=BLUE, alpha=0.85)
    ax1.axvline(1.0, color=GREEN, lw=2, ls="--", label="OSRM perfect (1.0×)")
    ax1.axvline(legs["delay_ratio"].median(), color=ACCENT, lw=2,
                label=f"median {legs['delay_ratio'].median():.2f}×")
    ax1.set_xlabel("Actual ÷ OSRM time (delay ratio)")
    ax1.set_ylabel("Number of legs")
    ax1.set_title("OSRM underestimates by ~2× across the board")
    ax1.legend()
    # right: actual vs osrm scatter
    s = legs.sample(min(4000, len(legs)), random_state=1)
    ax2.scatter(s["osrm_time"], s["actual_time"], s=6, alpha=0.25, color=INK)
    lim = np.percentile(legs["actual_time"], 99)
    ax2.plot([0, lim], [0, lim], color=GREEN, lw=2, ls="--", label="y = x (OSRM ideal)")
    ax2.plot([0, lim], [0, 2 * lim], color=ACCENT, lw=2, label="y = 2x (observed median)")
    ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
    ax2.set_xlabel("OSRM predicted time (min)")
    ax2.set_ylabel("Actual time (min)")
    ax2.set_title("Actual vs predicted leg time")
    ax2.legend()
    fig.suptitle("The problem: today's OSRM ETA is structurally low",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig5_osrm_miscalibration.png"), bbox_inches="tight")
    plt.close(fig)


def fig_feature_importance():
    imp = pd.read_csv(os.path.join(T, "graph_feature_importance.csv"),
                      index_col=0).iloc[:, 0].head(10).iloc[::-1]
    nice = {"prior_actual": "Corridor history (median actual)*",
            "node2vec_embeddings(all)": "node2vec embeddings*",
            "osrm_time": "OSRM time", "osrm_distance": "OSRM distance",
            "prior_ratio": "Corridor delay ratio*", "dst_indeg": "Dest in-degree*",
            "prior_n": "Corridor support*", "src_outdeg": "Source out-degree*",
            "dst_btw": "Dest betweenness*", "tod_code": "Time of day",
            "actual_distance": "Actual distance", "dst_pr": "Dest PageRank*"}
    labels = [nice.get(i, i) for i in imp.index]
    colors = [ACCENT if "*" in l else BLUE for l in labels]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(labels, imp.values, color=colors, alpha=0.9)
    ax.set_xlabel("XGBoost feature importance (gain)")
    ax.set_title("What the graph model relies on  (* = graph-derived feature)")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig6_feature_importance.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading ...")
    met = pd.read_parquet(os.path.join(D, "node_metrics.parquet"))
    edges = pd.read_parquet(os.path.join(D, "edges.parquet"))
    legs = pd.read_parquet(os.path.join(D, "legs.parquet"))
    print("fig1 network map ...");        fig_network_map(met, edges)
    print("fig2 bottleneck hubs ...");    fig_bottleneck_hubs(met)
    print("fig3 model comparison ...");   fig_model_comparison()
    print("fig4 ftl vs carting ...");     fig_ftl_carting()
    print("fig5 osrm miscalibration ..."); fig_osrm_miscalibration(legs)
    print("fig6 feature importance ...");  fig_feature_importance()
    print("Saved 6 figures to outputs/figures/")


if __name__ == "__main__":
    main()
