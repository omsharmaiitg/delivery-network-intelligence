"""
================================================================================
03 | BOTTLENECK & CORRIDOR AUDIT  -  Graph-Based Network Intelligence
================================================================================
PURPOSE
    Use the graph structure to answer: which HUBS are critical chokepoints, and
    which CORRIDORS are chronically delayed?  We compute:
      - betweenness centrality  (how much flow must pass through a hub)
      - in / out degree         (fan-in / fan-out connectivity)
      - clustering coefficient  (is the hub locally bypassable?)
      - pagerank                (volume-aware structural importance)
    and combine them with each hub's SLA-breach contribution (its share of the
    network's total excess delay minutes) to rank true operational bottlenecks.

OUTPUTS
    data/node_metrics.parquet        every facility + graph metrics + breach share
    outputs/tables/top_bottleneck_hubs.csv
    outputs/tables/chronic_corridors.csv
================================================================================
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data")
T = os.path.join(ROOT, "outputs", "tables")


def minmax(s):
    s = s.astype(float)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0.0


def main():
    print("[1/6] Loading graph + tables ...")
    with open(os.path.join(D, "graph.gpickle"), "rb") as f:
        G = pickle.load(f)
    nodes = pd.read_parquet(os.path.join(D, "nodes.parquet"))
    edges = pd.read_parquet(os.path.join(D, "edges.parquet"))

    # ---- CENTRALITY --------------------------------------------------------
    print("[2/6] Betweenness centrality (free-flow-time weighted) ...")
    # Cost = OSRM free-flow time -> shortest physical paths.  A hub with high
    # betweenness sits on many of the network's natural least-time routes.
    btw = nx.betweenness_centrality(G, weight="cost", normalized=True)

    print("[3/6] PageRank, degree, clustering ...")
    # PageRank on volume-weighted graph -> where does throughput concentrate.
    pr = nx.pagerank(G, weight="volume")
    indeg = dict(G.in_degree())
    outdeg = dict(G.out_degree())
    win = dict(G.in_degree(weight="volume"))
    wout = dict(G.out_degree(weight="volume"))
    # clustering on undirected projection (standard interpretation)
    clust = nx.clustering(G.to_undirected())

    met = pd.DataFrame({
        "center": list(G.nodes()),
        "betweenness": [btw[n] for n in G.nodes()],
        "pagerank": [pr[n] for n in G.nodes()],
        "in_degree": [indeg[n] for n in G.nodes()],
        "out_degree": [outdeg[n] for n in G.nodes()],
        "w_in_volume": [win[n] for n in G.nodes()],
        "w_out_volume": [wout[n] for n in G.nodes()],
        "clustering": [clust[n] for n in G.nodes()],
    })
    met = met.merge(nodes, on="center", how="left")

    # ---- SLA BREACH CONTRIBUTION ------------------------------------------
    print("[4/6] SLA-breach contribution per hub ...")
    total_excess = met["total_excess_min"].sum()
    met["breach_contrib_pct"] = 100.0 * met["total_excess_min"] / total_excess
    # delay intensity of the hub's own outbound corridors
    met["out_delay_ratio"] = met["out_median_ratio"]

    # ---- COMPOSITE BOTTLENECK SCORE ---------------------------------------
    # A real bottleneck is (a) structurally central, (b) high-throughput and
    # (c) slow.  Blend normalized components; weights chosen to favour structural
    # position while still requiring genuine delay + volume.
    print("[5/6] Composite bottleneck score ...")
    met["n_btw"] = minmax(met["betweenness"])
    met["n_pr"] = minmax(met["pagerank"])
    met["n_vol"] = minmax(np.log1p(met["total_volume"]))
    met["n_delay"] = minmax(met["out_delay_ratio"].clip(upper=met["out_delay_ratio"].quantile(0.99)))
    met["n_breach"] = minmax(np.log1p(met["total_excess_min"]))
    met["bottleneck_score"] = (
        0.30 * met["n_btw"] +
        0.20 * met["n_pr"] +
        0.20 * met["n_vol"] +
        0.15 * met["n_delay"] +
        0.15 * met["n_breach"]
    )

    # ---- CHRONIC CORRIDORS -------------------------------------------------
    print("[6/6] Ranking chronic corridors + saving ...")
    # Task definition: actual exceeds OSRM by >20%. Because that is near-universal
    # here, we rank by a SEVERITY x VOLUME impact = total excess delay minutes,
    # restricted to corridors with stable support.
    chronic = edges[(edges["median_delay_ratio"] > 1.20) & edges["stable"]].copy()
    chronic["impact_excess_min"] = chronic["total_excess_min"]
    chronic = chronic.sort_values("impact_excess_min", ascending=False)
    chronic_out = chronic[[
        "corridor", "src_name", "dst_name", "src_state", "dst_state",
        "median_delay_ratio", "median_actual", "median_osrm", "median_dist_km",
        "n_traversals", "sla_breach_rate", "total_excess_min",
        "ratio_ftl", "ratio_carting",
        "ratio_morning", "ratio_afternoon", "ratio_evening", "ratio_night",
    ]].round(3)

    # ---- HUB RANKING -------------------------------------------------------
    hub_rank = met.sort_values("breach_contrib_pct", ascending=False).copy()
    hub_out = hub_rank[[
        "center", "name", "city", "state",
        "betweenness", "pagerank", "in_degree", "out_degree",
        "w_in_volume", "w_out_volume", "clustering",
        "total_volume", "out_delay_ratio", "total_excess_min",
        "breach_contrib_pct", "bottleneck_score",
    ]].round(4)

    met.to_parquet(os.path.join(D, "node_metrics.parquet"), index=False)
    hub_out.to_csv(os.path.join(T, "top_bottleneck_hubs.csv"), index=False)
    chronic_out.to_csv(os.path.join(T, "chronic_corridors.csv"), index=False)

    # ---- CONSOLE REPORT ----------------------------------------------------
    print("\n" + "=" * 78)
    print("TOP 10 BOTTLENECK HUBS  (by share of network excess delay minutes)")
    print("=" * 78)
    show = hub_out.head(10)[["name", "betweenness", "in_degree", "out_degree",
                             "total_volume", "out_delay_ratio",
                             "breach_contrib_pct"]]
    show.columns = ["facility", "btwn", "in", "out", "vol", "delayX", "breach%"]
    print(show.to_string(index=False))

    print("\n" + "=" * 78)
    print("TOP 10 CHRONIC CORRIDORS  (by total excess delay minutes, support>=3)")
    print("=" * 78)
    sc = chronic_out.head(10)[["src_name", "dst_name", "median_delay_ratio",
                               "median_actual", "median_osrm", "n_traversals",
                               "total_excess_min"]]
    sc.columns = ["from", "to", "ratioX", "act_min", "osrm_min", "n", "excess_min"]
    print(sc.to_string(index=False))

    print(f"\nNetwork total excess delay = {total_excess:,.0f} minutes")
    print(f"Top-5 hubs together carry "
          f"{hub_out.head(5)['breach_contrib_pct'].sum():.1f}% of it.")


if __name__ == "__main__":
    main()
