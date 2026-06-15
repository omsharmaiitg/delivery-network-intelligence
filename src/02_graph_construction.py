"""
================================================================================
02 | GRAPH CONSTRUCTION  -  Delhivery Graph-Based Network Intelligence
================================================================================
PURPOSE
    Turn the leg-level table into a DIRECTED WEIGHTED GRAPH of the logistics
    network:  facilities = nodes,  corridors (source->dest) = edges.

    Edge weight captures the MEDIAN actual-vs-OSRM delay ratio per corridor,
    and we also persist route-type- and time-of-day-stratified weights so the
    same corridor can be read at different operating conditions.

OUTPUTS
    data/edges.parquet     one row per corridor (directed edge) + stratified stats
    data/nodes.parquet     one row per facility (node) + volume attributes
    data/graph.gpickle     networkx.DiGraph for centrality work in module 03
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
LEGS = os.path.join(ROOT, "data", "legs.parquet")

MIN_SUPPORT = 3   # min traversals for a corridor stat to be considered "stable"


def stratified_medians(legs: pd.DataFrame) -> pd.DataFrame:
    """Median delay ratio per corridor, broken out by route type and ToD."""
    # overall per corridor
    base = legs.groupby("corridor").agg(
        median_delay_ratio=("delay_ratio", "median"),
        mean_delay_ratio=("delay_ratio", "mean"),
        median_actual=("actual_time", "median"),
        median_osrm=("osrm_time", "median"),
        median_excess_min=("delay_min", "median"),
        median_dist_km=("osrm_distance", "median"),
        n_traversals=("delay_ratio", "size"),
        sla_breach_rate=("sla_breach", "mean"),
        total_excess_min=("delay_min", "sum"),
        ftl_share=("is_ftl", "mean"),
    )

    # route-type stratified (wide)
    rt = (legs.groupby(["corridor", "route_type"])["delay_ratio"].median()
          .unstack("route_type"))
    rt.columns = [f"ratio_{c.lower()}" for c in rt.columns]

    # time-of-day stratified (wide)
    tod = (legs.groupby(["corridor", "tod"])["delay_ratio"].median()
           .unstack("tod"))
    tod.columns = [f"ratio_{c}" for c in tod.columns]

    out = base.join(rt).join(tod).reset_index()
    return out


def main():
    print("[1/5] Loading legs ...")
    legs = pd.read_parquet(LEGS)

    # ---- EDGE TABLE --------------------------------------------------------
    print("[2/5] Aggregating corridors into directed edges ...")
    edges = stratified_medians(legs)
    # attach endpoint identifiers / names
    ends = (legs.groupby("corridor")
            .agg(source_center=("source_center", "first"),
                 destination_center=("destination_center", "first"),
                 src_name=("source_name", "first"),
                 dst_name=("destination_name", "first"),
                 src_state=("src_state", "first"),
                 dst_state=("dst_state", "first"))
            .reset_index())
    edges = edges.merge(ends, on="corridor")
    edges["stable"] = edges["n_traversals"] >= MIN_SUPPORT
    edges["chronic_delay"] = edges["median_delay_ratio"] > 1.20   # task definition
    print(f"      edges (corridors) = {len(edges):,}  "
          f"(stable >= {MIN_SUPPORT}: {edges['stable'].sum():,})")

    # ---- NODE TABLE --------------------------------------------------------
    print("[3/5] Building node (facility) attributes ...")
    out_v = legs.groupby("source_center").agg(
        out_volume=("delay_ratio", "size"),
        out_corridors=("destination_center", "nunique"),
        out_median_ratio=("delay_ratio", "median"),
        out_excess_min=("delay_min", "sum"),
        out_breach=("sla_breach", "sum"),
    )
    in_v = legs.groupby("destination_center").agg(
        in_volume=("delay_ratio", "size"),
        in_corridors=("source_center", "nunique"),
        in_median_ratio=("delay_ratio", "median"),
        in_excess_min=("delay_min", "sum"),
        in_breach=("sla_breach", "sum"),
    )
    name_map = pd.concat([
        legs[["source_center", "source_name", "src_city", "src_state"]]
            .rename(columns={"source_center": "center", "source_name": "name",
                             "src_city": "city", "src_state": "state"}),
        legs[["destination_center", "destination_name", "dst_city", "dst_state"]]
            .rename(columns={"destination_center": "center", "destination_name": "name",
                             "dst_city": "city", "dst_state": "state"}),
    ]).drop_duplicates("center").set_index("center")

    nodes = name_map.join(out_v, how="outer").join(in_v, how="outer")
    # fill numeric NaNs with 0, string NaNs with "UNKNOWN" (keeps parquet types clean)
    str_cols = ["name", "city", "state"]
    num_cols = [c for c in nodes.columns if c not in str_cols]
    nodes[num_cols] = nodes[num_cols].fillna(0)
    nodes[str_cols] = nodes[str_cols].fillna("UNKNOWN").astype(str)
    nodes["total_volume"] = nodes["out_volume"] + nodes["in_volume"]
    nodes["total_excess_min"] = nodes["out_excess_min"] + nodes["in_excess_min"]
    nodes["total_breach"] = nodes["out_breach"] + nodes["in_breach"]
    nodes = nodes.reset_index().rename(columns={"index": "center"})
    print(f"      nodes (facilities) = {len(nodes):,}")

    # ---- DIRECTED GRAPH ----------------------------------------------------
    print("[4/5] Assembling networkx DiGraph ...")
    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        G.add_node(r["center"], name=r["name"], city=r["city"], state=r["state"],
                   total_volume=float(r["total_volume"]))
    for _, e in edges.iterrows():
        G.add_edge(
            e["source_center"], e["destination_center"],
            delay_ratio=float(e["median_delay_ratio"]),
            osrm_time=float(e["median_osrm"]),
            actual_time=float(e["median_actual"]),
            dist_km=float(e["median_dist_km"]),
            volume=int(e["n_traversals"]),
            excess_min=float(e["total_excess_min"]),
            # cost used for shortest-path / betweenness = physical free-flow time
            cost=float(max(e["median_osrm"], 0.1)),
        )
    print(f"      |V| = {G.number_of_nodes():,}   |E| = {G.number_of_edges():,}")
    # connectivity
    wcc = list(nx.weakly_connected_components(G))
    giant = max(wcc, key=len)
    print(f"      weakly-connected components = {len(wcc):,}  "
          f"(giant covers {len(giant)/G.number_of_nodes():.1%} of nodes)")

    # ---- SAVE --------------------------------------------------------------
    print("[5/5] Saving ...")
    edges.to_parquet(os.path.join(ROOT, "data", "edges.parquet"), index=False)
    nodes.to_parquet(os.path.join(ROOT, "data", "nodes.parquet"), index=False)
    with open(os.path.join(ROOT, "data", "graph.gpickle"), "wb") as f:
        pickle.dump(G, f)
    print("      wrote edges.parquet, nodes.parquet, graph.gpickle")


if __name__ == "__main__":
    main()
