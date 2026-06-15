"""
================================================================================
  Delhivery Graph-Based Network Intelligence — Live Dashboard
  Run with:   streamlit run app.py
================================================================================
A lightweight operations console over the analysis artifacts:
  - network KPIs and the structural OSRM gap
  - interactive bottleneck-hub explorer with delay-risk scores
  - corridor delay lookup
  - live ETA prediction (graph-enhanced model) for any corridor profile
  - FTL vs Carting recommendation lookup
All artifacts are produced by src/01..07; this file only reads them.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
T = os.path.join(HERE, "outputs", "tables")
F = os.path.join(HERE, "outputs", "figures")

st.set_page_config(page_title="Delhivery Network Intelligence", layout="wide",
                   page_icon="🚚")

CITY_LL = {
    "Gurgaon": (28.46, 77.03), "Delhi": (28.61, 77.21), "Sonipat": (28.99, 77.02),
    "Chandigarh": (30.73, 76.78), "Ludhiana": (30.90, 75.86), "Jaipur": (26.91, 75.79),
    "Bangalore": (12.97, 77.59), "Bengaluru": (12.97, 77.59), "Hyderabad": (17.38, 78.49),
    "Bhiwandi": (19.30, 73.06), "Mumbai": (19.08, 72.88), "Pune": (18.52, 73.86),
    "Kolkata": (22.57, 88.36), "Bhubaneshwar": (20.30, 85.82), "Guwahati": (26.14, 91.74),
    "Bhopal": (23.26, 77.41), "Kanpur": (26.45, 80.33), "Ahmedabad": (23.02, 72.57),
    "MAA": (13.08, 80.27), "Surat": (21.17, 72.83), "Ranchi": (23.34, 85.31),
    "Muzaffrpur": (26.12, 85.39), "Purnia": (25.78, 87.47), "Vijayawada": (16.51, 80.65),
}


@st.cache_data
def load():
    met = pd.read_parquet(os.path.join(D, "node_metrics.parquet"))
    edges = pd.read_parquet(os.path.join(D, "edges.parquet"))
    comp = pd.read_csv(os.path.join(T, "model_comparison.csv"))
    dec = pd.read_csv(os.path.join(T, "ftl_carting_decision.csv"))
    impact = json.load(open(os.path.join(D, "impact_scenario.json")))
    return met, edges, comp, dec, impact


@st.cache_resource
def load_models():
    eta = pickle.load(open(os.path.join(D, "eta_artifacts.pkl"), "rb"))
    ftl = pickle.load(open(os.path.join(D, "ftl_artifacts.pkl"), "rb"))
    return eta, ftl


met, edges, comp, dec, impact = load()

st.title("🚚 Delhivery — Graph-Based Network Intelligence")
st.caption("Operations console · bottleneck hubs, corridor delay risk, ETA prediction, route-type decisions")

# ---- KPI row --------------------------------------------------------------
k = st.columns(4)
total_excess_hr = met["total_excess_min"].sum() / 60
k[0].metric("Facilities (nodes)", f"{len(met):,}")
k[1].metric("Corridors (edges)", f"{len(edges):,}")
k[2].metric("Network excess delay", f"{total_excess_hr/1000:,.0f}k hrs",
            help="Total actual-minus-OSRM minutes over the ~25-day window")
graph_row = comp[comp.model.str.contains("graph")].iloc[0]
osrm_row = comp[comp.model.str.contains("OSRM")].iloc[0]
k[3].metric("On-target ETA (graph model)", f"{graph_row.HIT15:.0f}%",
            delta=f"{graph_row.HIT15 - osrm_row.HIT15:+.0f} pts vs OSRM")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔴 Bottleneck hubs", "🛣️ Corridor delay", "⏱️ ETA predictor", "🚛 FTL vs Carting"])

# ===========================================================================
with tab1:
    st.subheader("Bottleneck hubs ranked by share of network delay")
    n = st.slider("How many hubs to show", 5, 40, 15)
    cols = ["name", "state", "breach_contrib_pct", "betweenness", "pagerank",
            "in_degree", "out_degree", "total_volume", "out_delay_ratio"]
    top = met.sort_values("breach_contrib_pct", ascending=False).head(n)[cols].copy()
    top.columns = ["Facility", "State", "Delay share %", "Betweenness", "PageRank",
                   "In-deg", "Out-deg", "Volume", "Delay ×"]
    st.dataframe(top.round(3), use_container_width=True, hide_index=True)

    # map with delay-risk scores
    top2 = met.sort_values("breach_contrib_pct", ascending=False).head(25).copy()
    top2["city2"] = top2["city"].apply(lambda x: str(x).split("_")[0])
    top2 = top2[top2["city2"].isin(CITY_LL)].drop_duplicates("city2")
    top2["lat"] = top2["city2"].map(lambda c: CITY_LL[c][0])
    top2["lon"] = top2["city2"].map(lambda c: CITY_LL[c][1])
    top2["size"] = 30000 + top2["breach_contrib_pct"] * 12000
    st.map(top2[["lat", "lon", "size"]], size="size", color="#c1440e")

# ===========================================================================
with tab2:
    st.subheader("Corridor delay lookup")
    stable = edges[edges["stable"]].copy()
    states = sorted(set(stable["src_state"].dropna()) | set(stable["dst_state"].dropna()))
    c1, c2 = st.columns(2)
    s_from = c1.selectbox("Source state (optional)", ["(any)"] + states)
    s_to = c2.selectbox("Destination state (optional)", ["(any)"] + states)
    view = stable
    if s_from != "(any)":
        view = view[view["src_state"] == s_from]
    if s_to != "(any)":
        view = view[view["dst_state"] == s_to]
    view = view.sort_values("total_excess_min", ascending=False).head(50)
    show = view[["src_name", "dst_name", "median_delay_ratio", "median_actual",
                 "median_osrm", "median_dist_km", "n_traversals", "total_excess_min"]].copy()
    show.columns = ["From", "To", "Delay ×", "Actual min", "OSRM min", "Dist km",
                    "Trips", "Excess min"]
    st.dataframe(show.round(1), use_container_width=True, hide_index=True)
    st.caption("Sorted by total excess delay minutes (the corridors worth fixing first).")

# ===========================================================================
with tab3:
    st.subheader("Graph-enhanced ETA predictor")
    st.caption("Estimate a leg's true delivery time using corridor history + graph position.")
    eta, ftl = load_models()
    c = st.columns(4)
    osrm_time = c[0].number_input("OSRM time (min)", 10.0, 5000.0, 300.0, step=10.0)
    osrm_dist = c[1].number_input("OSRM distance (km)", 1.0, 3000.0, 250.0, step=10.0)
    hour = c[2].slider("Dispatch hour", 0, 23, 9)
    is_ftl = c[3].selectbox("Route type", ["FTL", "Carting"]) == "FTL"

    # use corridor-agnostic priors (global) for an ad-hoc estimate
    prior_ratio = eta["glob_ratio"]
    prior_actual = osrm_time * prior_ratio
    speed = osrm_dist / (osrm_time / 60.0) if osrm_time > 0 else 0
    naive = osrm_time
    quick = osrm_time * prior_ratio  # history-anchored estimate
    cc = st.columns(3)
    cc[0].metric("OSRM says", f"{naive:.0f} min")
    cc[1].metric("History-anchored ETA", f"{quick:.0f} min",
                 delta=f"{quick - naive:+.0f} min vs OSRM")
    cc[2].metric("Implied delay factor", f"{prior_ratio:.2f}×")
    st.info("The full model also consumes node2vec embeddings + betweenness of the "
            "specific source/destination facilities; this quick estimate uses the "
            "global corridor-history prior for any unseen profile.")

# ===========================================================================
with tab4:
    st.subheader("FTL vs Carting recommendation")
    dist_in = st.slider("Corridor distance (km)", 5, 2000, 300)
    band = ("<50km" if dist_in < 50 else "50-150km" if dist_in < 150 else
            "150-400km" if dist_in < 400 else "400-800km" if dist_in < 800 else ">800km")
    sub = dec[(dec["median_dist_km"] >= dist_in * 0.7) &
              (dec["median_dist_km"] <= dist_in * 1.3)]
    rec = "FTL" if dist_in >= 400 else ("Carting" if dist_in < 150 else "Depends on hub centrality")
    cset = st.columns(3)
    cset[0].metric("Distance band", band)
    cset[1].metric("Default recommendation", rec)
    if len(sub):
        cset[2].metric("Avg time saved by FTL", f"{sub['time_saved_ftl'].mean():.0f} min")
    st.markdown(
        "- **Long-haul (>400 km):** FTL — cheaper per leg once full, saves hours.\n"
        "- **Short feeders (<150 km):** Carting — FTL premium not justified.\n"
        "- **150–400 km:** shift to FTL when the source is a central hub on an SLA-critical trunk.")
    summ = pd.read_csv(os.path.join(T, "ftl_carting_summary.csv"))
    st.dataframe(summ, use_container_width=True, hide_index=True)

st.divider()
st.caption("Built on a directed weighted graph of 1,657 facilities / 2,775 corridors. "
           "Artifacts generated by the src/ pipeline (modules 01–07).")
