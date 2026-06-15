"""
================================================================================
01 | DATA PIPELINE  -  Delhivery Graph-Based Network Intelligence
================================================================================
PURPOSE
    Parse and merge the raw Delhivery trip-scan data into a clean, leg-level
    "corridor traversal" table that is the atomic unit for everything downstream
    (graph construction, bottleneck audit, ETA models, FTL/Carting framework).

DATA MODEL (how the raw file is structured)
    - Each row is ONE scan event inside a trip.
    - A `trip_uuid` is a full physical journey and is composed of one or more
      legs. A leg is a single hop: (source_center -> destination_center).
    - Inside a leg, `actual_time`, `osrm_time`, `actual_distance_to_destination`
      and `osrm_distance` are CUMULATIVE and reset at the start of each new leg.
      => The row with the maximum cumulative `actual_time` for a given
         (trip_uuid, source_center, destination_center) group is the LEG TOTAL.
    - `factor` == actual_time / osrm_time (the delay ratio) at leg level.
    - `segment_*` columns are per-scan increments (not used for leg totals).

OUTPUT
    data/legs.parquet      one row per corridor traversal (a leg of a trip)
================================================================================
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "delivery_data.csv")
OUT = os.path.join(ROOT, "data", "legs.parquet")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_location(name: str):
    """'Anand_VUNagar_DC (Gujarat)' -> ('Anand', 'Gujarat')."""
    if pd.isna(name):
        return (np.nan, np.nan)
    state = np.nan
    if "(" in name and ")" in name:
        state = name[name.rfind("(") + 1 : name.rfind(")")].strip()
    city = name.split("_")[0].strip()
    return (city, state)


def time_of_day_bucket(hour: float):
    """Operational shift buckets used for stratification."""
    if pd.isna(hour):
        return "unknown"
    h = int(hour)
    if 0 <= h < 6:
        return "night"        # 00:00-05:59
    if 6 <= h < 12:
        return "morning"      # 06:00-11:59
    if 12 <= h < 18:
        return "afternoon"    # 12:00-17:59
    return "evening"          # 18:00-23:59


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def main():
    print("[1/6] Loading raw scan data ...")
    df = pd.read_csv(RAW)
    print(f"      raw rows = {len(df):,}  |  trips = {df['trip_uuid'].nunique():,}")

    # --- parse timestamps ---------------------------------------------------
    for c in ["trip_creation_time", "od_start_time", "od_end_time"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # --- collapse scans -> leg totals --------------------------------------
    # Within each (trip, source, dest) leg, cumulative columns peak on the last
    # scan. Take the row with the max cumulative actual_time as the leg total.
    print("[2/6] Collapsing scans into leg totals ...")
    df = df.sort_values(["trip_uuid", "source_center", "destination_center",
                         "actual_time"])
    keys = ["trip_uuid", "source_center", "destination_center"]
    legs = df.groupby(keys, as_index=False).agg(
        data=("data", "first"),
        route_type=("route_type", "first"),
        trip_creation_time=("trip_creation_time", "first"),
        od_start_time=("od_start_time", "first"),
        od_end_time=("od_end_time", "last"),
        source_name=("source_name", "first"),
        destination_name=("destination_name", "first"),
        actual_time=("actual_time", "max"),          # cumulative leg total (min)
        osrm_time=("osrm_time", "max"),              # cumulative leg total (min)
        actual_distance=("actual_distance_to_destination", "max"),  # km
        osrm_distance=("osrm_distance", "max"),      # km
        start_scan_to_end_scan=("start_scan_to_end_scan", "first"),  # wall-clock
        n_scans=("actual_time", "size"),
    )
    print(f"      leg rows = {len(legs):,}")

    # --- clean --------------------------------------------------------------
    print("[3/6] Cleaning ...")
    before = len(legs)
    legs = legs[legs["source_center"] != legs["destination_center"]]      # no self-loops
    legs = legs[(legs["osrm_time"] > 0) & (legs["actual_time"] > 0)]      # valid times
    legs = legs[(legs["osrm_distance"] > 0)]                              # valid distance
    # remove extreme physically-impossible ratios (data artefacts)
    legs["delay_ratio"] = legs["actual_time"] / legs["osrm_time"]
    legs = legs[(legs["delay_ratio"] >= 0.2) & (legs["delay_ratio"] <= 20)]
    print(f"      removed {before - len(legs):,} bad rows -> {len(legs):,} clean legs")

    # --- feature engineering -----------------------------------------------
    print("[4/6] Engineering features ...")
    legs["src_city"], legs["src_state"] = zip(*legs["source_name"].map(parse_location))
    legs["dst_city"], legs["dst_state"] = zip(*legs["destination_name"].map(parse_location))

    legs["start_hour"] = legs["od_start_time"].dt.hour
    legs["dow"] = legs["od_start_time"].dt.dayofweek
    legs["is_weekend"] = legs["dow"].isin([5, 6]).astype(int)
    legs["tod"] = legs["start_hour"].map(time_of_day_bucket)
    legs["month"] = legs["od_start_time"].dt.month

    # corridor identifier
    legs["corridor"] = legs["source_center"] + " -> " + legs["destination_center"]
    legs["is_ftl"] = (legs["route_type"] == "FTL").astype(int)

    # delay measures
    legs["delay_min"] = legs["actual_time"] - legs["osrm_time"]            # excess minutes
    legs["sla_breach"] = (legs["delay_ratio"] > 1.20).astype(int)         # >20% over OSRM
    legs["dwell_min"] = (legs["start_scan_to_end_scan"] - legs["actual_time"]).clip(lower=0)
    legs["speed_kmph_osrm"] = legs["osrm_distance"] / (legs["osrm_time"] / 60.0)
    legs["log_actual_time"] = np.log1p(legs["actual_time"])

    # --- summary ------------------------------------------------------------
    print("[5/6] Sanity summary ...")
    print(f"      facilities (nodes)  : src={legs['source_center'].nunique():,}  "
          f"dst={legs['destination_center'].nunique():,}")
    print(f"      distinct corridors  : {legs['corridor'].nunique():,}")
    print(f"      route mix           : {legs['route_type'].value_counts().to_dict()}")
    print(f"      median delay ratio  : {legs['delay_ratio'].median():.3f}")
    print(f"      SLA breach rate     : {legs['sla_breach'].mean():.1%}")
    print(f"      date range          : {legs['od_start_time'].min()}  ->  "
          f"{legs['od_start_time'].max()}")

    print("[6/6] Saving ...")
    legs.to_parquet(OUT, index=False)
    print(f"      wrote {OUT}  ({len(legs):,} rows, {legs.shape[1]} cols)")
    return legs


if __name__ == "__main__":
    main()
