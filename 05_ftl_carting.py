"""
================================================================================
05 | FTL vs CARTING DECISION FRAMEWORK  -  Graph-Based Network Intelligence
================================================================================
GOAL
    A data-backed rule for choosing route type (FTL = dedicated full truck,
    Carting = shared / part-load feeder) per corridor, with the time-cost
    trade-off made explicit and conditioned on distance, lane volume, time-of-
    day and the SOURCE facility's graph position.

WHY WE MODEL THE DELAY RATIO (not absolute time)
    Route type is NOT randomly assigned: FTL is already used on long trunk lanes
    that are slow in absolute minutes, Carting on short feeders. Predicting
    absolute time with `is_ftl` flipped therefore confounds "where FTL is used"
    with "what FTL does". We instead model the DELAY RATIO (actual / OSRM),
    which normalises out distance, and reconstruct time = OSRM x ratio for a
    FIXED corridor. In this data FTL runs at ratio ~1.92 vs Carting ~2.15, i.e.
    FTL is genuinely more reliable on the same lane.

COST MODEL (explicit, Finance-tunable - the FRAMEWORK is the deliverable)
    Per leg dispatch:
      FTL     = FTL_FIX  + FTL_KM  * dist     # full truck: high fixed, efficient/km
      Carting = CART_FIX + CART_KM * dist     # shared feeder: low fixed, pricier/km
    Money break-even ~260 km, so FTL is cost-efficient on long hauls and Carting
    on short feeders; time + reliability value then shift the boundary shorter.

DECISION METRIC  (generalized cost per leg)
      GC = money_cost + VOT_PER_HR*time_hr + SLA_PENALTY*(1+3*src_betweenness)*P(breach)
    Recommend the lower-GC route type; the betweenness term makes reliability
    worth more when the source is a central hub feeding SLA-critical trunks.

OUTPUTS
    outputs/tables/ftl_carting_decision.csv
    outputs/tables/ftl_carting_summary.csv
    outputs/tables/ftl_carting_by_volume.csv
    data/ftl_artifacts.pkl
================================================================================
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data")
T = os.path.join(ROOT, "outputs", "tables")
OBS_DAYS = 25

# FTL = full dedicated truck: EFFICIENT per-km (economies of a full load) but a
# high fixed dispatch cost. Carting = shared/part-load feeder: cheap to inject
# into the network (low fixed) but PRICIER per-km (multiple handling, detours).
# => money break-even at dist* = (FTL_FIX-CART_FIX)/(CART_KM-FTL_KM) ~ 260 km,
#    so FTL is cost-efficient on long hauls, Carting on short feeders. Time +
#    reliability value (both favour FTL) then shift the boundary shorter on
#    SLA-critical / central-hub lanes. All figures Finance-tunable.
FTL_FIX, FTL_KM = 2500.0, 22.0
CART_FIX, CART_KM = 400.0, 30.0
VOT_PER_HR = 40.0                  # value of in-transit time (working capital + service speed)
SLA_PENALTY = 1500.0               # cost of a breached leg (penalty + churn / contract risk)


def ftl_cost(dist):
    return FTL_FIX + FTL_KM * dist


def cart_cost(dist):
    return CART_FIX + CART_KM * dist


def main():
    print("[1/6] Loading legs + graph metrics ...")
    legs = pd.read_parquet(os.path.join(D, "legs.parquet")).reset_index(drop=True)
    met = pd.read_parquet(os.path.join(D, "node_metrics.parquet"))
    btw = dict(zip(met["center"], met["betweenness"]))
    pr = dict(zip(met["center"], met["pagerank"]))

    legs["src_btw"] = legs["source_center"].map(btw).fillna(0.0)
    legs["src_pr"] = legs["source_center"].map(pr).fillna(0.0)
    legs["tod_code"] = legs["tod"].astype("category").cat.codes
    legs["log_dist"] = np.log1p(legs["osrm_distance"])

    feats = ["log_dist", "tod_code", "is_weekend", "src_btw", "src_pr", "is_ftl"]

    print("[2/6] Training counterfactual delay-ratio model ...")
    X = legs[feats]
    ratio_model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                               subsample=0.85, colsample_bytree=0.8,
                               random_state=SEED, n_jobs=4).fit(X, legs["delay_ratio"].values)

    print("[3/6] Training breach-probability model ...")
    breach_model = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                 subsample=0.85, colsample_bytree=0.8,
                                 random_state=SEED, n_jobs=4,
                                 eval_metric="logloss").fit(X, legs["sla_breach"].values)

    r_ftl_all = ratio_model.predict(legs[feats].assign(is_ftl=1)).mean()
    r_cart_all = ratio_model.predict(legs[feats].assign(is_ftl=0)).mean()
    print(f"      mean predicted ratio  FTL={r_ftl_all:.3f}  Carting={r_cart_all:.3f} "
          f"(FTL more reliable: {r_ftl_all < r_cart_all})")

    print("[4/6] Validating on corridors seen with BOTH route types ...")
    piv = (legs.groupby(["corridor", "route_type"])
           .agg(t=("actual_time", "median"), n=("actual_time", "size")).reset_index())
    both = piv.pivot(index="corridor", columns="route_type", values="t").dropna()
    bn = piv.pivot(index="corridor", columns="route_type", values="n").dropna()
    both = both[(bn["FTL"] >= 2) & (bn["Carting"] >= 2)]
    emp_delta = both["Carting"] - both["FTL"]
    emp_txt = (f"{len(both):,} dual-mode corridors; empirical median time saved by FTL "
               f"= {emp_delta.median():.1f} min, {(emp_delta > 0).mean():.0%} favour FTL")
    print("      " + emp_txt)

    print("[5/6] Scoring every corridor under both route types ...")
    edges = pd.read_parquet(os.path.join(D, "edges.parquet"))
    edges = edges[edges["stable"]].copy()
    edges["src_btw"] = edges["source_center"].map(btw).fillna(0.0)
    edges["src_pr"] = edges["source_center"].map(pr).fillna(0.0)
    edges["log_dist"] = np.log1p(edges["median_dist_km"])
    edges["tod_code"] = 1
    edges["is_weekend"] = 0
    edges["daily_vol"] = edges["n_traversals"] / OBS_DAYS
    dist = edges["median_dist_km"].values
    osrm = edges["median_osrm"].values

    def score(is_ftl):
        Xc = edges.assign(is_ftl=is_ftl)[feats]
        ratio = ratio_model.predict(Xc)
        time_min = osrm * ratio
        p = breach_model.predict_proba(Xc)[:, 1]
        money = ftl_cost(dist) if is_ftl else cart_cost(dist)
        btw_mult = 1.0 + 3.0 * edges["src_btw"].values
        gc = money + VOT_PER_HR * (time_min / 60.0) + SLA_PENALTY * btw_mult * p
        return time_min, p, np.asarray(money, float), gc

    t_ftl, p_ftl, c_ftl, gc_ftl = score(1)
    t_cart, p_cart, c_cart, gc_cart = score(0)

    dec = edges[["corridor", "src_name", "dst_name", "median_dist_km",
                 "n_traversals", "daily_vol", "src_btw"]].copy()
    dec["time_ftl"] = t_ftl.round(1)
    dec["time_cart"] = t_cart.round(1)
    dec["time_saved_ftl"] = (t_cart - t_ftl).round(1)
    dec["cost_ftl"] = c_ftl.round(0)
    dec["cost_cart"] = c_cart.round(0)
    dec["cost_premium_ftl"] = (c_ftl - c_cart).round(0)
    dec["pbreach_ftl"] = p_ftl.round(3)
    dec["pbreach_cart"] = p_cart.round(3)
    dec["gc_ftl"] = gc_ftl.round(0)
    dec["gc_cart"] = gc_cart.round(0)
    dec["recommend"] = np.where(gc_ftl <= gc_cart, "FTL", "Carting")
    dec["gc_saving_vs_alt"] = np.abs(gc_ftl - gc_cart).round(0)
    dec = dec.sort_values("median_dist_km")

    print("[6/6] Summarising trade-off + saving ...")
    dec["dist_band"] = pd.cut(dec["median_dist_km"], [0, 50, 150, 400, 800, 1e6],
                              labels=["<50km", "50-150km", "150-400km",
                                      "400-800km", ">800km"])
    by_dist = dec.groupby("dist_band").agg(
        corridors=("corridor", "size"),
        pct_recommend_ftl=("recommend", lambda s: 100 * (s == "FTL").mean()),
        avg_time_saved_ftl_min=("time_saved_ftl", "mean"),
        avg_cost_premium_ftl=("cost_premium_ftl", "mean")).round(1).reset_index()

    dec["vol_band"] = pd.cut(dec["daily_vol"], [0, 1.5, 4, 1e6],
                             labels=["thin(<1.5/d)", "medium(1.5-4/d)", "dense(>4/d)"])
    by_vol = dec.groupby("vol_band").agg(
        corridors=("corridor", "size"),
        pct_recommend_ftl=("recommend", lambda s: 100 * (s == "FTL").mean())
        ).round(1).reset_index()

    dec.round(3).to_csv(os.path.join(T, "ftl_carting_decision.csv"), index=False)
    by_dist.to_csv(os.path.join(T, "ftl_carting_summary.csv"), index=False)
    by_vol.to_csv(os.path.join(T, "ftl_carting_by_volume.csv"), index=False)
    with open(os.path.join(D, "ftl_artifacts.pkl"), "wb") as f:
        pickle.dump(dict(ratio_model=ratio_model, breach_model=breach_model,
                         feats=feats, emp_txt=emp_txt, assumptions=dict(
                             FTL_FIX=FTL_FIX, FTL_KM=FTL_KM, CART_FIX=CART_FIX,
                             CART_KM=CART_KM,
                             VOT_PER_HR=VOT_PER_HR, SLA_PENALTY=SLA_PENALTY,
                             OBS_DAYS=OBS_DAYS)), f)

    print("\n" + "=" * 78)
    print("FTL vs CARTING  -  trade-off by DISTANCE band")
    print("=" * 78)
    print(by_dist.to_string(index=False))
    print("\nby LANE-VOLUME band:")
    print(by_vol.to_string(index=False))
    print(f"\nOverall: framework recommends FTL on "
          f"{100*(dec['recommend']=='FTL').mean():.0f}% of stable corridors "
          f"({(dec['recommend']=='FTL').sum():,}/{len(dec):,}).")


if __name__ == "__main__":
    main()
