"""
================================================================================
07 | IMPACT SCENARIO  -  what fixing the top-3 hubs is worth
================================================================================
Translates the graph findings into the two numbers the memo must quantify:
  (1) % reduction in late deliveries if the top-3 bottleneck hubs are upgraded
  (2) revenue-at-risk recovered

All business assumptions are explicit and configurable; the LOGIC is the point.
Outputs: outputs/tables/impact_scenario.csv  (+ console print used in the memo)
================================================================================
"""
import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data")
T = os.path.join(ROOT, "outputs", "tables")

# ---------------------- BUSINESS ASSUMPTIONS (tunable) ---------------------- #
OBS_DAYS = 25                 # dataset window
UPGRADE_EFFECT = 0.40         # an upgrade removes 40% of a hub's EXCESS delay
BREACH_X = 1.5                # "late" = actual > 1.5x OSRM (stricter, meaningful bar)
REV_AT_RISK_PER_LEG = 1800.0  # ₹ revenue-at-risk per delayed trunk-leg (consignment batch)
RECOVERY_FRACTION = 0.5       # fraction of at-risk revenue actually saved by fixing lateness
TOP3 = ["Gurgaon_Bilaspur_HB", "Bangalore_Nelmngla_H", "Bhiwandi_Mankoli_HB"]


def main():
    legs = pd.read_parquet(os.path.join(D, "legs.parquet"))
    met = pd.read_parquet(os.path.join(D, "node_metrics.parquet"))

    # identify top-3 hub centers by breach contribution (robust to name strings)
    met = met.sort_values("breach_contrib_pct", ascending=False)
    top3_centers = []
    for key in TOP3:
        hit = met[met["name"].str.startswith(key)]
        if len(hit):
            top3_centers.append(hit.iloc[0]["center"])
    top3_share = met[met["center"].isin(top3_centers)]["breach_contrib_pct"].sum()

    # legs that touch a top-3 hub (as source or destination)
    touch = legs[legs["source_center"].isin(top3_centers) |
                 legs["destination_center"].isin(top3_centers)].copy()

    # ---- current lateness (stricter 1.5x bar) -----------------------------
    legs["late"] = legs["delay_ratio"] > BREACH_X
    touch["late"] = touch["delay_ratio"] > BREACH_X
    net_late = int(legs["late"].sum())
    net_legs = len(legs)

    # ---- counterfactual: shrink EXCESS on top-3 legs by UPGRADE_EFFECT ----
    # new_actual = osrm + (1-effect)*excess ; recompute ratio + lateness
    excess = (touch["actual_time"] - touch["osrm_time"]).clip(lower=0)
    new_actual = touch["osrm_time"] + (1 - UPGRADE_EFFECT) * excess
    new_ratio = new_actual / touch["osrm_time"]
    new_late_touch = (new_ratio > BREACH_X)

    breaches_before_touch = int(touch["late"].sum())
    breaches_after_touch = int(new_late_touch.sum())
    breaches_avoided = breaches_before_touch - breaches_after_touch

    pct_reduction_network = 100.0 * breaches_avoided / net_late
    pct_reduction_on_touch = 100.0 * breaches_avoided / max(breaches_before_touch, 1)

    # ---- delay minutes recovered ------------------------------------------
    excess_recovered_min = float(UPGRADE_EFFECT * excess.sum())
    excess_recovered_min_yr = excess_recovered_min * (365.0 / OBS_DAYS)

    # ---- revenue at risk recovered ----------------------------------------
    rev_recovered_window = breaches_avoided * REV_AT_RISK_PER_LEG * RECOVERY_FRACTION
    rev_recovered_year = rev_recovered_window * (365.0 / OBS_DAYS)

    out = dict(
        top3_centers=top3_centers,
        top3_breach_share_pct=round(top3_share, 1),
        network_total_excess_min=int(met["total_excess_min"].sum() / 2 * 2),
        net_legs=net_legs, net_late_legs=net_late,
        breach_def=f"actual > {BREACH_X}x OSRM",
        upgrade_effect_pct=int(UPGRADE_EFFECT * 100),
        legs_touching_top3=len(touch),
        breaches_before_touch=breaches_before_touch,
        breaches_after_touch=breaches_after_touch,
        breaches_avoided=breaches_avoided,
        pct_reduction_in_late_network=round(pct_reduction_network, 1),
        pct_reduction_on_top3_corridors=round(pct_reduction_on_touch, 1),
        excess_min_recovered_window=int(excess_recovered_min),
        excess_hours_recovered_year=int(excess_recovered_min_yr / 60),
        rev_at_risk_per_leg=REV_AT_RISK_PER_LEG,
        recovery_fraction=RECOVERY_FRACTION,
        rev_recovered_window_inr=int(rev_recovered_window),
        rev_recovered_year_inr=int(rev_recovered_year),
    )
    pd.Series(out).to_csv(os.path.join(T, "impact_scenario.csv"))
    with open(os.path.join(D, "impact_scenario.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 72)
    print("IMPACT SCENARIO  -  upgrade top-3 hubs (Gurgaon, Bangalore, Bhiwandi)")
    print("=" * 72)
    print(f"Top-3 share of network excess delay      : {top3_share:.1f}%")
    print(f"Late-delivery definition                 : actual > {BREACH_X}x OSRM")
    print(f"Network late legs (window)               : {net_late:,} / {net_legs:,}")
    print(f"Legs touching a top-3 hub                : {len(touch):,}")
    print(f"  late before upgrade                    : {breaches_before_touch:,}")
    print(f"  late after  upgrade (-{int(UPGRADE_EFFECT*100)}% excess)        : {breaches_after_touch:,}")
    print(f"  breaches avoided                       : {breaches_avoided:,}")
    print(f"-> reduction in late deliveries (network): {pct_reduction_network:.1f}%")
    print(f"-> reduction on the top-3 corridors      : {pct_reduction_on_touch:.1f}%")
    print(f"Delay recovered                          : {excess_recovered_min/60:,.0f} hrs/window  "
          f"(~{excess_recovered_min_yr/60:,.0f} hrs/yr)")
    print(f"Revenue-at-risk recovered                : Rs {rev_recovered_window:,.0f}/window  "
          f"(~Rs {rev_recovered_year/1e7:,.2f} cr/yr)")
    print("\n(all business constants are explicit + configurable at top of file)")


if __name__ == "__main__":
    main()
