"""
================================================================================
04 | GRAPH-ENHANCED ETA PREDICTION  -  Graph-Based Network Intelligence
================================================================================
GOAL
    Predict leg ETA (actual_time, minutes) and prove a graph-enhanced model
    beats a trip-features-only baseline on BOTH:
        - MAE  (mean absolute error, minutes)
        - HIT@15%  (share of legs whose predicted ETA is within 15% of actual)

DESIGN (leakage-safe)
    - 80/20 random split (seed fixed).
    - ALL corridor / graph features are learned on the TRAIN fold only and then
      merged onto test.  Unseen test corridors fall back to state-pair, then
      global priors (a realistic cold-start treatment).
    - node2vec embeddings + centrality are computed on the TRAIN graph only.

MODELS
    baseline      : XGBoost on trip-level features (osrm time/dist, route type,
                    time-of-day, weekday, free-flow speed).
    graph_enhanced: baseline features
                    + source/dest betweenness, pagerank, degree
                    + corridor train-prior (median ratio, median actual)
                    + source & dest node2vec embeddings.
    (a Ridge linear baseline is also reported as a sanity reference.)

OUTPUTS
    data/eta_predictions.parquet         test rows + both models' predictions
    outputs/tables/model_comparison.csv  headline metrics table
================================================================================
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from node2vec import Node2Vec

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data")
T = os.path.join(ROOT, "outputs", "tables")
EMB_DIM = 24


# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.clip(np.asarray(y_pred, float), 1e-6, None)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ape = np.abs(y_true - y_pred) / y_true
    hit15 = np.mean(ape <= 0.15)
    hit25 = np.mean(ape <= 0.25)
    return dict(MAE=mae, RMSE=rmse, HIT15=hit15, HIT25=hit25)


def build_train_graph(train: pd.DataFrame) -> nx.DiGraph:
    """Directed graph from TRAIN legs only (median osrm cost per corridor)."""
    agg = train.groupby(["source_center", "destination_center"]).agg(
        cost=("osrm_time", "median"), volume=("osrm_time", "size")).reset_index()
    G = nx.DiGraph()
    for _, r in agg.iterrows():
        G.add_edge(r["source_center"], r["destination_center"],
                   cost=float(max(r["cost"], 0.1)), volume=int(r["volume"]))
    return G


def node2vec_embeddings(G: nx.DiGraph, dim=EMB_DIM):
    print("      training node2vec ...")
    n2v = Node2Vec(G, dimensions=dim, walk_length=15, num_walks=10,
                   workers=1, seed=SEED, quiet=True, weight_key="volume")
    model = n2v.fit(window=5, min_count=1, batch_words=128, seed=SEED)
    emb = {n: model.wv[str(n)] for n in G.nodes()}
    return emb


def main():
    print("[1/7] Loading legs ...")
    legs = pd.read_parquet(os.path.join(D, "legs.parquet")).reset_index(drop=True)

    base_num = ["osrm_time", "osrm_distance", "actual_distance",
                "speed_kmph_osrm", "start_hour", "dow", "is_weekend", "is_ftl"]
    legs["tod_code"] = legs["tod"].astype("category").cat.codes
    base_num += ["tod_code"]
    legs = legs.dropna(subset=base_num + ["actual_time"]).reset_index(drop=True)

    # ---- split -------------------------------------------------------------
    print("[2/7] Train/test split (80/20) ...")
    train, test = train_test_split(legs, test_size=0.20, random_state=SEED)
    print(f"      train={len(train):,}  test={len(test):,}")

    # ---- TRAIN-ONLY graph features ----------------------------------------
    print("[3/7] Building TRAIN graph + centrality ...")
    G = build_train_graph(train)
    btw = nx.betweenness_centrality(G, weight="cost", normalized=True)
    pr = nx.pagerank(G, weight="volume")
    outdeg = dict(G.out_degree())
    indeg = dict(G.in_degree())

    print("[4/7] node2vec embeddings on TRAIN graph ...")
    emb = node2vec_embeddings(G)
    zero = np.zeros(EMB_DIM, dtype=float)

    # corridor train priors (median delay ratio + median actual) -- TRAIN ONLY
    corr_prior = train.groupby("corridor").agg(
        prior_ratio=("delay_ratio", "median"),
        prior_actual=("actual_time", "median"),
        prior_n=("delay_ratio", "size")).reset_index()
    state_prior = train.groupby(["src_state", "dst_state"]).agg(
        sp_ratio=("delay_ratio", "median"),
        sp_actual=("actual_time", "median")).reset_index()
    glob_ratio = train["delay_ratio"].median()
    glob_actual = train["actual_time"].median()

    # ---- feature assembly --------------------------------------------------
    def featurize(df):
        df = df.merge(corr_prior, on="corridor", how="left")
        df = df.merge(state_prior, on=["src_state", "dst_state"], how="left")
        # cold-start fallback: corridor -> state-pair -> global
        df["prior_ratio"] = df["prior_ratio"].fillna(df["sp_ratio"]).fillna(glob_ratio)
        df["prior_actual"] = df["prior_actual"].fillna(df["sp_actual"]).fillna(glob_actual)
        df["prior_n"] = df["prior_n"].fillna(0)

        df["src_btw"] = df["source_center"].map(btw).fillna(0.0)
        df["dst_btw"] = df["destination_center"].map(btw).fillna(0.0)
        df["src_pr"] = df["source_center"].map(pr).fillna(0.0)
        df["dst_pr"] = df["destination_center"].map(pr).fillna(0.0)
        df["src_outdeg"] = df["source_center"].map(outdeg).fillna(0)
        df["dst_indeg"] = df["destination_center"].map(indeg).fillna(0)

        graph_scalar = ["prior_ratio", "prior_actual", "prior_n",
                        "src_btw", "dst_btw", "src_pr", "dst_pr",
                        "src_outdeg", "dst_indeg"]

        # node2vec source + dest vectors
        se = np.vstack(df["source_center"].map(lambda x: emb.get(x, zero)).values)
        de = np.vstack(df["destination_center"].map(lambda x: emb.get(x, zero)).values)
        emb_cols_s = [f"se{i}" for i in range(EMB_DIM)]
        emb_cols_d = [f"de{i}" for i in range(EMB_DIM)]
        df[emb_cols_s] = se
        df[emb_cols_d] = de
        return df, graph_scalar, emb_cols_s + emb_cols_d

    train, gscalar, gemb = featurize(train)
    test, _, _ = featurize(test)

    Xb_tr, Xb_te = train[base_num], test[base_num]
    Xg_tr = train[base_num + gscalar + gemb]
    Xg_te = test[base_num + gscalar + gemb]
    y_tr = train["actual_time"].values
    y_te = test["actual_time"].values
    # log target stabilises the skew; predict in log, evaluate in minutes
    yl_tr = np.log1p(y_tr)

    # ---- models ------------------------------------------------------------
    print("[5/7] Training models ...")
    xgb_kw = dict(n_estimators=600, max_depth=7, learning_rate=0.05,
                  subsample=0.85, colsample_bytree=0.8, min_child_weight=3,
                  random_state=SEED, n_jobs=4)

    # Ridge linear reference (on baseline features, standardized)
    sc = StandardScaler().fit(Xb_tr)
    ridge = Ridge(alpha=5.0).fit(sc.transform(Xb_tr), yl_tr)
    pred_ridge = np.expm1(ridge.predict(sc.transform(Xb_te)))

    base_model = XGBRegressor(**xgb_kw).fit(Xb_tr, yl_tr)
    pred_base = np.expm1(base_model.predict(Xb_te))

    graph_model = XGBRegressor(**xgb_kw).fit(Xg_tr, yl_tr)
    pred_graph = np.expm1(graph_model.predict(Xg_te))

    # naive OSRM as floor reference (what they use today)
    pred_osrm = test["osrm_time"].values

    # ---- evaluate ----------------------------------------------------------
    print("[6/7] Evaluating ...")
    rows = []
    for name, pred in [("OSRM (current)", pred_osrm),
                       ("Ridge baseline", pred_ridge),
                       ("XGB baseline (trip features)", pred_base),
                       ("XGB graph-enhanced", pred_graph)]:
        m = metrics(y_te, pred)
        m["model"] = name
        rows.append(m)
    comp = pd.DataFrame(rows)[["model", "MAE", "RMSE", "HIT15", "HIT25"]]
    comp["MAE"] = comp["MAE"].round(2)
    comp["RMSE"] = comp["RMSE"].round(2)
    comp["HIT15"] = (comp["HIT15"] * 100).round(1)
    comp["HIT25"] = (comp["HIT25"] * 100).round(1)

    # ---- feature importance (graph model) ---------------------------------
    imp = pd.Series(graph_model.feature_importances_,
                    index=Xg_tr.columns).sort_values(ascending=False)
    # collapse embedding dims into one bucket for readability
    emb_imp = imp[[c for c in imp.index if c.startswith(("se", "de"))]].sum()
    imp_named = imp[[c for c in imp.index if not c.startswith(("se", "de"))]]
    imp_named["node2vec_embeddings(all)"] = emb_imp
    imp_named = imp_named.sort_values(ascending=False)

    # ---- save --------------------------------------------------------------
    print("[7/7] Saving ...")
    out = test[["corridor", "source_center", "destination_center", "route_type",
                "tod", "osrm_time", "actual_time"]].copy()
    out["pred_baseline"] = pred_base
    out["pred_graph"] = pred_graph
    out.to_parquet(os.path.join(D, "eta_predictions.parquet"), index=False)
    comp.to_csv(os.path.join(T, "model_comparison.csv"), index=False)
    imp_named.round(4).to_csv(os.path.join(T, "graph_feature_importance.csv"))

    # persist artefacts for the FTL module + dashboard
    with open(os.path.join(D, "eta_artifacts.pkl"), "wb") as f:
        pickle.dump(dict(btw=btw, pr=pr, outdeg=outdeg, indeg=indeg,
                         corr_prior=corr_prior, state_prior=state_prior,
                         glob_ratio=glob_ratio, glob_actual=glob_actual,
                         base_num=base_num), f)

    # ---- report ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL COMPARISON  (test set, n=%d)" % len(test))
    print("=" * 70)
    print(comp.to_string(index=False))
    b = comp[comp.model == "XGB baseline (trip features)"].iloc[0]
    g = comp[comp.model == "XGB graph-enhanced"].iloc[0]
    print("\nGRAPH ADVANTAGE (measured, not claimed):")
    print(f"  MAE     : {b.MAE:.2f} -> {g.MAE:.2f} min "
          f"({100*(b.MAE-g.MAE)/b.MAE:+.1f}%)")
    print(f"  HIT@15% : {b.HIT15:.1f}% -> {g.HIT15:.1f}% "
          f"({g.HIT15-b.HIT15:+.1f} pts)")
    print("\nTop graph-model features:")
    print(imp_named.head(12).round(4).to_string())


if __name__ == "__main__":
    main()
