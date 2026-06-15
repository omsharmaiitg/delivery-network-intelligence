# Delhivery — Graph-Based Network Intelligence

Optimizing delivery ETAs and surfacing bottleneck hubs by modelling Delhivery's
logistics network as a **directed weighted graph** (facilities = nodes, corridors
= edges). End-to-end: data pipeline → graph construction → bottleneck audit →
graph-enhanced ETA model → FTL/Carting decision framework → strategy memo +
dashboard.

---

## Headline findings

| # | Finding | Number |
|---|---------|--------|
| 1 | OSRM (today's engine) underestimates actual leg time | **~2.0× median** (95% of legs run >20% over) |
| 2 | Delay is concentrated, not spread | **Top 5 hubs = 30.8%** of all network delay; top 3 = 24.3% |
| 3 | #1 bottleneck | **Gurgaon (Bilaspur HB)** — 12.5% of network delay, betweenness 0.19 |
| 4 | Worst corridor | **Gurgaon ⇄ Bangalore** — ~145k excess minutes in 3 weeks |
| 5 | Graph ETA model vs OSRM | on-target (±15%) **58% vs 4%**; MAE **28 vs 108 min** |
| 6 | Graph model vs trip-only baseline | MAE **−18.7%**, on-target **+13.2 pts** (measured, held-out) |
| 7 | FTL/Carting rule | FTL on 100% of >400 km lanes; Carting on short feeders |
| 8 | Upgrade top-3 hubs (−40% excess) | **−28% late deliveries** on their corridors; **~₹1.5 cr/yr** recovered |

---

## How to reproduce

```bash
pip install -r requirements.txt          # numpy pinned <2 for gensim/node2vec

python src/01_data_pipeline.py           # raw scans  -> data/legs.parquet
python src/02_graph_construction.py      # legs       -> edges/nodes/graph
python src/03_bottleneck_audit.py        # centrality + SLA-breach contribution
python src/04_eta_models.py              # baseline vs graph-enhanced ETA
python src/05_ftl_carting.py             # route-type decision framework
python src/06_visualizations.py          # all figures -> outputs/figures/
python src/07_impact_scenario.py         # top-3 upgrade impact numbers
node   src/make_memo.js                  # outputs/Network_Operations_Strategy_Memo.docx

streamlit run app.py                     # optional live dashboard
```

Modules are ordered and each writes artifacts the next one reads. Fixed
`SEED = 42` throughout.

---

## Project structure

```
delhivery_project/
├── src/
│   ├── 01_data_pipeline.py        scans -> clean leg-level corridor table
│   ├── 02_graph_construction.py   directed weighted graph + stratified edge weights
│   ├── 03_bottleneck_audit.py     betweenness / pagerank / degree / clustering + breach share
│   ├── 04_eta_models.py           XGBoost baseline vs node2vec graph-enhanced (leakage-safe)
│   ├── 05_ftl_carting.py          counterfactual delay-ratio model + generalized-cost decision
│   ├── 06_visualizations.py       6 publication figures
│   ├── 07_impact_scenario.py      quantified top-3 hub upgrade impact
│   └── make_memo.js               strategy memo (docx)
├── data/                          parquet/pickle artifacts (generated)
├── outputs/
│   ├── figures/                   fig1..fig6 png
│   ├── tables/                    hub ranking, chronic corridors, model comparison, FTL table, impact
│   └── Network_Operations_Strategy_Memo.docx
├── app.py                         Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Methodology notes (why these choices)

**Data model.** Each raw row is a scan; cumulative `actual_time`/`osrm_time`
reset per leg, so the max-cumulative row of each `(trip, source, dest)` group is
the leg total. Corridors = `(source → dest)`; delay ratio = `actual / osrm`.

**Edge weights.** Median delay ratio per corridor, persisted stratified by route
type (`ratio_ftl`, `ratio_carting`) and time of day (`ratio_morning…`), per the
brief. Median (not mean) for robustness to the long right tail.

**Bottleneck score.** Real bottlenecks are *structural* (betweenness, pagerank)
**and** *high-throughput* **and** *slow*. Hubs are ranked primarily by their
**share of total network excess-delay minutes** (a direct, business-readable
metric), with centrality explaining *why*. Because the literal ">20% over OSRM"
breach is near-universal (95%), corridors are ranked by **severity × volume**
(total excess minutes), not the binary flag.

**ETA model.** Target = `log1p(actual_time)`. Baseline uses trip features only.
Graph-enhanced adds source/dest betweenness, pagerank, degree, **corridor history
priors**, and **node2vec embeddings**. All corridor/graph features are learned on
the **train fold only** and merged to test with corridor → state-pair → global
cold-start fallback; node2vec + centrality are computed on the **train graph
only**. The "graph advantage" is therefore measured, not leaked.

**FTL vs Carting.** Route type isn't randomly assigned (FTL already runs the long
slow trunks), so we model the **delay ratio** (distance-normalised) counterfactually
rather than absolute time. Decision = generalized cost
`money + value-of-time + SLA-penalty·(1+3·betweenness)·P(breach)`; the betweenness
term makes reliability worth more at central hubs. Cost constants are explicit and
**Finance-tunable** — the framework is the deliverable, not the exact rupees.

**Business assumptions** (impact scenario, all in `src/07`): an upgrade removes
40% of a hub's excess delay; "late" = actual > 1.5× OSRM; revenue-at-risk per
delayed trunk-leg and recovery fraction are configurable constants.

---

## Environment

Python 3.12. Key libs: pandas, numpy (**1.26.x**, pinned for gensim), scikit-learn,
networkx, xgboost, gensim + node2vec, matplotlib. Node.js for the memo (`docx`).
`GraphSAGE` was substituted with **node2vec** (no torch-geometric dependency); the
brief allows "GraphSAGE/node2vec".

Dataset: public Delhivery feature-engineering dataset, ~144.9k scans / 14.8k trips,
Sep–Oct 2018 (~25-day window).
