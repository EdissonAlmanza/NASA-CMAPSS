# NASA CMAPSS Predictive Maintenance — Project‑Based Learning Plan (Sprints 0–5)

**Goal:** Build a portfolio‑ready, production‑grade predictive maintenance project using NASA CMAPSS (turbofan engine) data, progressing from BI storytelling → baselines → deep learning with alerts → MLOps & optimization → streaming digital twin.

**Audience fit (you):** BI + operations + supply chain background; wants Upwork differentiation with solid engineering + business value.

**Time assumptions:** 12–15 h/week. Target duration ≈ 10–12 weeks (can compress or stretch). Each sprint below can be 1–2 weeks.

**Deliverables rhythm:** Every sprint ends with a demo, a case‑study paragraph, screenshots, and a repo tag/release.

---

## Stack & Repo (used across all sprints)

- **Languages/Libs:** Python 3.11, Pandas, Scikit‑learn, XGBoost/LightGBM, PyTorch/Keras, SHAP, Optuna, OR‑Tools.
- **Serving/Orchestration:** FastAPI, Docker/Compose, MLflow, Prefect, (optional) TorchServe/Triton.
- **Data/Storage:** Parquet + DuckDB for BI extracts.
- **Streaming/Observability (L5):** Kafka/Redpanda, Prometheus/Grafana.
- **BI:** Power BI (primary), simple web UI (Streamlit) for interactive tests.
- **Repo layout:**
  ```
  /data (gitignored)  /bi             /ops (runbooks, ADRs)
  /src                /api            /infra (docker/compose)
  /ui                 /tests          /notebooks (exploratory only)
  ```
- **Quality:** pre‑commit (black/ruff), unit tests (pytest), GitHub Actions CI, `.env` via python‑dotenv.

---

## Sprint 0 — Foundation (1 week)

**Objective:** Reproducible scaffolding, dataset ready, BI connectivity verified.

### Learning goals

- **Conceptual:** CMAPSS structure; FD001–FD004 differences; RUL labeling (max\_cycle − current\_cycle).
- **Implementation:** Data download → Parquet; basic feature store pattern; environment pinning; CI bootstrap.
- **Inspirational:** Rolls‑Royce/GE “Power by the Hour” stories to shape the narrative.

### Tasks (Parts)

1. **Part A – Repo & Tooling**
   - Create mono‑repo; add Makefile (or invoke) for `setup/test/train/run`.
   - Pre‑commit hooks; CI pipeline to run tests and linting.
2. **Part B – Data Ingest**
   - Script `src/data/ingest_cmapss.py` to download/unpack CMAPSS and persist to Parquet with a clean schema.
   - Add data dictionary (`/docs/data_dictionary.md`).
3. **Part C – Feature Store v0 (Descriptive)**
   - Minimal transforms: standardization utilities; rolling mean/std templates.
4. **Part D – BI Connectivity**
   - Export a small DuckDB or Parquet folder; confirm Power BI connects.

### Deliverables & Acceptance

- ✅ `make setup` works locally + CI green.
- ✅ Parquet datasets saved; dictionary exists.
- ✅ Power BI opens a sample table (screenshot).
- **Portfolio artifact:** README (project overview + data schema) + 1 slide “What CMAPSS is.”

---

## Sprint 1 — L1 “Hangar”: Telemetry Explorer (1 week)

**Objective:** Executive‑grade descriptive analytics; no ML yet.

### Learning goals

- **Conceptual:** Health Index (HI) concepts; sensor drift vs. degradation.
- **Implementation:** Rolling windows, PCA‑based HI, DAX basics for trend KPIs.
- **Inspirational:** Airline maintenance control‑tower layouts.

### Tasks (Parts)

1. **Part A – Health Index**
   - Build PCA/z‑score HI per unit; validate monotonicity near end‑of‑life.
2. **Part B – Failure Replay**
   - Export per‑cycle snapshots to drive a play axis in Power BI (scrubber effect).
3. **Part C – BI Report**
   - Pages: *Fleet Overview*, *Unit Timeline*, *Sensor Drift*, *HI vs Cycles*, *Failure Replay*.

### Deliverables & Acceptance

- ✅ Can select any unit and follow its lifecycle + HI.
- ✅ Failure replay page scrubs cycles.
- **Portfolio artifact:** `.pbix` + 3 annotated screenshots + 2‑min demo video script.

---

## Sprint 2 — L2 “Runway”: Baseline RUL Models (1–2 weeks)

**Objective:** Classic ML baselines with clear business impact in BI.

### Learning goals

- **Conceptual:** Correct cross‑validation by unit; leakage pitfalls; threshold‑based early‑warning.
- **Implementation:** Feature windowing; LightGBM/XGBoost; Optuna tuning; SHAP; cost model.
- **Inspirational:** Kaggle CMAPSS baselines (discipline in error analysis).

### Tasks (Parts)

1. **Part A – Features & Splits**
   - Create `src/pipelines/features.py` with sliding windows (e.g., 30–50 cycles) and deltas.
   - Split by engine id; create `train/valid/test` respecting official splits for FD001.
2. **Part B – Baseline Models**
   - Train Linear, RF, LightGBM/XGBoost; log metrics (RMSE/MAE, EarlyWarning\@RUL\<k).
   - Save `model.pkl` + `model_card.md`.
3. **Part C – Explainability**
   - SHAP global and per‑prediction; export top sensors/features to BI.
4. **Part D – BI Impact**
   - Pages: *Risk Heatmap* (by unit & cycle), *Cost of Failure* with slider (e.g., cost if RUL\<threshold), *Drivers (SHAP)*.

### Deliverables & Acceptance

- ✅ RMSE improves vs. naive > X% (record the %).
- ✅ BI recalculates avoided cost when moving the threshold.
- ✅ Model card complete (data, features, metrics, limitations).
- **Portfolio artifact:** 90‑sec narrated screen capture + case‑study 1‑pager (before/after costs).

---

## Sprint 3 — L3 “Flight”: Sequence Deep Learning + Alerts (2 weeks)

**Objective:** Sequence model with calibrated alerts, API, and mini UI.

### Learning goals

- **Conceptual:** Operating conditions/regimes; sequence windowing; probability calibration.
- **Implementation:** PyTorch LSTM/GRU/TCN; temperature scaling or isotonic; FastAPI serving; Streamlit tester.
- **Inspirational:** “AI control tower” demos.

### Tasks (Parts)

1. **Part A – Sequence Model**
   - Dataloader for sliding windows; per‑regime normalization (FD001+FD002).
   - Train GRU/LSTM; compare to L2; save to ONNX.
2. **Part B – Calibration & Thresholds**
   - Calibrate predicted RUL/alert probability; define target precision/recall for alerts.
3. **Part C – API & Mini UI**
   - FastAPI `/predict` for a window; `/score_unit` that walks through a unit.
   - Streamlit app to paste a unit id and visualize RUL trajectory + first‑alert lead time.
4. **Part D – Simulated Stream**
   - Cron or small scheduler to append the next cycle, call API, and log alerts.

### Deliverables & Acceptance

- ✅ API returns RUL + calibrated alert prob; latency budget documented.
- ✅ UI shows first‑alert lead time per unit.
- ✅ Docker images for API and UI build & run locally.
- **Portfolio artifact:** Short video: paste a unit → watch alert appear with lead time.

---

## Sprint 4 — L4 “Orbit”: MLOps + Optimization (2 weeks)

**Objective:** Production‑ready pipeline + maintenance scheduling optimizer.

### Learning goals

- **Conceptual:** Experiment tracking, model registry, drift monitoring; MIP for capacity/maintenance scheduling.
- **Implementation:** MLflow tracking/registry; Prefect flows; PSI/KS drift checks; OR‑Tools MIP; CI/CD for compose.
- **Inspirational:** Airline ops center runbooks.

### Tasks (Parts)

1. **Part A – MLflow + Prefect**
   - Flows: `ingest → features → train → register → batch_score` with MLflow logging artifacts/params.
   - Register best model; promote to `Staging` tag.
2. **Part B – Monitoring**
   - Compute PSI/KS on key features vs. training baseline; alert when thresholds exceed.
   - Add basic runbook in `/ops`.
3. **Part C – Maintenance Scheduler (OR‑Tools)**
   - Inputs: RUL predictions, bays/crew capacity, part availability.
   - Objective: minimize expected downtime; constraints for capacity, earliest service dates.
   - Output: Gantt and plan (CSV/JSON) + BI page *Plan vs Risk*.
4. **Part D – One‑Command Demo**
   - `docker compose up`: runs batch score, writes outputs, opens BI (or exports data for it), runs optimizer.

### Deliverables & Acceptance

- ✅ MLflow UI shows experiments; best model registered.
- ✅ Drift dashboard flags changes; runbook describes actions.
- ✅ Optimizer produces feasible plan and reduces expected downtime in a scenario.
- ✅ CI builds images and runs a smoke test.
- **Portfolio artifact:** 3‑min demo: retrain → register → score → optimize → BI view of savings.

---

## Sprint 5 — L5 “Mission Control”: Streaming + Serving + Policy (2 weeks)

**Objective:** Digital twin with real‑time scoring, observability, and policy optimization.

### Learning goals

- **Conceptual:** Streaming windows; SLO/latency; inventory policy (s,S); RL/Approx DP for threshold policy.
- **Implementation:** Kafka/Redpanda stream; consumer builds windows and calls model (TorchServe/Triton optional); Prometheus/Grafana; simple RL/ADP for threshold tuning; integrate inventory with scheduler.
- **Inspirational:** Digital‑twin & Triton case studies; real airline on‑call scenarios.

### Tasks (Parts)

1. **Part A – Telemetry Stream**
   - Producer replays historical units with jitter; topic per fleet.
   - Consumer aggregates windows → calls model service → writes alerts.
2. **Part B – Serving & Observability**
   - TorchServe/Triton (optional) or FastAPI; export metrics; dashboards for latency, throughput, drift.
3. **Part C – Inventory Policy + Scheduler**
   - (s,S) policy for critical modules; simulate lead times; couple with OR‑Tools schedule.
4. **Part D – Decision Agent**
   - Tune alert threshold via Approx RL/ADP to minimize combined cost (downtime + maintenance + stockout).
5. **Part E – Executive Demo (“Storm Night”)**
   - Trigger demand spike/lead‑time delay; watch KPIs adapt in Mission Control UI.

### Deliverables & Acceptance

- ✅ End‑to‑end: stream → score → alert → schedule → inventory, observable in dashboards.
- ✅ Latency under target on laptop (document budget & tests).
- **Portfolio artifact:** 5–7 min narrated demo video + architecture diagram + ops playbook.

---

## Risk Log & Mitigations

- **Data leakage:** Split by unit id; never mix sequences. *Mitigation:* unit tests on split function.
- **Overfitting DL:** Early stopping; cross‑regime validation; simpler TCN/GRU first.
- **BI performance:** Pre‑aggregate extracts; use DuckDB/Parquet partitions.
- **Scope creep:** Freeze DoD per sprint; park stretch goals.
- **Infra fatigue:** Use Docker Compose presets; optionalize Triton/Kafka until L5.

---

## Portfolio Milestones (what to publish after each sprint)

- **S1:** Telemetry Explorer screenshots + 1‑pager on HI.
- **S2:** Case study PDF (cost savings) + 90‑sec video + repo tag `v0.2`.
- **S3:** API/UI GIF; blog post on calibration & lead time.
- **S4:** Architecture diagram + MLflow/Prefect screenshots + optimizer Gantt.
- **S5:** Full demo video + README with `docker compose up` quickstart.

---

## Demo Script Outline (modify each sprint)

1. Problem framing (unplanned downtime is expensive). 2) Data & HI. 3) Baseline RUL & business impact. 4) Sequence model + alerts. 5) Production flow (MLflow + Prefect). 6) Optimization (OR‑Tools). 7) Streaming Mission Control & scenario.

---

## Evaluation Rubric (self‑check before publishing)

- **Business clarity:** Can a non‑technical buyer understand savings within 60 seconds?
- **Reproducibility:** One‑command run works; versions pinned.
- **Engineering:** Tests, logging, structured configs.
- **Modeling:** Sensible metrics; calibration; explainability.
- **Operations:** Monitoring & runbooks; optimizer respects constraints.
- **Storytelling:** Clean visuals; short, compelling demos.

