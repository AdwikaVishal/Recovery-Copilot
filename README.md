# Recovery Copilot v4 — One Intelligent Recovery Workflow

A single end-to-end recovery system where the **trained Recovery ML model** and
the **AI/agent decision layer** are genuinely wired into ONE closed loop:

```
transaction → ML recovery probability → AI decision (optimizer + strategy)
           → Deterministic Policy Engine → action → outcome (webhook-confirmed)
           → analytics
```

Everything lives in `backend/`. The ML model is a **frozen artifact** (never
refit); the AI agents, policy engine and live loop sit on top of it.

---

## 1. Why this exists

Chargebacks, failed mandates and dunning calls are handled by disconnected tools:
a scoring model, a rules engine, a messaging agent. This buildathon project
connects them so every decision carries the **ML probability**, is traced, passes
a **deterministic policy boundary**, and closes back on a **confirmed outcome**
only via a trusted payment webhook — with honest analytics of what actually
recovered.

## 2. Architecture (one workflow, two responsibilities)

```
Inbound failure (webhook / simulator / HTTP)
        │
        ▼
engine/realtime.run_live_recovery   (SSE stage stream per transaction)
        │
        ├─ ML BRIDGE  app/recovery/bridge.py
        │     RevenueEvent → 44 raw features → 49 encoded → ExtraTreesClassifier
        │     → sigmoid calibration → threshold → MLPrediction (canonical P(recovery))
        │
        ├─ AI DECISION  agents/supervisor.process_event(ml_estimate=…)
        │     ├─ diagnosis_agent        (decline-code knowledge)
        │     ├─ customer_context_agent (consent / disputes / channels)
        │     ├─ recovery_optimizer     (candidates ranked by EV = P_ml × amount)
        │     └─ recovery_strategy_agent(elected strategy, deterministic primary)
        │
        ├─ POLICY  agents/policy_engine.py  (ALLOW / MODIFY / DENY / HUMAN_REVIEW)
        │
        ├─ ACTION  agents/execution_adapter.py (SIMULATED in demo mode)
        │
        └─ OUTCOME app/database.recovery_decisions
              ├─ decision recorded at run time (outcome=pending)
              └─ recovered_72h ONLY via confirm_live_recovery (trusted webhook)
```

**Single source of truth for P(recovery):** the calibrated ML probability. The
rule-based estimator (`agents/probability_estimator.py`) remains as the explicit
fallback and for offline benchmark replay (`txn_*`), never silently overriding
an available ML score. `probability_source` is traced end-to-end.

## 3. ML model facts (authoritative, from `backend/recovery_model_artifacts/`)

| Item | Value |
| --- | --- |
| Model | ExtraTreesClassifier, 500 trees, `class_weight=balanced`, seed 42 |
| Features | 44 raw → 49 encoded (payment_method one-hot, 6 categories) |
| Preprocessing | SimpleImputer(median, 43 stats) + OneHotEncoder(`handle_unknown=ignore`) |
| Calibration | LogisticRegression sigmoid (coef ~3.01, intercept ~−3.43), monotone |
| Threshold | **0.04** (from `final_metrics.json`) |
| Ref date | 2023-01-01 11:58:01 |
| Test metrics | ROC-AUC 0.5795 · PR-AUC 0.0686 · Brier 0.0366 · LogLoss 0.1621 |
| Calibrated | Brier 0.0365 · LogLoss 0.1606 |

Model quality is **modest** (sklearn metrics shown above); it is used to
*inform* decisions and is never treated as ground truth. See §9.

> **sklearn version pin:** artifacts were pickled with scikit-learn 1.8.0.
> Requirement `scikit-learn==1.8.0` keeps inference reproducible; running 1.9.0
> still loads but emits `InconsistentVersionWarning`. Artifacts are **not**
> retrained or replaced.

## 4. The intelligence hand-off (what actually changed)

- `app/recovery/bridge.py` — new. `recovery_prediction_for_event` maps a live
  failure to the recovery-population `pending` status and returns an
  `MLPrediction` the agents can consume.
- `agents/recovery_optimizer.py` — accepts `ml_estimate`; every candidate gets
  `probability = calibrated_ml`, `expected_value = amount × P_ml`, and ML factors
  surface in `decision_factors` / `selection_reason`. Without it, falls back to
  rule-based explicitly.
- `agents/supervisor.py` — `process_event(..., ml_estimate=…, execute_action=…)`.
  `execute_action=False` is the advisory (decision-only) mode that stops before
  execution; the optimizer trace is included in the returning `SupervisorOutput`.
- `engine/realtime.py` — computes the ML estimate for live events (skips
  `txn_*` benchmark replay), publishes an `ml.prediction` SSE stage, records a
  `recovery_decisions` row per transaction, and updates its outcome on trusted
  confirmation.
- `app/database.py` — new `recovery_decisions` table + helpers.
- `app/recovery/routes.py` / `schemas.py` — new endpoints (below).
- `static/dashboard.html` — AI-decision section (real backend calls only) plus a
  prediction→decision→outcome ledger and closed-loop analytics.
- `tests/test_ml_ai_integration.py` — 14 acceptance tests pinning the contract.

## 5. Endpoints

| Endpoint | What it does |
| --- | --- |
| `POST /api/recovery/predict` | Score one recovery-population transaction (44→49→model) |
| `POST /api/recovery/batch-predict` | Batch score (consistent with single) |
| `GET /api/recovery/model-info` / `features` / `metrics` | Frozen-artifact metadata |
| `GET /api/recovery/predictions` | Persisted predictions |
| `POST /api/recovery/decision` | **Advisory closed loop**: prediction → AI decision → policy verdict (no execution) |
| `POST /api/recovery/process` | **Full run**: closed loop + (simulated) execution + persisted outcome record |
| `GET /api/recovery/outcomes` | Prediction→decision→outcome ledger |
| `GET /api/recovery/analytics` | Closed-loop aggregates (source, decision, verdict, outcome, recovered ₹) |
| `POST /api/webhooks/payment` | Live ingress; payment.captured/subscription.charged confirm recovery |
| `GET /api/events/stream` | SSE stage stream |
| `GET /api/live/metrics` | Live recovery metrics |
| `GET /` | Dashboard (Recovery AI tab has the closed-loop UI) |

## 6. Run it

```bash
# app only (keeps DB) — server on :8766 e.g.
./start.sh api 8766

# full demo pipeline + API
./start.sh all 8766

# or, directly (FastAPI dev server on :8322)
cd backend && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8322

# live webhook simulator
python3 -m tools.simulate_realtime --interval 3 --count 10
```

Dashboard: `http://localhost:8322/` → **Recovery AI** tab → *AI Recovery Decision*.

## 7. Tests & demo

```bash
cd backend && python3 -m pytest -q --ignore=tests/test_ingestion.py
# 168 passed (30 in test_ml_ai_integration.py + test_full_chain_integration.py)
# 42 errors are pre-existing in test_ingestion.py (import-related) — untouched.
```

Deterministic E2E demo used for this deliverable:

```
POST /api/demo/run          → 5-scenario narrative (trusted webhook closes the loop)
POST /api/batch/run         → 100-event synthetic benchmark
POST /api/recovery/process
{ "transaction_id":"demo_ml_1","customer_id":"cust_demo",
  "amount":850,"payment_method":"apple_pay" }

→ recovery_probability = calibrated ML (ml:ExtraTreesClassifier-500-final)
→ probability_source = "ml:ExtraTreesClassifier-500-final"
→ policy verdict + AI decision + persisted recovery_decisions row
→ GET /api/recovery/outcomes + /api/recovery/analytics show the chain
```

## 8. Honest labeling

- **Execution is SIMULATED** in demo mode (`WEBHOOK_MODE=demo`, the execution
  adapter returns a deterministic result). Leadership essential reads know this;
  nothing claims real money movement.
- **Recovered amounts** only ever come from a trusted confirmation webhook
  (`recovered_72h` outcome), never from the simulator.
- **Metrics are not fabricated.** `final_metrics.json` (above) is the truth;
  earlier audit notes of different AUC values are not reproduced here.

## 9. Known limitations (kept honest)

1. **Weak model** — test ROC-AUC ≈ 0.58, PR-AUC ≈ 0.07. The ML is an informer,
   not ground truth; policy always bounds it.
2. **Out-of-contract input** — the scoring API documents `history` as *prior*
   transactions only. A history event dated *after* the request transaction can
   silently re-rank which row is scored (`build_raw_features`, `_to_frame`).
   No schema guard exists. Prior-only history (the documented contract, tested
   by `test_13_no_future_information_leakage`) is correct.
3. **sklearn version-pin warning** — see §3.
4. **Vestigial/unused** — root `frontend/` has no backend references. A stale
   orphan `recovery_copilot.db` at the repo root (no `recovery_predictions` table)
   was removed; only `backend/recovery_copilot.db` is used at runtime.
5. **`txn_*` benchmark replay** intentionally bypasses the ML bridge so the
   100-event batch benchmark stays byte-identical (rule-based path).

## 10. Repository hygiene

`.gitignore` excludes node_modules, `*.db`, `__pycache__`, notebooks, and the
heavy model artifacts (`.joblib`, model metadata) — the 102 MB model artifact
should not be committed.