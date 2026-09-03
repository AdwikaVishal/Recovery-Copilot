# 🔁 Recovery Copilot v4
### One intelligent, closed-loop revenue recovery workflow

**Razorpay AI Buildathon 2026 — Track: AI Revenue Recovery**

> A trained ML model and an AI/agent decision layer, genuinely wired into
> **one** closed loop — bounded by a deterministic policy engine, confirmed
> only by trusted payment webhooks, never by the LLM itself.

```
transaction → ML recovery probability → AI decision (optimizer + strategy)
           → Deterministic Policy Engine → action → outcome (webhook-confirmed)
           → analytics
```

---

## Table of contents

1. [Why this exists](#1-why-this-exists)
2. [Architecture](#2-architecture)
3. [Why the AI is on a leash (design philosophy)](#3-why-the-ai-is-on-a-leash-design-philosophy)
4. [ML model card](#4-ml-model-card)
5. [What actually changed — the intelligence hand-off](#5-what-actually-changed--the-intelligence-hand-off)
6. [API reference](#6-api-reference)
7. [Quickstart](#7-quickstart)
8. [Tests & deterministic demo](#8-tests--deterministic-demo)
9. [Honest labeling](#9-honest-labeling)
10. [Known limitations & roadmap](#10-known-limitations--roadmap)
11. [Repository structure](#11-repository-structure)
12. [Repository hygiene](#12-repository-hygiene)

---

## 1. Why this exists

Chargebacks, failed mandates, and dunning calls are usually handled by
disconnected tools — a scoring model here, a rules engine there, a messaging
agent somewhere else, none of them talking to each other. **Recovery Copilot**
connects them into a single accountable pipeline where:

- every decision carries a **calibrated ML probability**,
- every step is **traced** (you can always answer "why did it do that?"),
- every action passes a **deterministic policy boundary** before it can fire,
- and every recovered rupee is **confirmed by a trusted payment webhook** —
  never by the simulator, and never by the AI's own say-so.

The result is honest analytics of what *actually* recovered, not what the
system merely attempted.

## 2. Architecture

```
Inbound failure (webhook / simulator / HTTP)
        │
        ▼
engine/realtime.run_live_recovery        SSE stage stream, per transaction
        │
        ├─ ML BRIDGE            app/recovery/bridge.py
        │     RevenueEvent → 44 raw features → 49 encoded →
        │     ExtraTreesClassifier → sigmoid calibration → threshold →
        │     MLPrediction (canonical P(recovery))
        │
        ├─ AI DECISION          agents/supervisor.process_event(ml_estimate=…)
        │     ├─ diagnosis_agent          decline-code knowledge
        │     ├─ customer_context_agent   consent / disputes / channels
        │     ├─ recovery_optimizer       candidates ranked by EV = P_ml × amount
        │     └─ recovery_strategy_agent  elected strategy (deterministic primary)
        │
        ├─ POLICY                agents/policy_engine.py
        │     ALLOW / MODIFY / DENY / HUMAN_REVIEW
        │
        ├─ ACTION                agents/execution_adapter.py   (SIMULATED in demo mode)
        │
        └─ OUTCOME               app/database.recovery_decisions
              ├─ decision recorded at run time      (outcome = pending)
              └─ recovered_72h set ONLY by confirm_live_recovery (trusted webhook)
```

**Single source of truth for P(recovery):** the calibrated ML probability. The
rule-based estimator (`agents/probability_estimator.py`) exists purely as an
explicit fallback and for offline benchmark replay (`txn_*`) — it never
silently overrides an available ML score. `probability_source` is traced
end-to-end so you can always tell which path produced a number.

## 3. Why the AI is on a leash (design philosophy)

Most "AI agent" systems hand an LLM tools, a goal, and open-ended authority to
mutate state. In payments, that's not a feature — it's a liability. Recovery
Copilot inverts the pattern:

| Layer | Job | Authority |
| --- | --- | --- |
| ML model | Estimate P(recovery) | Informs, never decides |
| AI agents | Diagnose failure, propose a strategy | Propose only |
| Policy engine | Check opt-outs, retry caps, cooling periods, thresholds | **Final say** |
| Execution adapter | Carry out the allowed action | Bounded, logged |
| Payment webhook | Confirm the money actually moved | **Only source of truth** |

The LLM can be wrong, hallucinate, or drift. The policy engine and the webhook
can't be talked out of their job. That separation is the whole point.

## 4. ML model card

*(authoritative source: `backend/recovery_model_artifacts/`)*

| Item | Value |
| --- | --- |
| Model | ExtraTreesClassifier, 500 trees, `class_weight=balanced`, seed 42 |
| Features | 44 raw → 49 encoded (payment_method one-hot, 6 categories) |
| Preprocessing | `SimpleImputer(median, 43 stats)` + `OneHotEncoder(handle_unknown="ignore")` |
| Calibration | Logistic sigmoid (coef ≈ 3.01, intercept ≈ −3.43), monotone |
| Decision threshold | **0.04** (from `final_metrics.json`) |
| Reference date | 2023-01-01 11:58:01 |
| Test metrics (raw) | ROC-AUC 0.5795 · PR-AUC 0.0686 · Brier 0.0366 · LogLoss 0.1621 |
| Test metrics (calibrated) | Brier 0.0365 · LogLoss 0.1606 |

**Model quality is modest by design honesty, not by accident** — ROC-AUC 0.58
means this model is a weak-to-moderate signal, and the system is architected
around that fact rather than hiding it: the ML output *informs* the AI layer,
the policy engine still gates every action, and nothing here is presented as
ground truth. See [§10](#10-known-limitations--roadmap) for how this gets
better next.

> **sklearn version pin:** artifacts were pickled with scikit-learn 1.8.0.
> `scikit-learn==1.8.0` keeps inference reproducible; running 1.9.0 still
> loads but emits `InconsistentVersionWarning`. Artifacts are **not** retrained
> or replaced at runtime.

## 5. What actually changed — the intelligence hand-off

| File | Role |
| --- | --- |
| `app/recovery/bridge.py` | **New.** `recovery_prediction_for_event` maps a live failure to the recovery-population `pending` status and returns an `MLPrediction` the agents consume. |
| `agents/recovery_optimizer.py` | Accepts `ml_estimate`; every candidate gets `probability = calibrated_ml`, `expected_value = amount × P_ml`, and ML factors surface in `decision_factors` / `selection_reason`. Falls back to rule-based explicitly when no ML estimate is available. |
| `agents/supervisor.py` | `process_event(..., ml_estimate=…, execute_action=…)`. `execute_action=False` is advisory (decision-only) mode — it stops before execution; the optimizer trace is included in the returned `SupervisorOutput`. |
| `engine/realtime.py` | Computes the ML estimate for live events (skips `txn_*` benchmark replay), publishes an `ml.prediction` SSE stage, records a `recovery_decisions` row per transaction, and updates its outcome on trusted confirmation. |
| `app/database.py` | New `recovery_decisions` table + helpers. |
| `app/recovery/routes.py` / `schemas.py` | New endpoints (see [§6](#6-api-reference)). |
| `static/dashboard.html` | AI-decision section (real backend calls only) plus a prediction → decision → outcome ledger and closed-loop analytics. |
| `tests/test_ml_ai_integration.py` | 14 acceptance tests pinning the contract described above. |

## 6. API reference

| Endpoint | What it does |
| --- | --- |
| `POST /api/recovery/predict` | Score one recovery-population transaction (44 → 49 → model) |
| `POST /api/recovery/batch-predict` | Batch score, consistent with single-score path |
| `GET /api/recovery/model-info` / `/features` / `/metrics` | Frozen-artifact metadata |
| `GET /api/recovery/predictions` | Persisted predictions |
| `POST /api/recovery/decision` | **Advisory closed loop** — prediction → AI decision → policy verdict, no execution |
| `POST /api/recovery/process` | **Full run** — closed loop + (simulated) execution + persisted outcome record |
| `GET /api/recovery/outcomes` | Prediction → decision → outcome ledger |
| `GET /api/recovery/analytics` | Closed-loop aggregates (source, decision, verdict, outcome, recovered ₹) |
| `POST /api/webhooks/payment` | Live ingress — `payment.captured` / `subscription.charged` confirm recovery |
| `GET /api/events/stream` | SSE stage stream |
| `GET /api/live/metrics` | Live recovery metrics |
| `GET /` | Dashboard — **Recovery AI** tab has the closed-loop UI |

## 7. Quickstart

```bash
# app only (keeps existing DB) — e.g. server on :8766
./start.sh api 8766

# full demo pipeline + API
./start.sh all 8766

# or run the FastAPI dev server directly
cd backend && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8322

# live webhook simulator, one event every 3s
python3 -m tools.simulate_realtime --interval 3 --count 10
```

Then open the dashboard: **`http://localhost:8322/`** → **Recovery AI** tab →
*AI Recovery Decision*.

## 8. Tests & deterministic demo

```bash
cd backend && python3 -m pytest -q --ignore=tests/test_ingestion.py
# 168 passed (30 in test_ml_ai_integration.py + test_full_chain_integration.py)
# 42 errors are pre-existing in test_ingestion.py (import-related) — untouched.
```

Deterministic end-to-end demo used for this deliverable:

```bash
POST /api/demo/run          # 5-scenario narrative; trusted webhook closes the loop
POST /api/batch/run         # 100-event synthetic benchmark

POST /api/recovery/process
{ "transaction_id": "demo_ml_1", "customer_id": "cust_demo",
  "amount": 850, "payment_method": "apple_pay" }

# → recovery_probability = calibrated ML (ml:ExtraTreesClassifier-500-final)
# → probability_source   = "ml:ExtraTreesClassifier-500-final"
# → policy verdict + AI decision + persisted recovery_decisions row

GET /api/recovery/outcomes      # see the confirmed chain
GET /api/recovery/analytics     # see the aggregate view
```

## 9. Honest labeling

- **Execution is SIMULATED** in demo mode (`WEBHOOK_MODE=demo` — the execution
  adapter returns a deterministic result). Nothing here claims real money
  movement.
- **Recovered amounts** only ever come from a trusted confirmation webhook
  (`recovered_72h` outcome) — never from the simulator.
- **Metrics are not fabricated.** `final_metrics.json` is the source of truth;
  earlier internal drafts with different AUC values are not reproduced here.

## 10. Known limitations & roadmap

| # | Limitation | Planned fix |
| --- | --- | --- |
| 1 | **Weak model** — test ROC-AUC ≈ 0.58, PR-AUC ≈ 0.07. | Re-engineer features (recency/frequency of failures, merchant-level priors), try gradient-boosted trees with class-imbalance-aware objectives, and re-evaluate calibration on a larger holdout. The ML stays an *informer*; the policy engine keeps bounding it regardless of model quality. |
| 2 | **Out-of-contract input** — the scoring API documents `history` as *prior* transactions only. A history event dated *after* the request transaction can silently re-rank which row gets scored (`build_raw_features`, `_to_frame`); no schema guard currently exists. | Add an explicit schema/temporal guard that rejects or flags future-dated history rows before scoring. `test_13_no_future_information_leakage` already pins the correct (prior-only) contract — the guard formalizes it at the API boundary. |
| 3 | **sklearn version-pin warning** — artifacts pickled on 1.8.0, project runs on 1.9.0. | Re-pickle artifacts once the pin is bumped and validated, or vendor the exact training environment. |
| 4 | **Vestigial `frontend/`** — unused, no backend references. A stale orphan `recovery_copilot.db` at the repo root has already been removed; only `backend/recovery_copilot.db` is used at runtime. | Either wire `frontend/` up to the real API or delete it — currently tracked as dead weight. |
| 5 | **`txn_*` benchmark replay** intentionally bypasses the ML bridge so the 100-event batch benchmark stays byte-identical (rule-based path). | Working as intended — documented here so it isn't mistaken for a bug. |

## 11. Repository structure

```
backend/
├── app/
│   ├── main.py                 FastAPI app, CORS allow-list, router wiring
│   ├── database.py             SQLite models incl. recovery_decisions
│   ├── models.py                Pydantic schemas (RevenueEvent, PolicyVerdict, …)
│   └── recovery/
│       ├── bridge.py            ML → MLPrediction bridge
│       ├── routes.py            /api/recovery/* endpoints
│       └── schemas.py
├── agents/
│   ├── supervisor.py             orchestrates the agent chain
│   ├── diagnosis_agent.py
│   ├── customer_context_agent.py
│   ├── recovery_optimizer.py
│   ├── recovery_strategy_agent.py
│   ├── policy_engine.py          deterministic ALLOW/MODIFY/DENY/HUMAN_REVIEW
│   ├── execution_adapter.py      simulated in demo mode
│   ├── probability_estimator.py  rule-based fallback
│   └── outcome_handler.py
├── engine/
│   ├── realtime.py               SSE live pipeline + recovery_decisions writes
│   ├── ingestion.py               webhook + simulator normalization
│   ├── webhook.py
│   ├── pipeline.py / audit.py
├── recovery_model_artifacts/     frozen model + preprocessing (not committed, see §12)
├── static/dashboard.html         operator dashboard, AI Recovery Decision tab
├── tools/simulate_realtime.py    live webhook simulator
├── tests/                        168 passing (excl. pre-existing ingestion errors)
└── requirements.txt
frontend/                         unused — see §10
start.sh
```

## 12. Repository hygiene

`.gitignore` excludes `node_modules/`, `*.db`, `__pycache__/`, notebooks, and
the heavy model artifacts (`.joblib`, model metadata) — the 102 MB model
artifact should never be committed. Secrets (`.env`, `.env.*`) are excluded
throughout.

---

*Built for Razorpay AI Buildathon 2026 — AI Revenue Recovery track.*
