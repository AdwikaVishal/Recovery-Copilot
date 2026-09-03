# FEATURE 1 — Intelligent Recovery Action Scoring (Final Audit Report)

**Scope:** deterministic, explainable Expected-Recovery-Value scoring over a full
candidate-action matrix, with the Deterministic Policy Engine as the FINAL
AUTHORITY. No redesign of the agent pipeline; no fake ML; no UI redesign.

---

## A. Objective

Replace the single-strategy replay decision with a scored, ranked candidate
matrix: every permitted recovery action family is evaluated for the event,
scored with P(recovery) x amount-at-risk (Expected Recovery Value), ranked, and
markedly ineligible alternatives are surfaced with an explicit reason instead of
being silently dropped. Selection remains deterministic and fully explainable.

## B. Candidate actions

| Action family | Action | Channel | When it appears |
|---|---|---|---|
| RETRY_PAYMENT | `retry_payment` | `razorpay_api` | Card / recurring payment declines; contact-free direct charge |
| PAYMENT_LINK | `send_payment_link` | `whatsapp` | Invalid-expired instrument, invoice overdue, generic abandonment |
| PAYMENT_REMINDER | `send_dunning_message` | `whatsapp` | Low-friction nudge; lowest value, zero charge risk |
| REAUTHORIZE_MANDATE | `re_authorize_mandate` | `whatsapp` | Recurring events / mandate declines / RBI AFA path |
| HUMAN_REVIEW | `escalate_to_human` | `internal` | Low-confidence / high-risk / exhausted cases; never auto-executes money |
| STOP | *(governance)* | — | Opt-out / policy DENY / unsafe-contact; enforced by supervisor + policy |

## C. Scoring function

For each candidate:
```
Expected Recovery Value = P(recovery | strategy, decline_code) x amount_at_risk
```
- **P(recovery)** comes from the deterministic `RecoveryProbabilityEstimator`:
  known priors (`BASE_PROBABILITIES`) per (strategy, decline_code) — e.g.
  RETRY insufficient_funds=0.62, bank_timeout=0.74, do_not_honor=0.38; PAYMENT_LINK
  expired_card=0.50, incorrect_cvc=0.42; REAUTHORIZE mandate_afa=0.55 — refined
  by the outcome->learning loop only once `>=3` empirical outcomes exist. Every
  estimate carries `model_version`, `confidence`, and raw empirical `n/m` counts.
- **No synthetic uplift:** batch/replay events (`txn_*`) run on cold priors only,
  so the offline benchmark stays byte-identical across runs.

## D. Ranking / ordering

`candidates.sort(key=(eligible_first, -expected_value, strategy))`
- Eligible candidates ranked by Expected Recovery Value (desc).
- Ineligible candidates stay visible (EV-ranked within their set) with a reason,
  so a blocked option never displaces a runnable recommendation.
- Deterministic tie-break by strategy name. No randomness anywhere.
- Election: the deterministic rule-based primary is elected on cold data (exact
  benchmark contract); the learning loop can overturn it only with real evidence
  favoring an eligible alternative. Decision factors are recorded
  (`decision_factors`) beside each ranked candidate for the audit trail.

## E. Safety — Policy Engine is FINAL AUTHORITY

- The scorer only proposes. The Deterministic Policy Engine (opt-out, AFA,
  max_retries, cooling period, discount ceiling, contact frequency, contact
  window, amount ceiling, risk, confidence) makes the authoritative decision.
- DENY => zero execution, zero money (`STOPPED`, ₹0), audited.
- Advisory ineligibility flags (never bypass policy):
  `AFA_REQUIRED`, `RETRY_LIMIT_REACHED`, `PAYMENT_METHOD_INVALID`,
  `OPT_OUT_OR_SAFETY`, `CONTACT_FREQUENCY_LIMIT`, `OUTSIDE_CONTACT_WINDOW`,
  `NOT_APPLICABLE`.
- HUMAN_REVIEW never auto-executes (`requires_human_approval`, approval-gated).

## F. Benchmarks

| Metric | Value |
|---|---|
| Batch events | 100 |
| Baseline counterfactual | ₹68,029.74 |
| Copilot cold-run recovered | **27/100 — ₹94,060.14 (+38.3% vs baseline)** |
| blocked_by_policy | 6 |
| human_review | 49 |
| pending | 18 |
| errors | 0 |
| Audit verifiability | 100/100 |

**Honest A/B (expected vs actual, `GET /api/recovery/scoring-comparison`):**
the scorer's Expected Recovery Value summed over the executed, policy-authorized
events (₹521,995.97) exceeds actual recovered (₹94,060.14). This is the honest
calibration view over synthetic data — the report states plainly that the ranked
scorer is decision-visible but did NOT change batch results (rule-based primary
wins by design), so no uplift is claimed from ranking outputs.

## G. Real-time compatibility

Unchanged SSE stages now carry richer payloads:
- `strategy.candidates_generated`: `count`, `eligible`, per-candidate
  `strategy/action/probability/expected_value/eligible/ineligibility_reason/
  reason_codes/empirical_attempts/empirical_successes`.
- `strategy.ranked`: `ranked` (rank, EV, risk, friction, cost), `selected`,
  `decision_factors`, `reason`.
- Batch and real-time share the same `build_optimizer_output`; `now` is threaded
  for contact-window scoring (batch passes a fixed clock; live uses `None`).

## H. Test coverage

New `tests/test_recovery_scoring.py` — **11 tests** (10 required + cold-DB parity):
transient->RETRY top; expired_card->RETRY ineligible & PAYMENT_LINK wins;
incorrect_cvc->PAYMENT_LINK top; mandate AFA->REAUTHORIZE top & RETRY AFA-blocked;
opt-out->all contact actions ineligible, nothing executes; retry budget
exhausted->ineligible; contact-frequency limit->comms ineligible; low
confidence->HUMAN_REVIEW ranked above risky auto-actions & elected; comparable
probabilities->higher EV wins (exact P x Amount math); policy DENY->zero recovery,
scorer recommendation + denial present in audit. Full suite: **163 passed, 0 failed**
(+3.4%/11 tests vs the previous 152), scenario suite 20/20 preserved.

## I. Known limitations (honesty)

- Priors are rule-based; real-world calibration requires genuine outcome volume.
- The EV priors overstate actuals on synthetic data (see F) — deliberately not
  hidden with fake precision.
- Advisory eligibility mirrors policy conditions but policy is the binding check.

## J. ML seam

`RecoveryProbabilityEstimator` is an ABC (`estimate(event, context, candidate)`).
A trained ML scorer can swap in behind the same interface; the learning loop
already writes empirical probabilities that replace cold priors at runtime.

## K. Future work

- Train the ML probability estimator on the recorded outcomes ledger.
- Dashboard: render eligibility chips + decision factors in the candidate card.
- Calibrate priors from confirmed real outcomes as volume grows.

## L. Success criteria

- [x] Supported actions only (RETRY / LINK / REMINDER / REAUTHORIZE / HUMAN_REVIEW / STOP).
- [x] Ranked by Expected Recovery Value; deterministic, explainable; **no ML promises**.
- [x] Policy Engine is FINAL AUTHORITY; no unsafe action can bypass policy.
- [x] Opt-out / AFA / retry-limit / frequency-limit / window all surface with reasons.
- [x] 10 required scorer tests written and green (11 total incl. parity guard).
- [x] Batch + real-time share the scorer; benchmark & 20/20 scenarios unchanged.
- [x] Honest A/B comparison endpoint; no fake uplift claimed.
- [x] Audit trail shows scorer recommendations + policy verdicts per event.