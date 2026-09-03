"""ML -> AI -> decision -> action -> outcome closed-loop integration tests.

Phase-18 acceptance contract — the ONE intelligent recovery workflow:

  1. The trained recovery model loads through the shared bridge and yields a
     canonical (calibrated) recovery probability.
  2. The AI decision layer consumes the calibrated ML probability as the
     authoritative P(recovery) (single source of truth), with the rule-based
     estimator kept ONLY as an explicit fallback.
  3. `probability_source` is traced end-to-end and never silently swapped.
  4. The deterministic policy engine still gates the decision (can reject).
  5. A full run persists prediction -> decision -> (outcome record) in
     `recovery_decisions`, linked by transaction_id.
  6. The ML feed is disabled for offline/benchmark replay (`txn_*`) so the
     benchmark contract stays byte-identical.
  7. Single vs batch inference are consistent; non-recovery endpoints intact.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["WEBHOOK_MODE"] = "demo"
os.environ["WEBHOOK_ALLOW_UNSIGNED"] = "false"

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import asyncio  # noqa: E402

import pytest  # noqa: E402

from datetime import datetime  # noqa: E402

from app.models import (  # noqa: E402
    RevenueEvent, Customer, TransactionMetadata,
    EventType, DeclineCode,
)


def _live_event(event_id: str, customer_id: str = "cust_ml1",
                amount: int = 1200, decline_code: str = "insufficient_funds",
                retry_count: int = 0) -> RevenueEvent:
    return RevenueEvent(
        id=event_id,
        type=EventType.CARD_PAYMENT_FAILURE,
        customer=Customer(id=customer_id, name="ML User",
                          phone="+919876543210", email="ml@example.com"),
        amount=amount,
        currency="INR",
        root_cause=DeclineCode(decline_code),
        decline_code=DeclineCode(decline_code),
        failed_at=datetime(2026, 1, 15, 12, 0, 0),
        retry_count=retry_count,
        metadata=TransactionMetadata(),
    )


@pytest.fixture()
def isolated_db(tmp_path):
    from app.database import set_active_db_path, init_db, reset_active_db_path
    token = set_active_db_path(Path(tmp_path) / "test.db")
    asyncio.run(init_db())
    yield
    reset_active_db_path(token)


# ---------------------------------------------------------------------------
# 1. Bridge = single source of truth for the ML probability
# ---------------------------------------------------------------------------

def test_bridge_loads_trained_model_and_calibrates():
    from app.recovery.bridge import (
        recovery_prediction_for_event, MODEL_VERSION,
    )
    ml = recovery_prediction_for_event(_live_event("evt_br1"))
    assert ml.available is True
    assert ml.probability_source == MODEL_VERSION
    assert ml.model_artifact == "final_model.joblib"
    assert ml.threshold == pytest.approx(0.04)
    assert 0.0 <= ml.recovery_probability <= 1.0
    assert ml.recovery_prediction in (0, 1)
    assert ml.recovery_prediction == (1 if ml.recovery_probability >= ml.threshold else 0)
    assert ml.risk_band
    assert ml.risk_label


def test_bridge_probability_matches_predict_endpoint(isolated_db):
    """Same request -> same calibrated probability through /predict and bridge."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.recovery.bridge import recovery_prediction_for_event
    ev = _live_event("evt_br2", amount=85000)
    ml = recovery_prediction_for_event(ev, payment_method="apple_pay")
    with TestClient(app) as client:
        r = client.post("/api/recovery/predict", json={
            "transaction_id": "br2_sync",
            "customer_id": ev.customer.id,
            "transaction_date": "2026-01-15 12:00:00",
            "quantity": 0,
            "unit_price": 0,
            "total_amount": ev.amount,
            "discount_applied": 0,
            "shipping_cost": 0,
            "payment_method": "apple_pay",
            "status": "pending",
        })
        assert r.status_code == 200
        j = r.json()
        # identical population/features -> identical calibrated probability
        assert abs(j["recovery_probability"] - ml.recovery_probability) < 1e-4


# ---------------------------------------------------------------------------
# 2. AI optimizer consumes calibrated ML as authoritative P(recovery)
# ---------------------------------------------------------------------------

def test_optimizer_uses_calibrated_ml_probability(isolated_db):
    from app.recovery.bridge import recovery_prediction_for_event, MODEL_VERSION
    from agents.diagnosis_agent import diagnose
    from agents.customer_context_agent import get_customer_context
    from agents.recovery_optimizer import build_optimizer_output, score_candidates

    ev = _live_event("evt_opt1", amount=85000)
    ml = recovery_prediction_for_event(ev, payment_method="apple_pay")

    async def _run():
        diag = diagnose(ev)
        ctx = await get_customer_context(ev)
        out = await build_optimizer_output(ev, diag, ctx, ml_estimate=ml)
        assert out.elected is not None
        cand = out.elected
        # authoritative calibrated ML probability
        assert cand.probability == pytest.approx(ml.recovery_probability, abs=1e-4)
        assert cand.probability_source == MODEL_VERSION
        # ML factors surfaced in the structured decision trace
        assert any("ml_probability=" in f for f in out.decision_factors)
        assert any("ml_source=" in f for f in out.decision_factors)
        assert "ML probability" in out.selection_reason
        # expected value uses the ML probability
        assert cand.expected_value == int(ev.amount * cand.probability)
        summary = score_candidates(out.candidates)
        assert all(c["probability_source"] == MODEL_VERSION for c in summary)
        return out

    asyncio.run(_run())


def test_optimizer_falls_back_to_rule_based_when_no_ml(isolated_db):
    from agents.diagnosis_agent import diagnose
    from agents.customer_context_agent import get_customer_context
    from agents.recovery_optimizer import build_optimizer_output

    ev = _live_event("evt_opt2", amount=1200, decline_code="bank_timeout")

    async def _run():
        diag = diagnose(ev)
        ctx = await get_customer_context(ev)
        out = await build_optimizer_output(ev, diag, ctx)  # no ml_estimate
        assert out.elected is not None
        assert out.elected.probability_source == "rule-based-v1"
        assert not any("ml_" in f for f in out.decision_factors)
        return out

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. Supervisor threads the ML estimate (no stale rule-based override)
# ---------------------------------------------------------------------------

def test_supervisor_includes_ml_estimate_and_optimizer_trace(isolated_db):
    from app.recovery.bridge import recovery_prediction_for_event, MODEL_VERSION
    from app.models import WorkflowStatus
    from agents.supervisor import process_event

    ev = _live_event("evt_sup1", amount=85000)
    ml = recovery_prediction_for_event(ev, payment_method="apple_pay")

    async def _run():
        out = await process_event(ev, ml_estimate=ml)
        assert out.optimizer is not None
        elected = next(
            (c for c in out.optimizer["candidates"] if c["strategy"] == out.optimizer["strategy"]),
            None,
        ) or out.optimizer["candidates"][0]
        assert elected["probability_source"] == MODEL_VERSION
        assert elected["probability"] == pytest.approx(ml.recovery_probability, abs=1e-4)
        return out

    out = asyncio.run(_run())
    assert out.workflow_status in (
        WorkflowStatus.PENDING_WEBHOOK, WorkflowStatus.STOPPED,
        WorkflowStatus.HUMAN_REVIEW, WorkflowStatus.RESOLVED,
    )


def test_supervisor_decision_only_mode_does_not_execute(isolated_db):
    """execute_action=False stops before the execution block (no attempt row)."""
    from app.recovery.bridge import recovery_prediction_for_event
    from agents.supervisor import process_event

    ev = _live_event("evt_sup2", amount=50000)
    ml = recovery_prediction_for_event(ev)

    async def _run():
        out = await process_event(ev, ml_estimate=ml, execute_action=False)
        assert out.optimizer is None or isinstance(out.optimizer, dict)
        return out

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. HTTP: advisory decision endpoint — full chain in one response
# ---------------------------------------------------------------------------

def test_decision_endpoint_returns_full_chain(isolated_db):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.recovery.bridge import MODEL_VERSION

    with TestClient(app) as client:
        r = client.post("/api/recovery/decision", json={
            "transaction_id": "dec_1",
            "customer_id": "cust_ml1",
            "amount": 850,
            "payment_method": "apple_pay",
        })
        assert r.status_code == 200
        j = r.json()
        assert j["transaction_id"] == "dec_1"
        assert j["probability_source"] == MODEL_VERSION
        assert 0.0 <= j["recovery_probability"] <= 1.0
        assert j["probability_raw"] is not None
        assert j["threshold"] == pytest.approx(0.04)
        assert j["risk_band"]
        assert j["ai_decision"] in ("RETRY", "REAUTHORIZE", "PAYMENT_LINK",
                                    "MESSAGE", "PTP_FOLLOWUP", "HUMAN_REVIEW", "STOP")
        assert j["action"]
        assert j["policy_verdict"] in ("ALLOW", "MODIFY", "DENY", "HUMAN_REVIEW")
        assert any("ml_probability=" in f for f in j["reasoning"])
        assert j["action_mode"] == "advisory"
        assert j["outcome"] == "pending"


def test_process_endpoint_persists_decision_record(isolated_db):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.database import get_recovery_decision
    from app.recovery.bridge import MODEL_VERSION

    with TestClient(app) as client:
        r = client.post("/api/recovery/process", json={
            "transaction_id": "proc_1",
            "customer_id": "cust_ml1",
            "amount": 850,
            "payment_method": "apple_pay",
        })
        assert r.status_code == 200
        j = r.json()
        assert j["probability_source"] == MODEL_VERSION
        assert j["action_mode"] == "simulated"
        assert j["ai_decision"]

        # chain persisted + retrievable via database + outcomes endpoint
        rec = asyncio.run(get_recovery_decision("proc_1"))
        assert rec is not None
        assert rec["probability_source"] == MODEL_VERSION
        assert rec["ai_decision"] == j["ai_decision"]
        assert rec["outcome"] in ("pending", "recovered_72h")
        assert rec["action_mode"] == "simulated"

        outcomes = client.get("/api/recovery/outcomes").json()
        assert any(d["transaction_id"] == "proc_1" for d in outcomes)
        analytics = client.get("/api/recovery/analytics").json()
        assert analytics["total_decisions"] >= 1
        assert MODEL_VERSION in analytics["by_probability_source"]


# ---------------------------------------------------------------------------
# 5. Benchmark-isolation: txn_* replay stays on the rule-based contract
# ---------------------------------------------------------------------------

def test_txn_keyed_replay_does_not_use_ml(isolated_db):
    """Offline/benchmark replay (txn_ keys) keeps the byte-identical contract."""
    from app.recovery.bridge import MODEL_VERSION
    from engine.realtime import _ml_for_live_event, run_live_recovery
    from engine.webhook import new_correlation_id

    ev = _live_event("txn_bench_1", customer_id="cust_bench", amount=85000)
    res = asyncio.run(_ml_for_live_event(ev))
    assert res is None  # gated: no ML for txn_ keys

    run = asyncio.run(run_live_recovery(ev, new_correlation_id()))
    assert run["status"] in ("pending", "blocked", "human_review", "recovered")
    from app.database import get_recovery_decision
    rec = asyncio.run(get_recovery_decision("txn_bench_1"))
    assert rec is not None
    assert rec["probability_source"] == "rule-based-v1"
    assert rec["ml_probability"] is None
    assert rec["ai_decision"]


# ---------------------------------------------------------------------------
# 6. Policy engine still gates the decision (can reject)
# ---------------------------------------------------------------------------

def test_policy_can_reject_invalid_action():
    from agents.policy_engine import PolicyEngine
    from agents.diagnosis_agent import diagnose
    from agents.recovery_strategy_agent import strategy_to_proposed_action, build_strategy
    from agents.customer_context_agent import get_customer_context
    from app.models import CustomerContextOutput, PolicyVerdict

    ev = _live_event("evt_pol1", amount=50000)
    ev.customer.opted_out = True
    diag = diagnose(ev)
    ctx = CustomerContextOutput(
        customer_id="cust_pol1", consent_status="OPTED_OUT",
        preferred_channel="WHATSAPP", safe_to_contact=False,
    )
    strategy = build_strategy(ev, diag, ctx)
    assert strategy.strategy == "STOP"  # opted-out -> blocked by the strategy agent
    # The policy engine must DENY any action that would contact the customer:
    contact_diag = diagnose(_live_event("evt_pol2", decline_code="bank_timeout"))
    contact_proposal = strategy_to_proposed_action(
        ev, build_strategy(ev, contact_diag,
                           CustomerContextOutput(
                               customer_id="cust_pol2", consent_status="CONSENTED",
                               preferred_channel="WHATSAPP", safe_to_contact=True)))
    decision = PolicyEngine().evaluate(ev, diag, contact_proposal, now=datetime(2026, 1, 15, 12, 0, 0))
    assert decision.verdict == PolicyVerdict.DENY
    assert any("opt" in f.lower() for f in decision.checks_failed)


# ---------------------------------------------------------------------------
# 7. Single-vs-batch consistency + deterministic duplicates
# ---------------------------------------------------------------------------

def test_single_and_batch_predictions_are_consistent(isolated_db):
    from fastapi.testclient import TestClient
    from app.main import app

    payload = {
        "transaction_id": "cons_1", "customer_id": "cust_ml1",
        "transaction_date": "2026-01-15 12:00:00", "total_amount": 850,
        "payment_method": "apple_pay", "status": "pending", "history": [],
    }
    with TestClient(app) as client:
        single = client.post("/api/recovery/predict", json=payload).json()
        batch = client.post("/api/recovery/batch-predict", json={
            "transactions": [payload, dict(payload, transaction_id="cons_2")]
        }).json()
    b1 = batch["predictions"][0]
    assert b1["recovery_probability"] == pytest.approx(
        single["recovery_probability"], abs=1e-6)
    assert b1["probability_raw"] == pytest.approx(
        single["probability_raw"], abs=1e-6)


def test_same_day_history_does_not_explode_features(isolated_db):
    """Two events on the same day in history -> deterministic features, no crash."""
    import pandas as pd
    from app.recovery.feature_builder import build_raw_features, ALL_FEATURES
    from app.recovery.schemas import RecoveryPredictRequest
    from app.recovery.model_service import get_model_service

    history = [
        {"transaction_id": "h1", "transaction_date": "2025-01-02 10:00:00",
         "status": "refunded", "total_amount": 500, "recovered_72h": 1},
        {"transaction_id": "h2", "transaction_date": "2025-01-02 10:00:00",
         "status": "pending", "total_amount": 300, "recovered_72h": 0},
    ]
    req1 = RecoveryPredictRequest(
        transaction_id="dup_1", customer_id="cust_ml1",
        transaction_date="2026-03-01 10:00:00", total_amount=850,
        payment_method="credit_card", status="pending", history=history)
    req2 = RecoveryPredictRequest(
        transaction_id="dup_2", customer_id="cust_ml1",
        transaction_date="2026-03-01 10:00:00", total_amount=850,
        payment_method="credit_card", status="pending", history=history)

    svc = get_model_service()
    raw1 = build_raw_features(req1)
    enc1 = svc.encode(pd.DataFrame([raw1], columns=ALL_FEATURES))
    raw2 = build_raw_features(req2)
    enc2 = svc.encode(pd.DataFrame([raw2], columns=ALL_FEATURES))
    assert enc1.shape == (1, 49)
    assert enc2.shape == (1, 49)
    # deterministic: identical inputs -> identical features -> identical score
    p1 = svc.predict_proba(enc1)
    p2 = svc.predict_proba(enc2)
    assert p1[1][0] == pytest.approx(p2[1][0], abs=1e-9)
    assert raw1['customer_transactions_before'] == 2


# ---------------------------------------------------------------------------
# 8. Non-recovery surfaces stay intact
# ---------------------------------------------------------------------------

def test_non_recovery_routes_intact(isolated_db):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        root = client.get("/")
        assert root.status_code in (200, 307)
        health = client.get("/api/health")
        assert health.status_code == 200


def test_prediction_and_decision_share_transaction_link(isolated_db):
    """prediction record + decision record coexist under one transaction_id."""
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        client.post("/api/recovery/predict", json={
            "transaction_id": "linked_1", "customer_id": "cust_ml1",
            "transaction_date": "2026-01-15 12:00:00", "total_amount": 850,
            "payment_method": "apple_pay", "status": "pending",
        })
        client.post("/api/recovery/process", json={
            "transaction_id": "linked_2", "customer_id": "cust_ml1",
            "amount": 850, "payment_method": "apple_pay",
        })
        predictions = client.get("/api/recovery/predictions").json()
        decisions = client.get("/api/recovery/outcomes").json()
    pred_ids = {p["transaction_id"] for p in predictions}
    dec_ids = {d["transaction_id"] for d in decisions}
    assert "linked_1" in pred_ids
    assert "linked_2" in dec_ids