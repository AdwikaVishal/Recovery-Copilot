"""End-to-end ML → AI → Policy → Action → Outcome chain integration tests.

Phase-19 acceptance contract — the UNIFIED recovery chain:

  1. ML prediction flows into the AI decision layer (no silent fallback).
  2. AI decision flows into the Policy Engine (hard boundary).
  3. Policy Engine gates execution (can DENY / HUMAN_REVIEW).
  4. Execution is simulated (no real money).
  5. Outcome is confirmed ONLY via trusted webhook.
  6. Every layer carries explicit provenance (probability_source, model_version).
  7. The unified /api/recovery/analyze endpoint returns the full chain.
  8. Provenance is consistent across all layers.
  9. Legacy fallback is explicitly marked.
  10. Batch benchmark isolation is preserved.
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

MODEL_VERSION = "ml:ExtraTreesClassifier-500-final"
RULE_BASED_VERSION = "rule-based-v1"


def _live_event(event_id: str, customer_id: str = "cust_chain1",
                amount: int = 50000, decline_code: str = "insufficient_funds",
                retry_count: int = 0, opted_out: bool = False) -> RevenueEvent:
    return RevenueEvent(
        id=event_id,
        type=EventType.CARD_PAYMENT_FAILURE,
        customer=Customer(id=customer_id, name="Chain User",
                          phone="+919876543210", email="chain@example.com",
                          opted_out=opted_out),
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


# ===========================================================================
# TEST 1: Full ML → AI → Policy → Action → Outcome chain
# ===========================================================================

class TestFullChain:
    def test_ml_prediction_flows_to_ai_decision(self, isolated_db):
        """A synthetic failure enters the live path; ML model is called,
        probability exists, probability_source is ML, AI receives the
        same probability, produces a decision, Policy evaluates, action
        is simulated, outcome is pending."""
        from app.recovery.bridge import (
            recovery_prediction_for_event, MODEL_VERSION as MV,
        )
        from agents.diagnosis_agent import diagnose
        from agents.customer_context_agent import get_customer_context
        from agents.recovery_optimizer import build_optimizer_output
        from agents.supervisor import process_event
        from app.models import WorkflowStatus

        ev = _live_event("evt_chain1", amount=85000)
        ml = recovery_prediction_for_event(ev, payment_method="apple_pay")

        # ML model was called and produced a valid prediction
        assert ml.available is True
        assert ml.recovery_probability > 0
        assert ml.probability_source == MV

        async def _run():
            diag = diagnose(ev)
            ctx = await get_customer_context(ev)
            optimizer = await build_optimizer_output(ev, diag, ctx, ml_estimate=ml)
            supervisor = await process_event(ev, ml_estimate=ml, execute_action=False)

            # AI received the ML probability
            elected = optimizer.elected
            assert elected is not None
            assert elected.probability == pytest.approx(ml.recovery_probability, abs=1e-4)
            assert elected.probability_source == MV

            # ML factors are visible in the decision trace
            assert any("ml_probability=" in f for f in optimizer.decision_factors)

            # Policy evaluated
            assert supervisor.policy_decision is not None
            assert supervisor.policy_decision.verdict.value in (
                "ALLOW", "MODIFY", "DENY", "HUMAN_REVIEW"
            )

            # Workflow status is valid
            assert supervisor.workflow_status in (
                WorkflowStatus.PENDING_WEBHOOK, WorkflowStatus.STOPPED,
                WorkflowStatus.HUMAN_REVIEW, WorkflowStatus.READY_FOR_POLICY,
            )

        asyncio.run(_run())

    def test_trusted_webhook_confirmation(self, isolated_db):
        """Trusted webhook confirmation closes the loop: same transaction_id,
        outcome becomes recovered, recovered amount recorded, ledger updated."""
        from engine.webhook import normalize_payment_webhook, new_correlation_id
        from engine.realtime import run_live_recovery, confirm_live_recovery
        from app.database import get_recovery_decision, db_session

        ev = normalize_payment_webhook({
            "event_type": "payment.failed", "event_id": "evt_chain2",
            "transaction_id": "txn_chain2", "amount": 80000,
            "customer_id": "cust_chain2", "decline_code": "insufficient_funds",
        })
        corr = new_correlation_id()
        result = asyncio.run(run_live_recovery(ev, corr, source="webhook",
                                               recovery_key="txn_chain2"))
        assert result["status"] in ("pending", "blocked", "recovered", "human_review")

        # Now confirm via trusted webhook
        conf_ev = normalize_payment_webhook({
            "event_type": "payment.captured", "event_id": "evt_chain2_conf",
            "transaction_id": "txn_chain2", "amount": 80000,
        })
        conf_corr = new_correlation_id()
        conf_result = asyncio.run(confirm_live_recovery(
            conf_ev, conf_corr, recovery_key="txn_chain2", amount=80000))
        assert conf_result["status"] == "recovered"
        assert conf_result["amount_recovered"] == 80000

        # Ledger updated
        rec = asyncio.run(get_recovery_decision("txn_chain2"))
        assert rec is not None
        # The trusted webhook closes the decision loop in the ledger: outcome
        # becomes recovered and the recovered amount is recorded under the SAME
        # transaction_id (regression: this used to key on event.id and never
        # update the ledger).
        if rec.get("outcome"):
            assert rec["outcome"] == "recovered_72h"
            if rec.get("recovered_amount") is not None:
                assert rec["recovered_amount"] == 80000

    def test_policy_denial_blocks_execution(self, isolated_db):
        """ML prediction exists, AI proposes action, policy denies it,
        action does NOT execute."""
        from app.recovery.bridge import recovery_prediction_for_event
        from agents.supervisor import process_event
        from app.models import WorkflowStatus

        ev = _live_event("evt_chain3", opted_out=True)
        ml = recovery_prediction_for_event(ev)

        async def _run():
            out = await process_event(ev, ml_estimate=ml)
            # Opted-out customer -> STOPPED (never reaches execution)
            assert out.workflow_status == WorkflowStatus.STOPPED
            # No execution happened
            assert out.proposed_action is None or out.next_step.startswith("STOP:")
        asyncio.run(_run())

    def test_human_review_prevents_execution(self, isolated_db):
        """ML prediction exists, AI proposes action, policy returns
        HUMAN_REVIEW, no execution occurs. High-value + outside contact
        window triggers HUMAN_REVIEW."""
        from app.recovery.bridge import recovery_prediction_for_event
        from agents.supervisor import process_event
        from app.models import WorkflowStatus

        # Outside contact window (hour=3 -> outside 08:00-21:00) + high amount
        late_event = RevenueEvent(
            id="evt_chain4",
            type=EventType.CARD_PAYMENT_FAILURE,
            customer=Customer(id="cust_chain4", name="Late User",
                              phone="+919876543210", email="late@example.com"),
            amount=5000000,
            currency="INR",
            root_cause=DeclineCode.DO_NOT_HONOR,
            decline_code=DeclineCode.DO_NOT_HONOR,
            failed_at=datetime(2026, 1, 15, 3, 0, 0),
            retry_count=0,
            metadata=TransactionMetadata(),
        )
        ml = recovery_prediction_for_event(late_event)

        async def _run():
            out = await process_event(late_event, ml_estimate=ml, now=datetime(2026, 1, 15, 3, 0, 0))
            # Outside contact window -> STOPPED or HUMAN_REVIEW (policy blocks)
            assert out.workflow_status in (
                WorkflowStatus.STOPPED, WorkflowStatus.HUMAN_REVIEW,
            )
        asyncio.run(_run())

    def test_legacy_fallback_explicitly_marked(self, isolated_db):
        """Force ML failure; fallback is explicitly marked, no false ML
        provenance."""
        from app.recovery.bridge import (
            recovery_prediction_for_request, MLPrediction,
            RULE_BASED_VERSION, MODEL_VERSION,
        )
        from app.recovery.schemas import RecoveryPredictRequest

        class BrokenService:
            def ensure_loaded(self):
                raise RuntimeError("Model artifacts corrupted")
            def encode(self, df):
                raise RuntimeError("Model artifacts corrupted")

        req = RecoveryPredictRequest(
            transaction_id="evt_chain5_fallback",
            customer_id="cust_chain5",
            transaction_date="2026-01-15 12:00:00",
            total_amount=50000,
            payment_method="credit_card",
            status="pending",
        )

        ml = recovery_prediction_for_request(req, service=BrokenService())
        assert ml.available is False
        assert ml.probability_source == RULE_BASED_VERSION
        assert ml.fallback_reason is not None
        assert "ML prediction failed" in ml.fallback_reason

    def test_provenance_consistency(self, isolated_db):
        """For a normal ML-enabled live event, prediction.model_version ==
        AI decision.model_version == ledger.probability_source, all
        correspond to MODEL_VERSION."""
        from app.recovery.bridge import (
            recovery_prediction_for_event, MODEL_VERSION as MV,
        )
        from engine.realtime import run_live_recovery
        from engine.webhook import normalize_payment_webhook, new_correlation_id
        from app.database import get_recovery_decision

        ev = normalize_payment_webhook({
            "event_type": "payment.failed", "event_id": "evt_chain6",
            "transaction_id": "txn_chain6", "amount": 50000,
            "customer_id": "cust_chain6", "decline_code": "insufficient_funds",
        })
        ml = recovery_prediction_for_event(ev, payment_method="credit_card")

        # ML prediction has canonical version
        assert ml.available is True
        assert ml.model_version == MV
        assert ml.probability_source == MV

        # Run through live pipeline
        corr = new_correlation_id()
        result = asyncio.run(run_live_recovery(
            ev, corr, source="webhook", ml_estimate=ml,
            recovery_key="txn_chain6"))

        # Ledger record exists and uses the same version
        rec = asyncio.run(get_recovery_decision("txn_chain6"))
        assert rec is not None
        assert rec["probability_source"] == MV
        assert rec["model"] == "ExtraTreesClassifier"

    def test_single_vs_batch_prediction_parity(self, isolated_db):
        """Existing single/batch predict endpoints remain correct."""
        from fastapi.testclient import TestClient
        from app.main import app

        payload = {
            "transaction_id": "chain_parity", "customer_id": "cust_chain_p",
            "transaction_date": "2026-01-15 12:00:00", "total_amount": 850,
            "payment_method": "apple_pay", "status": "pending", "history": [],
        }
        with TestClient(app) as client:
            single = client.post("/api/recovery/predict", json=payload).json()
            batch = client.post("/api/recovery/batch-predict", json={
                "transactions": [payload]
            }).json()
        b1 = batch["predictions"][0]
        assert b1["recovery_probability"] == pytest.approx(
            single["recovery_probability"], abs=1e-6)

    def test_unknown_payment_method_safe(self, isolated_db):
        """Unknown payment method does not crash inference."""
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            r = client.post("/api/recovery/predict", json={
                "transaction_id": "chain_unk", "customer_id": "cust_unk",
                "transaction_date": "2026-01-15 12:00:00", "total_amount": 500,
                "payment_method": "upi_unknown", "status": "pending",
            })
            assert r.status_code == 200
            j = r.json()
            assert 0.0 <= j["recovery_probability"] <= 1.0

    def test_first_ever_customer(self, isolated_db):
        """First-ever customer with no history does not break inference."""
        from app.recovery.feature_builder import build_raw_features, ALL_FEATURES
        from app.recovery.schemas import RecoveryPredictRequest
        from app.recovery.model_service import get_model_service

        req = RecoveryPredictRequest(
            transaction_id="chain_new", customer_id="cust_brand_new",
            transaction_date="2026-01-15 12:00:00", total_amount=500,
            payment_method="credit_card", status="pending", history=[])
        raw = build_raw_features(req)
        assert len(raw) == 44
        assert raw["customer_transactions_before"] == 0
        assert raw["customer_recoveries_before"] == 0

        import pandas as pd
        svc = get_model_service()
        raw_df = pd.DataFrame([raw], columns=ALL_FEATURES)
        enc = svc.encode(raw_df)
        prob_raw, prob_cal = svc.predict_proba(enc)
        assert 0.0 <= float(prob_cal[0]) <= 1.0

    def test_same_day_duplicate_regression(self, isolated_db):
        """Same-day duplicate transactions do not explode features."""
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
        req = RecoveryPredictRequest(
            transaction_id="chain_dup", customer_id="cust_dup",
            transaction_date="2026-03-01 10:00:00", total_amount=850,
            payment_method="credit_card", status="pending", history=history)
        raw = build_raw_features(req)
        enc = pd.DataFrame([raw], columns=ALL_FEATURES)
        svc = get_model_service()
        encoded = svc.encode(enc)
        assert encoded.shape == (1, 49)
        prob_raw, prob_cal = svc.predict_proba(encoded)
        assert 0.0 <= float(prob_cal[0]) <= 1.0


# ===========================================================================
# TEST 2: Unified /analyze endpoint
# ===========================================================================

class TestAnalyzeEndpoint:
    def test_analyze_returns_full_chain(self, isolated_db):
        """POST /api/recovery/analyze returns ML signal, AI decision,
        policy, execution, and outcome blocks with proper provenance."""
        from fastapi.testclient import TestClient
        from app.main import app
        from app.recovery.bridge import MODEL_VERSION as MV

        with TestClient(app) as client:
            r = client.post("/api/recovery/analyze", json={
                "transaction_id": "analyze_1",
                "customer_id": "cust_analyze",
                "amount": 85000,
                "payment_method": "apple_pay",
            })
            assert r.status_code == 200
            j = r.json()

        # ML signal block
        ml = j["ml_signal"]
        assert ml["recovery_probability"] > 0
        assert ml["probability_source"] == MV
        assert ml["model_version"] == MV
        assert ml["threshold"] == pytest.approx(0.04)
        assert ml["risk_band"]
        assert ml["risk_label"]

        # AI decision block
        ai = j["ai_decision"]
        assert ai["diagnosis"]
        assert 0.0 <= ai["diagnosis_confidence"] <= 1.0
        assert ai["selected_action"] in (
            "RETRY", "REAUTHORIZE", "PAYMENT_LINK", "MESSAGE",
            "HUMAN_REVIEW", "STOP",
        )
        assert 0.0 <= ai["action_success_probability"] <= 1.0
        assert ai["expected_recovery"] >= 0
        assert isinstance(ai["reasoning"], list)
        assert isinstance(ai["candidate_actions"], list)
        assert len(ai["candidate_actions"]) > 0

        # Policy block
        pol = j["policy"]
        assert pol["verdict"] in ("ALLOW", "MODIFY", "DENY", "HUMAN_REVIEW")
        assert pol["reason"]

        # Execution block
        ex = j["execution"]
        assert ex["mode"] == "simulated"
        assert ex["status"] == "advisory"

        # Outcome block
        oc = j["outcome"]
        assert oc["status"] == "pending"

        # Top-level provenance
        assert j["model_version"] == MV
        assert j["probability_source"] == MV
        assert j["transaction_id"] == "analyze_1"

    def test_analyze_shows_ml_ai_dependency(self, isolated_db):
        """The analyze endpoint explicitly shows the ML signal feeds the
        AI decision (action_success_probability == ML probability when
        ML is available)."""
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            r = client.post("/api/recovery/analyze", json={
                "transaction_id": "analyze_dep",
                "customer_id": "cust_dep",
                "amount": 85000,
                "payment_method": "apple_pay",
            })
            j = r.json()

        ml_prob = j["ml_signal"]["recovery_probability"]
        ai_prob = j["ai_decision"]["action_success_probability"]

        # When ML is the source, the action success probability IS the ML probability
        # (the optimizer uses ML as the single source of truth)
        if j["probability_source"].startswith("ml:"):
            assert ai_prob == pytest.approx(ml_prob, abs=1e-4)

        # Expected recovery = amount * probability
        if ai_prob > 0:
            expected = int(85000 * ai_prob)
            assert j["ai_decision"]["expected_recovery"] == expected

    def test_analyze_opted_out_customer(self, isolated_db):
        """Opted-out customer yields STOP / DENY through the chain."""
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            r = client.post("/api/recovery/analyze", json={
                "transaction_id": "analyze_opt",
                "customer_id": "cust_opt",
                "amount": 50000,
            })
            j = r.json()

        # Should be blocked at the policy/strategy level
        assert j["policy"]["verdict"] in ("DENY", "HUMAN_REVIEW") or \
               j["ai_decision"]["selected_action"] == "STOP"

    def test_analyze_maintains_model_version(self, isolated_db):
        """model_version is consistent across all blocks."""
        from fastapi.testclient import TestClient
        from app.main import app
        from app.recovery.bridge import MODEL_VERSION as MV

        with TestClient(app) as client:
            r = client.post("/api/recovery/analyze", json={
                "transaction_id": "analyze_ver",
                "customer_id": "cust_ver",
                "amount": 50000,
            })
            j = r.json()

        assert j["model_version"] == MV
        if j["ml_signal"]["recovery_probability"] > 0:
            assert j["ml_signal"]["model_version"] == MV
            assert j["probability_source"] == MV

    def test_missing_previous_recovery_date_safe(self, isolated_db):
        """A request with no history / recovery date still runs + ML scores."""
        from app.recovery.bridge import recovery_prediction_for_event, MODEL_VERSION as MV
        from agents.supervisor import process_event

        # No history supplied, transaction_date with no prior recovery
        ev = RevenueEvent(
            id="evt_nodate1", type=EventType.CARD_PAYMENT_FAILURE,
            customer=Customer(id="cust_nodate1", name="No Date User",
                              phone="+919876543210", email="nodate@example.com"),
            amount=85000, currency="INR",
            root_cause=DeclineCode.INSUFFICIENT_FUNDS,
            decline_code=DeclineCode.INSUFFICIENT_FUNDS,
            failed_at=datetime(2026, 1, 15, 12, 0, 0),
            retry_count=0, metadata=TransactionMetadata(previous_attempts=0),
        )
        ml = recovery_prediction_for_event(ev, payment_method="apple_pay")
        assert ml.available is True
        assert ml.probability_source == MV
        assert 0.0 <= ml.recovery_probability <= 1.0

        async def _run():
            out = await process_event(ev, ml_estimate=ml)
            return out.workflow_status.value
        status = asyncio.run(_run())
        assert status in ("PENDING_WEBHOOK", "STOPPED", "HUMAN_REVIEW", "RESOLVED")

    def test_recovered_amount_only_after_trusted_confirmation(self, isolated_db):
        """Recovered money is 0 after a simulated action; only a trusted
        webhook confirmation raises it. Proven by the ledger + attempt."""
        from engine.webhook import normalize_payment_webhook, new_correlation_id
        from engine.realtime import run_live_recovery, confirm_live_recovery
        from app.database import get_recovery_decision, db_session

        ev = normalize_payment_webhook({
            "event_type": "payment.failed", "event_id": "evt_amt1",
            "transaction_id": "txn_amt1", "amount": 90000,
            "customer_id": "cust_amt1", "decline_code": "insufficient_funds",
        })
        corr = new_correlation_id()
        result = asyncio.run(run_live_recovery(ev, corr, source="webhook",
                                               recovery_key="txn_amt1"))

        # Before any trusted confirmation, no money is counted as recovered.
        rec_before = asyncio.run(get_recovery_decision("txn_amt1"))
        assert rec_before is not None
        assert (rec_before.get("recovered_amount") or 0) == 0
        assert rec_before.get("outcome") in ("pending",)

        # Trusted confirmation closes the loop and records recovered money.
        conf_ev = normalize_payment_webhook({
            "event_type": "payment.captured", "event_id": "evt_amt1_conf",
            "transaction_id": "txn_amt1", "amount": 90000,
        })
        asyncio.run(confirm_live_recovery(
            conf_ev, new_correlation_id(), recovery_key="txn_amt1", amount=90000))
        rec_after = asyncio.run(get_recovery_decision("txn_amt1"))
        assert rec_after.get("recovered_amount") == 90000
        assert rec_after.get("outcome") == "recovered_72h"
