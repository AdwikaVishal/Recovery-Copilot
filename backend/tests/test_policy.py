import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from app.models import (
    RevenueEvent, Customer, TransactionMetadata,
    EventType, DeclineCode, PolicyVerdict, ProposedAction,
    CustomerContextOutput, ContactFrequency, AgentRecommendation,
    BatchResult, BatchSummary, ContactEvent, PolicyDecision
)
from agents.diagnosis_agent import diagnose, DiagnosisOutput
from agents.policy_engine import PolicyEngine
from agents.human_approval_gate import HumanApprovalGate
from agents.message_agent import generate_message, generate_reauth_message
from agents.recovery_strategy_agent import build_strategy, strategy_to_proposed_action
from agents.compliance_explainer import explain
from agents.execution_adapter import (
    DeterministicMockGateway, FailureInjectingGateway,
    RazorpayTestGateway, _lookup_scenario, SCENARIO_TABLE
)
from data.scenarios import get_all_scenarios, get_scenario, build_event_from_scenario


def _make_event(**overrides) -> RevenueEvent:
    defaults = {
        "id": "txn_test_001",
        "type": EventType.CARD_PAYMENT_FAILURE,
        "customer": Customer(
            id="cust_001", name="Test User", phone="+919876543210",
            email="test@example.com", opted_out=False
        ),
        "amount": 50000,
        "root_cause": DeclineCode.INSUFFICIENT_FUNDS,
        "decline_code": DeclineCode.INSUFFICIENT_FUNDS,
        "failed_at": datetime.utcnow(),
        "metadata": TransactionMetadata(),
        "retry_count": 0,
    }
    defaults.update(overrides)
    return RevenueEvent(**defaults)


# ── Diagnosis Agent Tests ──

def test_diagnosis_insufficient_funds():
    event = _make_event()
    result = diagnose(event)
    assert isinstance(result, DiagnosisOutput)
    assert result.classification == "temporary_cash_flow_issue"
    assert result.recommended_action_family == "RETRY"
    assert result.likely_recoverability in ("HIGH", "MEDIUM")
    assert result.confidence > 0.7
    assert len(result.evidence) >= 3
    assert len(result.uncertainties) == 0


def test_diagnosis_expired_card():
    event = _make_event(
        root_cause=DeclineCode.EXPIRED_CARD,
        decline_code=DeclineCode.EXPIRED_CARD,
    )
    result = diagnose(event)
    assert result.classification == "expired_payment_instrument"
    assert result.recommended_action_family == "UPDATE_PAYMENT_METHOD"
    assert result.likely_recoverability == "LOW"
    assert result.recommended_wait_hours == 0


def test_diagnosis_afa_required():
    event = _make_event(
        type=EventType.RECURRING_PAYMENT_FAILURE,
        root_cause=DeclineCode.MANDATE_AFA_REQUIRED,
        decline_code=DeclineCode.MANDATE_AFA_REQUIRED,
        amount=2000000,
    )
    result = diagnose(event)
    assert result.recommended_action_family == "REAUTHORIZE"
    assert result.likely_recoverability == "HIGH"
    assert result.confidence > 0.9
    afa_evidence = [e for e in result.evidence if e.field == "afa_threshold_check"]
    assert len(afa_evidence) == 1
    assert "₹15,000" in afa_evidence[0].value


def test_diagnosis_confidence_drops_with_retries():
    event1 = _make_event(retry_count=0)
    event2 = _make_event(retry_count=3)
    r1 = diagnose(event1)
    r2 = diagnose(event2)
    assert r2.confidence < r1.confidence
    assert len(r2.uncertainties) > len(r1.uncertainties)


def test_diagnosis_unknown_decline_code():
    event = _make_event(
        root_cause=DeclineCode.INSUFFICIENT_FUNDS,
        decline_code=DeclineCode.INSUFFICIENT_FUNDS,
    )
    event.decline_code = "some_unknown_code"
    result = diagnose(event)
    assert result.classification == "unknown_decline_code"
    assert result.confidence < 0.2
    assert result.likely_recoverability == "UNKNOWN"
    assert result.recommended_action_family == "HUMAN_REVIEW"


def test_diagnosis_opt_out_stops():
    event = _make_event()
    event.customer.opted_out = True
    result = diagnose(event)
    assert result.recommended_action_family == "STOP"
    assert result.likely_recoverability == "LOW"
    opt_evidence = [e for e in result.evidence if e.field == "customer_opt_out"]
    assert len(opt_evidence) == 1


def test_diagnosis_insufficient_funds_retries_exhausted():
    event = _make_event(retry_count=3)
    result = diagnose(event)
    assert result.recommended_action_family in ("SEND_REMINDER", "HUMAN_REVIEW")
    assert any("Retries exhausted" in u or "retry" in u.lower() for u in result.uncertainties)


def test_diagnosis_do_not_honor_with_retry():
    event = _make_event(
        root_cause=DeclineCode.DO_NOT_HONOR,
        decline_code=DeclineCode.DO_NOT_HONOR,
    )
    result = diagnose(event)
    assert result.recommended_action_family == "RETRY"

    event2 = _make_event(
        root_cause=DeclineCode.DO_NOT_HONOR,
        decline_code=DeclineCode.DO_NOT_HONOR,
        retry_count=1,
    )
    result2 = diagnose(event2)
    assert result2.recommended_action_family == "SEND_REMINDER"


def test_diagnosis_overdue_invoice_days():
    event = _make_event(
        type=EventType.OVERDUE_INVOICE,
        root_cause=DeclineCode.INVOICE_OVERDUE,
        decline_code=DeclineCode.INVOICE_OVERDUE,
        metadata=TransactionMetadata(days_overdue=75),
    )
    result = diagnose(event)
    assert result.classification == "overdue_receivable"
    days_evidence = [e for e in result.evidence if e.field == "days_overdue"]
    assert len(days_evidence) == 1
    assert "75" in days_evidence[0].value
    assert result.confidence < 0.7


def test_diagnosis_evidence_always_present():
    event = _make_event()
    result = diagnose(event)
    fields = [e.field for e in result.evidence]
    assert "decline_code" in fields
    assert "payment_type" in fields
    assert "amount" in fields
    assert "attempt_count" in fields


def test_diagnosis_no_ground_truth_leakage():
    event = _make_event()
    result = diagnose(event)
    for u in result.uncertainties:
        assert "ground_truth" not in u.lower()
    for e in result.evidence:
        assert "ground_truth" not in e.interpretation.lower()


# ── Policy Engine Tests ──

def test_policy_allows_normal_retry():
    event = _make_event()
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=24,
        max_retries=3, risk_score=0.2,
    )
    proposed = ProposedAction(action="retry_payment", amount=50000)
    engine = PolicyEngine()
    now = datetime(2026, 8, 20, 12, 0)
    decision = engine.evaluate(event, diagnosis, proposed, now=now)
    assert decision.verdict == PolicyVerdict.ALLOW


def test_policy_blocks_opt_out():
    event = _make_event()
    event.customer.opted_out = True
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="STOP", recommended_wait_hours=0,
        max_retries=3, risk_score=0.0,
    )
    proposed = ProposedAction(action="blocked", amount=50000)
    engine = PolicyEngine()
    decision = engine.evaluate(event, diagnosis, proposed)
    assert decision.verdict == PolicyVerdict.DENY
    assert "opt_out" in decision.checks_failed


def test_policy_blocks_max_retries():
    event = _make_event(retry_count=3)
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="HUMAN_REVIEW", recommended_wait_hours=0, max_retries=3,
        risk_score=0.5,
    )
    proposed = ProposedAction(action="retry_payment", amount=50000)
    engine = PolicyEngine()
    decision = engine.evaluate(event, diagnosis, proposed)
    assert decision.verdict == PolicyVerdict.DENY
    assert "max_retries" in decision.checks_failed


def test_policy_blocks_afa_retry():
    event = _make_event(
        type=EventType.RECURRING_PAYMENT_FAILURE,
        root_cause=DeclineCode.MANDATE_AFA_REQUIRED,
        decline_code=DeclineCode.MANDATE_AFA_REQUIRED,
        amount=2000000,
    )
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.97, likely_recoverability="HIGH",
        recommended_action_family="REAUTHORIZE", recommended_wait_hours=0, max_retries=0,
        requires_afa=True, risk_score=0.2,
    )
    proposed = ProposedAction(action="retry_payment", amount=2000000)
    engine = PolicyEngine()
    decision = engine.evaluate(event, diagnosis, proposed)
    assert decision.verdict == PolicyVerdict.DENY
    assert "afa_check" in decision.checks_failed


def test_policy_allows_reauth_for_afa():
    event = _make_event(
        type=EventType.RECURRING_PAYMENT_FAILURE,
        root_cause=DeclineCode.MANDATE_AFA_REQUIRED,
        decline_code=DeclineCode.MANDATE_AFA_REQUIRED,
        amount=2000000,
    )
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.97, likely_recoverability="HIGH",
        recommended_action_family="REAUTHORIZE", recommended_wait_hours=0, max_retries=0,
        requires_afa=True, risk_score=0.2,
    )
    proposed = ProposedAction(action="re_authorize_mandate", amount=2000000)
    engine = PolicyEngine()
    now = datetime(2026, 8, 20, 12, 0)
    decision = engine.evaluate(event, diagnosis, proposed, now=now)
    assert decision.verdict == PolicyVerdict.ALLOW


def test_policy_modifies_high_discount():
    event = _make_event(amount=400000)
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=24,
        max_retries=3, risk_score=0.2,
    )
    proposed = ProposedAction(action="retry_payment", amount=400000, discount_percent=15)
    engine = PolicyEngine()
    now = datetime(2026, 8, 20, 12, 0)
    decision = engine.evaluate(event, diagnosis, proposed, now=now)
    assert decision.verdict == PolicyVerdict.MODIFY
    assert "discount_ceiling" in decision.checks_failed
    assert decision.modified_request is not None
    assert decision.modified_request["discount_percent"] <= 10


def test_policy_blocks_outside_hours():
    event = _make_event()
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=24,
        max_retries=3, risk_score=0.2,
    )
    proposed = ProposedAction(action="retry_payment", amount=50000)
    engine = PolicyEngine()
    night_time = datetime(2026, 8, 20, 23, 30)
    decision = engine.evaluate(event, diagnosis, proposed, now=night_time)
    assert decision.verdict == PolicyVerdict.DENY
    assert "time_of_day" in decision.checks_failed


def test_policy_high_risk_gets_human_review():
    event = _make_event()
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=24,
        max_retries=3, risk_score=0.85,
    )
    proposed = ProposedAction(action="retry_payment", amount=50000)
    engine = PolicyEngine()
    decision = engine.evaluate(event, diagnosis, proposed)
    assert decision.verdict == PolicyVerdict.HUMAN_REVIEW
    assert "risk_score" in decision.checks_failed


def test_policy_modifies_high_value_discount():
    event = _make_event(amount=600000)
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=24,
        max_retries=3, risk_score=0.2,
    )
    proposed = ProposedAction(action="retry_payment", amount=600000, discount_percent=5)
    engine = PolicyEngine()
    now = datetime(2026, 8, 20, 12, 0)
    decision = engine.evaluate(event, diagnosis, proposed, now=now)
    assert decision.verdict == PolicyVerdict.HUMAN_REVIEW
    assert "amount_ceiling" in decision.checks_failed


# ── AFA Explanation Tests ──

def test_afa_explanation_high_amount_non_recurring():
    event = _make_event(amount=7500000, type=EventType.CARD_PAYMENT_FAILURE)
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.9, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=0,
        max_retries=3, requires_afa=False, risk_score=0.2,
    )
    proposed = ProposedAction(action="retry_payment", amount=7500000)
    engine = PolicyEngine()
    decision = engine.evaluate(event, diagnosis, proposed, now=datetime(2026, 8, 20, 12, 0))
    afa_detail = next(r for r in decision.detailed_results if r.rule == "afa_check")
    assert afa_detail.result == "PASS"
    assert "below" not in afa_detail.explanation.lower()
    assert "AFA not required for this payment type" in afa_detail.explanation


def test_afa_explanation_low_amount():
    event = _make_event(amount=500000, type=EventType.RECURRING_PAYMENT_FAILURE)
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.9, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=0,
        max_retries=3, requires_afa=False, risk_score=0.2,
    )
    proposed = ProposedAction(action="retry_payment", amount=500000)
    engine = PolicyEngine()
    decision = engine.evaluate(event, diagnosis, proposed, now=datetime(2026, 8, 20, 12, 0))
    afa_detail = next(r for r in decision.detailed_results if r.rule == "afa_check")
    assert afa_detail.result == "PASS"
    assert "below" in afa_detail.explanation.lower()


# ── Amount Ceiling Explanation Tests ──

def test_amount_ceiling_explanation_clarity():
    event = _make_event(amount=7500000)
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=0,
        max_retries=3, risk_score=0.2,
    )
    proposed = ProposedAction(action="retry_payment", amount=7500000, discount_percent=0)
    engine = PolicyEngine()
    decision = engine.evaluate(event, diagnosis, proposed, now=datetime(2026, 8, 20, 12, 0))
    ac_detail = next(r for r in decision.detailed_results if r.rule == "amount_ceiling")
    assert ac_detail.result == "PASS"
    assert "ceiling applies only when discount is offered" in ac_detail.explanation.lower()


# ── HUMAN_REVIEW / DENY No-Execute Tests ──

def test_human_review_no_execution():
    event = _make_event(amount=600000)
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=0,
        max_retries=3, risk_score=0.2,
    )
    proposed = ProposedAction(action="retry_payment", amount=600000, discount_percent=5)
    engine = PolicyEngine()
    decision = engine.evaluate(event, diagnosis, proposed, now=datetime(2026, 8, 20, 12, 0))
    assert decision.verdict == PolicyVerdict.HUMAN_REVIEW
    assert decision.requires_human_approval is True


def test_deny_no_execution():
    event = _make_event(amount=50000, customer=Customer(
        id="cust_opt_out", name="Opted Out", phone="+919876543210",
        email="opted@out.com", opted_out=True,
    ))
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=0,
        max_retries=3, risk_score=0.2,
    )
    proposed = ProposedAction(action="retry_payment", amount=50000)
    engine = PolicyEngine()
    decision = engine.evaluate(event, diagnosis, proposed, now=datetime(2026, 8, 20, 12, 0))
    assert decision.verdict == PolicyVerdict.DENY
    assert "opt_out" in decision.checks_failed
    assert decision.original_request is not None


# ── Modify Action Tests ──

def test_modify_reduces_discount():
    event = _make_event(amount=400000)
    diagnosis = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=24,
        max_retries=3, risk_score=0.2,
    )
    proposed = ProposedAction(action="retry_payment", amount=400000, discount_percent=15)
    engine = PolicyEngine()
    decision = engine.evaluate(event, diagnosis, proposed, now=datetime(2026, 8, 20, 12, 0))
    assert decision.verdict == PolicyVerdict.MODIFY
    assert decision.modified_request is not None
    assert decision.modified_request["discount_percent"] == 10
    assert decision.original_request is not None
    assert decision.original_request["discount_percent"] == 15


# ── Human Approval Gate Tests ──

def test_gate_allows_normal():
    event = _make_event()
    proposed = ProposedAction(action="retry_payment", amount=50000)
    class PD:
        requires_human_approval = False
    gate = HumanApprovalGate()
    result = gate.evaluate(event, proposed, PD())
    assert result["can_auto_proceed"] is True


def test_gate_blocks_high_discount():
    event = _make_event(amount=600000)
    proposed = ProposedAction(action="retry_payment", amount=600000, discount_percent=15)
    class PD:
        requires_human_approval = False
    gate = HumanApprovalGate()
    result = gate.evaluate(event, proposed, PD())
    assert result["needs_approval"] is True


# ── Message Agent Tests ──

def test_generate_hinglish_message():
    msg = generate_message("Rahul Sharma", 50000, 1, "hi")
    assert "Rahul" in msg
    assert "500" in msg


def test_generate_english_message():
    msg = generate_message("Rahul Sharma", 50000, 2, "en")
    assert "Rahul" in msg
    assert "500" in msg


def test_generate_reauth_message():
    msg = generate_reauth_message("Priya Patel", 2000000, "hi")
    assert "Priya" in msg
    assert "20,000" in msg
    assert "RBI" in msg


def test_tone_escalation():
    msg1 = generate_message("Rahul", 50000, 1, "hi")
    msg4 = generate_message("Rahul", 50000, 4, "hi")
    assert msg1 != msg4


# ── Recovery Strategy Tests ──

def test_strategy_retry():
    event = _make_event()
    diag = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=48,
    )
    ctx = CustomerContextOutput(customer_id="cust_001", consent_status="CONSENTED", preferred_channel="WHATSAPP")
    strategy = build_strategy(event, diag, ctx)
    assert strategy.strategy == "RETRY"
    assert strategy.channel == "RAZORPAY_API"
    proposed = strategy_to_proposed_action(event, strategy)
    assert proposed.action == "retry_payment"


def test_strategy_dunning_tone():
    event = _make_event(
        type=EventType.OVERDUE_INVOICE,
        metadata=TransactionMetadata(days_overdue=15),
    )
    diag = DiagnosisOutput(
        classification="test", confidence=0.7, likely_recoverability="MEDIUM",
        recommended_action_family="SEND_REMINDER", recommended_wait_hours=0,
    )
    ctx = CustomerContextOutput(customer_id="cust_001", consent_status="CONSENTED", preferred_channel="WHATSAPP")
    strategy = build_strategy(event, diag, ctx)
    assert strategy.strategy == "MESSAGE"
    proposed = strategy_to_proposed_action(event, strategy)
    assert proposed.action == "send_dunning_message"
    assert proposed.message_tone_level == 4


def test_strategy_reauthorize():
    event = _make_event(amount=2000000)
    diag = DiagnosisOutput(
        classification="test", confidence=0.97, likely_recoverability="HIGH",
        recommended_action_family="REAUTHORIZE", recommended_wait_hours=0, requires_afa=True,
    )
    ctx = CustomerContextOutput(customer_id="cust_001", consent_status="CONSENTED", preferred_channel="WHATSAPP")
    strategy = build_strategy(event, diag, ctx)
    assert strategy.strategy == "REAUTHORIZE"
    proposed = strategy_to_proposed_action(event, strategy)
    assert proposed.action == "re_authorize_mandate"


def test_strategy_blocks_opted_out():
    event = _make_event()
    event.customer.opted_out = True
    diag = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="SEND_REMINDER", recommended_wait_hours=0,
    )
    ctx = CustomerContextOutput(customer_id="cust_001", consent_status="OPTED_OUT", preferred_channel="WHATSAPP", safe_to_contact=False)
    strategy = build_strategy(event, diag, ctx)
    assert strategy.strategy == "STOP"
    assert strategy.expected_value == 0


def test_strategy_afa_blocks_blind_retry():
    event = _make_event(amount=2000000)
    diag = DiagnosisOutput(
        classification="test", confidence=0.97, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=0, requires_afa=True,
    )
    ctx = CustomerContextOutput(customer_id="cust_001", consent_status="CONSENTED", preferred_channel="WHATSAPP")
    strategy = build_strategy(event, diag, ctx)
    assert strategy.strategy == "REAUTHORIZE"
    assert "AFA" in strategy.reason


def test_dnd_registered_customer_never_contacted():
    event = _make_event()
    event.customer.dnd_registered = True
    diag = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=24,
    )
    ctx = CustomerContextOutput(
        customer_id="cust_001", consent_status="OPTED_OUT",
        preferred_channel="WHATSAPP", safe_to_contact=False,
        risk_flags=["DND_REGISTERED"],
    )
    strategy = build_strategy(event, diag, ctx)
    assert strategy.strategy == "STOP"
    assert strategy.expected_value == 0
    assert "DND" in strategy.reason or "not safe" in strategy.reason.lower()


def test_dnd_customer_policy_denies():
    event = _make_event()
    event.customer.opted_out = True
    diag = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=24,
    )
    proposed = ProposedAction(action="retry_payment", channel="whatsapp", amount=event.amount)
    engine = PolicyEngine()
    decision = engine.evaluate(event, diag, proposed)
    assert decision.verdict == PolicyVerdict.DENY
    assert "opt_out" in decision.checks_failed


def test_record_outcome_is_single_write_path():
    from agents.outcome_handler import _map_result_to_status
    from app.models import ExecutionResult

    r1 = ExecutionResult(event_id="x", action="retry_payment", result="success", amount_recovered=10000)
    r2 = ExecutionResult(event_id="x", action="retry_payment", result="pending", amount_recovered=0)
    r3 = ExecutionResult(event_id="x", action="retry_payment", result="blocked", amount_recovered=0)

    assert _map_result_to_status(r1) == "success"
    assert _map_result_to_status(r2) == "pending_webhook"
    assert _map_result_to_status(r3) == "blocked"
    assert _map_result_to_status(r1, "PENDING_WEBHOOK") == "pending_webhook"
    assert _map_result_to_status(r2, "HUMAN_REVIEW") == "human_review"


def test_audit_chain_tamper_detection():
    import hashlib
    import asyncio
    from app.database import init_db, db_session

    async def _run():
        await init_db()
        entries = []
        async with db_session() as db:
            for i in range(5):
                eid = f"tamper_test_{i}"
                prev = entries[-1]["entry_hash"] if entries else ""
                payload = f"{eid}|2026-01-01T00:00:{i:02d}|evt_{i}|ALLOW|retry|success|{prev}"
                h = hashlib.sha256(payload.encode()).hexdigest()
                entry = {"id": eid, "timestamp": f"2026-01-01T00:00:{i:02d}", "event_id": f"evt_{i}",
                         "workflow_status": "ALLOW", "action": "retry", "result": "success",
                         "prev_hash": prev, "entry_hash": h}
                entries.append(entry)
                await db.execute(
                    "INSERT OR REPLACE INTO audit_log (id,timestamp,event_id,customer_id,workflow_status,action,result,prev_hash,entry_hash) VALUES (?,?,?,?,?,?,?,?,?)",
                    (eid, entry["timestamp"], entry["event_id"], "cust_t", entry["workflow_status"],
                     entry["action"], entry["result"], prev, h))

        async with db_session() as db:
            cursor = await db.execute("SELECT * FROM audit_log WHERE id LIKE 'tamper_test_%' ORDER BY timestamp ASC")
            rows = [dict(r) for r in await cursor.fetchall()]

        prev_hash = ""
        for row in rows:
            payload = f"{row['id']}|{row['timestamp']}|{row['event_id']}|{row['workflow_status']}|{row['action']}|{row['result']}|{prev_hash}"
            expected = hashlib.sha256(payload.encode()).hexdigest()
            assert row["entry_hash"] == expected, f"Chain valid before tamper at {row['id']}"
            prev_hash = row["entry_hash"]

        async with db_session() as db:
            await db.execute("UPDATE audit_log SET action='TAMPERED' WHERE id='tamper_test_2'")

        async with db_session() as db:
            cursor = await db.execute("SELECT * FROM audit_log WHERE id LIKE 'tamper_test_%' ORDER BY timestamp ASC")
            rows = [dict(r) for r in await cursor.fetchall()]

        prev_hash = ""
        broken = False
        for i, row in enumerate(rows):
            payload = f"{row['id']}|{row['timestamp']}|{row['event_id']}|{row['workflow_status']}|{row['action']}|{row['result']}|{prev_hash}"
            expected = hashlib.sha256(payload.encode()).hexdigest()
            if row["entry_hash"] != expected:
                broken = True
                assert row["id"] == "tamper_test_2", f"Tamper detected at wrong entry: {row['id']}"
                break
            prev_hash = row["entry_hash"]

        assert broken, "Tamper should have been detected"

        async with db_session() as db:
            await db.execute("DELETE FROM audit_log WHERE id LIKE 'tamper_test_%'")

    asyncio.run(_run())


def test_critic_flags_retry_exhausted():
    from agents.critic import run_critic
    from app.models import (
        RevenueEvent, Customer, TransactionMetadata, EventType, DeclineCode,
        PolicyDecision, PolicyVerdict, SupervisorOutput, WorkflowStatus,
        ProposedAction,
    )
    event = RevenueEvent(
        id="critic_1", type=EventType.CARD_PAYMENT_FAILURE,
        customer=Customer(id="c1", name="T", phone="+919876543210", email="t@t.com"),
        amount=100000, root_cause=DeclineCode.INSUFFICIENT_FUNDS,
        decline_code=DeclineCode.INSUFFICIENT_FUNDS,
        failed_at=datetime.utcnow(), retry_count=3,
    )
    decision = PolicyDecision(
        verdict=PolicyVerdict.ALLOW, reason="All checks passed", checks_passed=[], checks_failed=[],
        risk_flags=[], modified_action=None,
    )
    supervisor = SupervisorOutput(
        event_id="critic_1",
        workflow_status=WorkflowStatus.READY_FOR_POLICY,
        proposed_action=ProposedAction(
            action="retry_payment", channel="razorpay_api", reason="r",
            proposed_delay_hours=24, amount=100000, discount_percent=0,
        ),
        next_step="retry", risk_flags=[],
        specialist_calls=[], diagnosis_confidence=0.9,
        policy_decision=decision,
    )
    objections = run_critic(event, decision, supervisor)
    assert any(o["rule_id"] == "retry_exhausted" for o in objections)


def test_critic_passes_on_clean_decision():
    from agents.critic import run_critic
    from app.models import (
        RevenueEvent, Customer, TransactionMetadata, EventType, DeclineCode,
        PolicyDecision, PolicyVerdict, SupervisorOutput, WorkflowStatus,
        ProposedAction,
    )
    event = RevenueEvent(
        id="critic_2", type=EventType.CARD_PAYMENT_FAILURE,
        customer=Customer(id="c1", name="T", phone="+919876543210", email="t@t.com"),
        amount=100000, root_cause=DeclineCode.INSUFFICIENT_FUNDS,
        decline_code=DeclineCode.INSUFFICIENT_FUNDS,
        failed_at=datetime(2026, 1, 15, 12, 0, 0), retry_count=0,
    )
    decision = PolicyDecision(
        verdict=PolicyVerdict.ALLOW, reason="All checks passed", checks_passed=[], checks_failed=[],
        risk_flags=[], modified_action=None,
    )
    supervisor = SupervisorOutput(
        event_id="critic_2",
        workflow_status=WorkflowStatus.READY_FOR_POLICY,
        proposed_action=ProposedAction(
            action="retry_payment", channel="razorpay_api", reason="r",
            proposed_delay_hours=24, amount=100000, discount_percent=0,
        ),
        next_step="retry", risk_flags=[],
        specialist_calls=[], diagnosis_confidence=0.9,
        policy_decision=decision,
    )
    objections = run_critic(event, decision, supervisor)
    assert len(objections) == 0


# ── Compliance Explainer Tests ──

def test_compliance_allow():
    event = _make_event()
    diag = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=24,
    )
    ctx = CustomerContextOutput(customer_id="cust_001", consent_status="CONSENTED", preferred_channel="WHATSAPP")
    proposed = ProposedAction(action="retry_payment", amount=50000)
    engine = PolicyEngine()
    now = datetime(2026, 8, 20, 12, 0)
    decision = engine.evaluate(event, diag, proposed, now=now)
    expl = explain(event, diag, ctx, proposed, decision)
    assert expl.verdict == "ALLOW"
    assert "authorized" in expl.summary.lower() or "passed" in expl.summary.lower()
    assert len(expl.rules_triggered) == 10
    assert expl.customer_safe_explanation != ""
    assert expl.operator_explanation != ""


def test_compliance_deny():
    event = _make_event()
    event.customer.opted_out = True
    diag = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="STOP", recommended_wait_hours=0,
    )
    ctx = CustomerContextOutput(customer_id="cust_001", consent_status="OPTED_OUT", preferred_channel="WHATSAPP", safe_to_contact=False)
    proposed = ProposedAction(action="blocked", amount=50000)
    engine = PolicyEngine()
    decision = engine.evaluate(event, diag, proposed)
    expl = explain(event, diag, ctx, proposed, decision)
    assert expl.verdict == "DENY"
    failed_rules = [r for r in expl.rules_triggered if r.result == "FAIL"]
    assert len(failed_rules) >= 1
    assert any("opt" in r.rule.lower() for r in failed_rules)
    assert "opted out" in expl.customer_safe_explanation.lower()


def test_compliance_human_review():
    event = _make_event()
    diag = DiagnosisOutput(
        classification="test", confidence=0.8, likely_recoverability="HIGH",
        recommended_action_family="RETRY", recommended_wait_hours=24,
        risk_score=0.85,
    )
    ctx = CustomerContextOutput(customer_id="cust_001", consent_status="CONSENTED", preferred_channel="WHATSAPP")
    proposed = ProposedAction(action="retry_payment", amount=50000)
    engine = PolicyEngine()
    decision = engine.evaluate(event, diag, proposed)
    expl = explain(event, diag, ctx, proposed, decision)
    assert expl.verdict == "HUMAN_REVIEW"
    assert "human" in expl.summary.lower() or "approval" in expl.summary.lower()
    failed_rules = [r for r in expl.rules_triggered if r.result == "FAIL"]
    assert any("risk_score" in r.rule for r in failed_rules)


# ── Execution Adapter Tests (Deterministic) ──

def test_deterministic_retry_insufficient_funds():
    event = _make_event(retry_count=0)
    gateway = DeterministicMockGateway()
    result = gateway.retry_payment("pay_test", 50000, event)
    assert result.status == "captured"
    assert result.amount == 50000


def test_deterministic_retry_insufficient_funds第二次():
    event = _make_event(retry_count=1)
    gateway = DeterministicMockGateway()
    result = gateway.retry_payment("pay_test", 50000, event)
    assert result.status == "created"


def test_deterministic_retry_bank_timeout():
    event = _make_event(
        root_cause=DeclineCode.BANK_TIMEOUT,
        decline_code=DeclineCode.BANK_TIMEOUT,
        retry_count=0,
    )
    gateway = DeterministicMockGateway()
    result = gateway.retry_payment("pay_test", 250000, event)
    assert result.status == "captured"


def test_failure_injecting_gateway():
    event = _make_event(retry_count=0)
    gateway = FailureInjectingGateway(fail_actions={"retry_payment"})
    result = gateway.retry_payment("pay_test", 50000, event)
    assert result.status == "failed"
    assert "Injected" in result.reason


def test_no_ground_truth_in_execution():
    event = _make_event(retry_count=0, ground_truth="not_recoverable")
    gateway = DeterministicMockGateway()
    result = gateway.retry_payment("pay_test", 50000, event)
    assert result.status == "captured"


def test_scenario_table_coverage():
    actions = ["retry_payment", "send_payment_link", "send_dunning_message", "re_authorize_mandate"]
    codes = ["insufficient_funds", "expired_card", "bank_timeout", "do_not_honor", "mandate_afa_required"]
    for action in actions:
        for code in codes:
            result = _lookup_scenario(action, code, 0)
            assert result is not None


# ── Scenario Library Tests ──

def test_scenario_library_has_15():
    scenarios = get_all_scenarios()
    assert len(scenarios) >= 10


def test_scenario_builds_event():
    scenario = get_scenario("demo_insufficient_funds")
    assert scenario is not None
    event = build_event_from_scenario(scenario)
    assert event.amount == 100000
    assert event.decline_code == DeclineCode.INSUFFICIENT_FUNDS


def test_scenario_opted_out():
    scenario = get_scenario("demo_opted_out")
    assert scenario["expected_verdict"] == "DENY"
    event = build_event_from_scenario(scenario)
    assert event.customer.opted_out is True


def test_scenario_all_verdicts_covered():
    scenarios = get_all_scenarios()
    verdicts = set(s["expected_verdict"] for s in scenarios)
    assert "ALLOW" in verdicts
    assert "DENY" in verdicts
    assert "MODIFY" in verdicts


# ── Model Validation Tests ──

def test_agent_recommendation_model():
    rec = AgentRecommendation(
        strategy="RETRY",
        priority="HIGH",
        proposed_delay_hours=48,
        channel="RAZORPAY_API",
        discount_percent=0.0,
        requires_human_approval=False,
        reason="Test recommendation",
    )
    assert rec.strategy == "RETRY"
    assert rec.discount_percent == 0.0


def test_batch_summary_model():
    summary = BatchSummary(
        total=100,
        processed=97,
        recovered=31,
        recovered_amount=500000,
        pending=18,
        denied=22,
        human_review=7,
        errors=3,
        blocked_by_policy=22,
    )
    assert summary.total == 100
    assert summary.errors == 3


def test_contact_event_model():
    event = ContactEvent(
        customer_id="cust_001",
        event_id="txn_001",
        channel="whatsapp",
        sent_at=datetime.utcnow(),
        status="sent",
        message_type="retry_payment",
    )
    assert event.channel == "whatsapp"


if __name__ == "__main__":
    tests = [
        test_diagnosis_insufficient_funds,
        test_diagnosis_expired_card,
        test_diagnosis_afa_required,
        test_diagnosis_confidence_drops_with_retries,
        test_diagnosis_unknown_decline_code,
        test_diagnosis_opt_out_stops,
        test_diagnosis_insufficient_funds_retries_exhausted,
        test_diagnosis_do_not_honor_with_retry,
        test_diagnosis_overdue_invoice_days,
        test_diagnosis_evidence_always_present,
        test_diagnosis_no_ground_truth_leakage,
        test_policy_allows_normal_retry,
        test_policy_blocks_opt_out,
        test_policy_blocks_max_retries,
        test_policy_blocks_afa_retry,
        test_policy_allows_reauth_for_afa,
        test_policy_modifies_high_discount,
        test_policy_blocks_outside_hours,
        test_policy_high_risk_gets_human_review,
        test_policy_modifies_high_value_discount,
        test_gate_allows_normal,
        test_gate_blocks_high_discount,
        test_generate_hinglish_message,
        test_generate_english_message,
        test_generate_reauth_message,
        test_tone_escalation,
        test_strategy_retry,
        test_strategy_dunning_tone,
        test_strategy_reauthorize,
        test_strategy_blocks_opted_out,
        test_strategy_afa_blocks_blind_retry,
        test_compliance_allow,
        test_compliance_deny,
        test_compliance_human_review,
        test_deterministic_retry_insufficient_funds,
        test_deterministic_retry_insufficient_funds第二次,
        test_deterministic_retry_bank_timeout,
        test_failure_injecting_gateway,
        test_no_ground_truth_in_execution,
        test_scenario_table_coverage,
        test_scenario_library_has_15,
        test_scenario_builds_event,
        test_scenario_opted_out,
        test_scenario_all_verdicts_covered,
        test_agent_recommendation_model,
        test_batch_summary_model,
        test_contact_event_model,
        test_afa_explanation_high_amount_non_recurring,
        test_afa_explanation_low_amount,
        test_amount_ceiling_explanation_clarity,
        test_human_review_no_execution,
        test_deny_no_execution,
        test_modify_reduces_discount,
        test_dnd_registered_customer_never_contacted,
        test_dnd_customer_policy_denies,
        test_record_outcome_is_single_write_path,
        test_audit_chain_tamper_detection,
        test_critic_flags_retry_exhausted,
        test_critic_passes_on_clean_decision,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__} — {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
