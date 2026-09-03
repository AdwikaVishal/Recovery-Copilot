"""Intelligent Recovery Action Scoring — acceptance tests (Feature 1).

Covers the deterministic, explainable EV-based scoring layer:
  1. insufficient_funds (transient) -> RETRY is the top candidate
  2. expired_card -> RETRY ineligible, PAYMENT_LINK elected
  3. incorrect_cvc -> PAYMENT_LINK top
  4. mandate AFA -> REAUTHORIZE is the AFA-compliant top action
  5. opted-out customer -> every contact action ineligible; nothing executes
  6. retry budget exhausted -> RETRY ineligible with a reason
  7. contact-frequency limit -> communication candidates ineligible
  8. low-confidence diagnosis -> HUMAN_REVIEW eligible / preferred above risky actions
  9. comparable probabilities -> higher Expected Recovery Value wins (exact EV math)
  10. policy DENY -> nothing executes (no money), scorer recommendation + denial in audit

Safety invariants held throughout:
  * The ranked scorer is ADVISORY. The Policy Engine stays the final authority and
    is never bypassed; a DENY means zero recovery regardless of scorer ranking.
  * On a cold DB the optimizer elects the exact rule-based primary (benchmark contract).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

os.environ["WEBHOOK_MODE"] = "demo"
os.environ["WEBHOOK_ALLOW_UNSIGNED"] = "false"


import pytest  # noqa: E402


@pytest.fixture()
def isolated_db(tmp_path):
    from app.database import set_active_db_path, init_db, reset_active_db_path
    path = Path(tmp_path) / "test.db"
    token = set_active_db_path(path)
    asyncio.run(init_db())
    yield path
    reset_active_db_path(token)


def _run(coro):
    result = asyncio.run(coro)
    asyncio.set_event_loop(asyncio.new_event_loop())
    return result


def _live_event(payload: dict):
    """Build a LiveRevenueEvent exactly as the webhook pipeline would."""
    from engine.webhook import normalize_payment_webhook
    return normalize_payment_webhook(payload)


def _payload(**overrides) -> dict:
    p = {
        "event_type": "payment.failed",
        "event_id": "evt_score1",
        "transaction_id": "txn_score1",
        "amount": 250000,
        "currency": "INR",
        "customer_id": "cust_score1",
        "customer_name": "Scorer User",
        "decline_code": "insufficient_funds",
        "retry_count": 0,
        "correlation_id": "corr_score1",
    }
    p.update(overrides)
    return p


def _diagnosis(**overrides):
    from app.models import DiagnosisOutput
    d = DiagnosisOutput(
        classification="temporary_cash_flow_issue",
        confidence=0.82,
        likely_recoverability="HIGH",
        recommended_action_family="RETRY",
        recommended_wait_hours=48,
    )
    for k, v in overrides.items():
        setattr(d, k, v)
    return d


def _context(**overrides):
    from app.models import CustomerContextOutput, ContactFrequency
    c = CustomerContextOutput(
        customer_id="cust_score1",
        consent_status="CONSENTED",
        preferred_channel="WHATSAPP",
        safe_to_contact=True,
        contact_frequency=ContactFrequency(),
    )
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def _optimize(event, diagnosis=None, context=None, now=None):
    from agents.recovery_optimizer import build_optimizer_output
    from agents.diagnosis_agent import diagnose
    from agents.customer_context_agent import get_customer_context

    async def go():
        diag = diagnosis if diagnosis is not None else diagnose(event)
        ctx = context if context is not None else await get_customer_context(event)
        return await build_optimizer_output(event, diag, ctx, now=now)

    return _run(go())


def _by_strategy(out, strategy):
    return next((c for c in out.candidates if c.strategy == strategy), None)


# ---------------------------------------------------------------------------
# 1. Transient failure -> RETRY is the top candidate.
# ---------------------------------------------------------------------------

def test_transient_failure_retry_is_top_candidate(isolated_db):
    ev = _live_event(_payload(decline_code="insufficient_funds", retry_count=0))
    out = _optimize(ev)

    retry = _by_strategy(out, "RETRY")
    assert retry is not None
    assert retry.eligible is True
    assert retry.expected_value == int(ev.amount * 0.62)  # base prior for insufficient_funds

    # Ranked BY expected recovery value: RETRY must be first.
    assert out.candidates[0].strategy == "RETRY"
    assert out.elected.strategy == "RETRY"
    # Explicit EV ordering contract.
    evs = [c.expected_value for c in out.candidates]
    assert evs == sorted(evs, reverse=True)


# ---------------------------------------------------------------------------
# 2. expired_card -> RETRY ineligible; PAYMENT_LINK elected (EV + eligibility).
# ---------------------------------------------------------------------------

def test_expired_card_retry_ineligible_payment_link_wins(isolated_db):
    ev = _live_event(_payload(decline_code="expired_card", retry_count=1))
    out = _optimize(ev)

    retry = _by_strategy(out, "RETRY")
    link = _by_strategy(out, "PAYMENT_LINK")

    assert retry is not None and retry.eligible is False
    assert "Payment method invalid" in retry.ineligibility_reason
    assert "PAYMENT_METHOD_INVALID" in retry.reason_codes
    assert link is not None and link.eligible is True

    assert out.elected is not None and out.elected.strategy == "PAYMENT_LINK"
    assert out.elected.action == "send_payment_link"


# ---------------------------------------------------------------------------
# 3. incorrect_cvc -> PAYMENT_LINK top.
# ---------------------------------------------------------------------------

def test_incorrect_cvc_payment_link_top(isolated_db):
    ev = _live_event(_payload(decline_code="incorrect_cvc", retry_count=1))
    out = _optimize(ev)

    link = _by_strategy(out, "PAYMENT_LINK")
    assert link is not None and link.eligible is True
    assert link.expected_value == int(ev.amount * 0.42)  # base prior (incorrect_cvc)
    assert out.elected.strategy == "PAYMENT_LINK"
    assert out.candidates[0].strategy == "PAYMENT_LINK"


# ---------------------------------------------------------------------------
# 4. mandate AFA -> REAUTHORIZE is the compliant top action.
# ---------------------------------------------------------------------------

def test_mandate_afa_reauthorize_is_top(isolated_db):
    ev = _live_event(_payload(
        event_type="recurring_payment_failure",
        decline_code="mandate_afa_required",
        amount=2500000, retry_count=0,
    ))
    out = _optimize(ev)

    reauth = _by_strategy(out, "REAUTHORIZE")
    retry = _by_strategy(out, "RETRY")
    assert reauth is not None and reauth.eligible is True
    assert reauth.channel.upper() in ("WHATSAPP",)
    assert "AFA_COMPLIANT_ACTION" in reauth.reason_codes

    # A blind retry is ineligible under AFA (RBI trickle).
    assert retry is not None and retry.eligible is False
    assert "AFA required" in retry.ineligibility_reason
    assert "AFA_REQUIRED" in retry.reason_codes

    assert out.elected.strategy == "REAUTHORIZE"
    assert out.candidates[0].strategy == "REAUTHORIZE"


# ---------------------------------------------------------------------------
# 5. opted-out customer -> every contact action ineligible; nothing executes.
# ---------------------------------------------------------------------------

def test_opted_out_blocks_all_contact_actions(isolated_db):
    from app.models import CustomerContextOutput
    ctx = CustomerContextOutput(
        customer_id="cust_score1", consent_status="OPTED_OUT",
        preferred_channel="WHATSAPP", safe_to_contact=False,
    )
    ev = _live_event(_payload(decline_code="bank_timeout", retry_count=0))
    diag = _diagnosis(classification="transient_failure", confidence=0.88,
                      likely_recoverability="HIGH", recommended_action_family="RETRY")
    out = _optimize(ev, diagnosis=diag, context=ctx)

    for c in out.candidates:
        if c.strategy in ("PAYMENT_LINK", "MESSAGE", "REAUTHORIZE"):
            assert c.eligible is False, f"{c.strategy} must be ineligible for opted-out"
            assert "OPT_OUT_OR_SAFETY" in c.reason_codes
    # The contact-free retry (direct charge, no comms) is the only eligible path,
    # but the supervisor's deterministic primary (STOP on opt-out) means the
    # scorer elects NO candidate — nothing auto-executes, policy decides.
    eligible_only = [c for c in out.candidates if c.eligible]
    assert [c.strategy for c in eligible_only] == ["RETRY"]
    assert out.elected is None
    assert "safe_to_contact=False" in out.decision_factors

    # End-to-end: the supervisor + policy STOP this customer; zero execution.
    from agents.supervisor import process_event
    from app.models import WorkflowStatus
    ev2 = _live_event(_payload(decline_code="bank_timeout", retry_count=0, event_id="evt_opt"))
    ev2.customer.opted_out = True
    result = _run(process_event(ev2))
    assert result.workflow_status == WorkflowStatus.STOPPED

    async def count_rows():
        from app.database import db_session
        async with db_session() as db:
            cur = await db.execute("SELECT COUNT(*) c FROM recovery_attempts")
            attempts = (await cur.fetchone())["c"]
            cur = await db.execute("SELECT COUNT(*) c FROM strategy_outcomes")
            outcomes = (await cur.fetchone())["c"]
        return attempts, outcomes

    assert _run(count_rows()) == (0, 0)


# ---------------------------------------------------------------------------
# 6. retry budget exhausted -> RETRY ineligible with a reason + code.
# ---------------------------------------------------------------------------

def test_retry_budget_exhausted_ineligible(isolated_db):
    ev = _live_event(_payload(decline_code="bank_timeout", retry_count=3))
    diag = _diagnosis(classification="transient_failure", confidence=0.88,
                      likely_recoverability="HIGH", recommended_action_family="HUMAN_REVIEW",
                      max_retries=3)
    out = _optimize(ev, diagnosis=diag)

    retry = _by_strategy(out, "RETRY")
    assert retry is not None and retry.eligible is False
    assert "budget exhausted" in retry.ineligibility_reason
    assert "RETRY_LIMIT_REACHED" in retry.reason_codes
    assert out.elected is not None and out.elected.strategy != "RETRY"


# ---------------------------------------------------------------------------
# 7. contact-frequency limit -> communication candidates ineligible.
# ---------------------------------------------------------------------------

def test_contact_frequency_limit_blocks_communications(isolated_db):
    from app.models import CustomerContextOutput, ContactFrequency
    ctx = CustomerContextOutput(
        customer_id="cust_score1", consent_status="CONSENTED",
        preferred_channel="WHATSAPP", safe_to_contact=True,
        contact_frequency=ContactFrequency(contacts_last_7d=5),
    )
    ev = _live_event(_payload(decline_code="payment_link_expired", retry_count=0))
    out = _optimize(ev, context=ctx)

    for c in out.candidates:
        if c.strategy in ("PAYMENT_LINK", "MESSAGE"):
            assert c.eligible is False, f"{c.strategy} must be ineligible at freq limit"
            assert "CONTACT_FREQUENCY_LIMIT" in c.reason_codes


# ---------------------------------------------------------------------------
# 8. low-confidence / high-risk diagnosis -> HUMAN_REVIEW proposed as safe path.
# ---------------------------------------------------------------------------

def test_low_confidence_raises_human_review(isolated_db):
    ev = _live_event(_payload(decline_code="do_not_honor", retry_count=1))
    # Inconclusive diagnosis: low confidence, exhausted context, no specialist rule.
    diag = _diagnosis(classification="ambiguous_decline", confidence=0.15,
                      likely_recoverability="UNKNOWN", recommended_action_family="HUMAN_REVIEW",
                      risk_score=0.6, max_retries=1)
    out = _optimize(ev, diagnosis=diag)

    human = _by_strategy(out, "HUMAN_REVIEW")
    assert human is not None
    assert human.eligible is True
    assert human.requires_human_approval is True
    assert "LOW_CONFIDENCE" in human.reason_codes

    # HARD_GUARD: on an uncertain diagnosis the risky blind retry is ineligible
    # (budget not usable) and HUMAN_REVIEW is the elected primary, ranking above
    # every auto-action in the candidates.
    retry = _by_strategy(out, "RETRY")
    assert retry is not None and retry.eligible is False
    assert out.candidates[0].strategy == "HUMAN_REVIEW"
    assert out.elected.strategy == "HUMAN_REVIEW"
    human_rank = out.candidates.index(human)
    auto_ranks = [out.candidates.index(c) for c in out.candidates
                  if c.strategy in ("RETRY", "PAYMENT_LINK", "MESSAGE")]
    assert human_rank < min(auto_ranks), "HUMAN_REVIEW must outrank risky auto-actions on uncertainty"


# ---------------------------------------------------------------------------
# 9. comparable probabilities -> higher Expected Recovery Value wins (exact math).
# ---------------------------------------------------------------------------

def test_higher_expected_value_wins(isolated_db):
    ev = _live_event(_payload(decline_code="do_not_honor", amount=500000, retry_count=0))
    out = _optimize(ev)

    retry = _by_strategy(out, "RETRY")
    link = _by_strategy(out, "PAYMENT_LINK")
    assert retry is not None and link is not None

    # RETRY prior for do_not_honor (0.38) vs PAYMENT_LINK (default 0.25): same
    # amount, so expected value is strictly larger for the higher probability.
    assert retry.expected_value == int(ev.amount * 0.38)
    assert link.expected_value == int(ev.amount * 0.25)
    assert retry.expected_value > link.expected_value
    assert out.candidates[0].strategy == "RETRY"
    # Ranking is exactly by Probability x Amount; ties break deterministically.
    evs = [c.expected_value for c in out.candidates]
    assert evs == sorted(evs, reverse=True)


# ---------------------------------------------------------------------------
# 10. policy DENY -> zero recovery; audit shows scorer recommendation + denial.
# ---------------------------------------------------------------------------

def test_policy_deny_blocks_execution_entirely(isolated_db):
    from agents.supervisor import process_event
    from app.database import db_session
    from app.models import PolicyVerdict, WorkflowStatus

    now = datetime.utcnow()
    ev = _live_event(_payload(decline_code="bank_timeout", retry_count=0))
    # Forge a recent prior attempt: cooling period can never be satisfied.
    ev.last_attempt_at = (now - timedelta(hours=1)).isoformat()

    out = _run(process_event(ev, now=now))

    assert out.workflow_status == WorkflowStatus.STOPPED
    assert out.policy_decision is not None
    assert out.policy_decision.verdict == PolicyVerdict.DENY

    # Nothing may execute -> no money recorded, no outcome row, no attempt.
    async def no_money():
        async with db_session() as db:
            cur = await db.execute("SELECT COUNT(*) c FROM recovery_attempts")
            attempts = (await cur.fetchone())["c"]
            cur = await db.execute("SELECT COUNT(*) c FROM strategy_outcomes")
            outcomes = (await cur.fetchone())["c"]
            return attempts, outcomes

    attempts, outcomes = _run(no_money())
    assert (attempts, outcomes) == (0, 0)

    # Audit trail shows the scorer's recommendation AND the policy denial.
    async def audit_rows():
        async with db_session() as db:
            cur = await db.execute(
                "SELECT specialist_calls, proposed_action, policy_decision, workflow_status "
                "FROM audit_log WHERE event_id = ? ORDER BY timestamp",
                (ev.id,),
            )
            rows = [dict(r) for r in await cur.fetchall()]
        return rows

    rows = _run(audit_rows())
    assert rows, "DENY must still be audited (no silent drop)"
    all_calls = []
    for r in rows:
        call_agents = [c.get("agent") for c in __import__("json").loads(r["specialist_calls"] or "[]")]
        all_calls.extend(call_agents)
    assert "recovery_scoring_agent" in all_calls, "scorer recommendation must be in the trace"
    assert "policy_engine" in all_calls, "policy denial must be in the trace"
    verdicts = [__import__("json").loads(r["policy_decision"] or "{}").get("verdict") for r in rows]
    assert any(v == "DENY" for v in verdicts)


# ---------------------------------------------------------------------------
# Bonus: cold-DB election parity — optimizer output == rule-based primary.
# ---------------------------------------------------------------------------

def test_cold_db_election_parity_with_rule_based_primary(isolated_db):
    from agents.recovery_strategy_agent import build_strategy

    ev = _live_event(_payload(decline_code="insufficient_funds", retry_count=0))
    out = _optimize(ev)

    ctx = _context()
    diag = _diagnosis()
    primary = build_strategy(ev, diag, ctx)
    assert out.strategy.strategy == primary.strategy
    assert out.strategy.expected_value == primary.expected_value