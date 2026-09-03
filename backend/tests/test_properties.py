import asyncio
import sys
import os
from datetime import datetime
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pytest
import hypothesis
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.models import (
    RevenueEvent, Customer, TransactionMetadata,
    EventType, DeclineCode,
)


def evaluate_policy(event):
    """Run the REAL decision path: diagnosis -> context -> strategy ->
    proposed action -> PolicyEngine (the same contract the supervisor uses).
    Returns (PolicyDecision, ProposedAction)."""
    from agents.diagnosis_agent import diagnose
    from agents.customer_context_agent import get_customer_context
    from agents.recovery_strategy_agent import build_strategy, strategy_to_proposed_action
    from agents.policy_engine import PolicyEngine

    async def _run():
        diagnosis = diagnose(event)
        context = await get_customer_context(event)
        strategy = build_strategy(event, diagnosis, context)
        proposed = strategy_to_proposed_action(event, strategy)
        decision = PolicyEngine().evaluate(
            event, diagnosis, proposed, now=datetime(2026, 1, 15, 12, 0, 0)
        )
        return decision, proposed

    return asyncio.run(_run())


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Property tests hit customer context (DB queries) — give each module run
    a clean temp SQLite database so no state leaks between cases."""
    from pathlib import Path
    from app.database import set_active_db_path, init_db, reset_active_db_path
    token = set_active_db_path(Path(tmp_path) / "property.db")
    asyncio.run(init_db())
    yield
    reset_active_db_path(token)


def _event(id, decline_code, retry_count, opted_out=False, **kw):
    return RevenueEvent(
        id=id,
        type=kw.pop("event_type", EventType.CARD_PAYMENT_FAILURE),
        customer=Customer(
            id="c1", name="T", phone="+919876543210", email="t@t.com",
            opted_out=opted_out,
        ),
        amount=kw.pop("amount", 50000),
        root_cause=decline_code,
        decline_code=decline_code,
        failed_at=datetime(2026, 1, 15, 12, 0, 0),
        retry_count=retry_count,
        metadata=kw.pop("metadata", TransactionMetadata()),
    )


@settings(max_examples=400, deadline=None)
@given(
    amount=st.integers(min_value=1500000, max_value=100000000),
    retry_count=st.integers(min_value=0, max_value=10),
)
def test_high_value_mandate_never_blind_retry(amount, retry_count):
    event = _event(
        "prop_1", DeclineCode.MANDATE_AFA_REQUIRED, retry_count,
        event_type=EventType.RECURRING_PAYMENT_FAILURE, amount=amount,
    )
    decision, proposed = evaluate_policy(event)
    if decision.verdict.value in ("ALLOW", "MODIFY"):
        assert proposed.action != "retry_payment", (
            f"High-value mandate ₹{amount // 100} must not get blind retry, "
            f"got {proposed.action}"
        )


@settings(max_examples=200, deadline=None)
@given(
    amount=st.integers(min_value=10000, max_value=50000000),
    decline_code=st.sampled_from(list(DeclineCode)),
    retry_count=st.integers(min_value=0, max_value=10),
)
def test_opted_out_always_deny(amount, decline_code, retry_count):
    event = _event("prop_2", decline_code, retry_count, opted_out=True, amount=amount)
    decision, proposed = evaluate_policy(event)
    assert decision.verdict.value == "DENY", (
        f"Opted-out customer must get DENY, got {decision.verdict}"
    )
    assert proposed.action == "blocked", (
        f"Opted-out customer action must be blocked, got {proposed.action}"
    )


@settings(max_examples=400, deadline=None)
@given(
    amount=st.integers(min_value=10000, max_value=50000000),
    decline_code=st.sampled_from(list(DeclineCode)),
    retry_count=st.integers(min_value=0, max_value=10),
)
def test_retry_count_never_exceeds_max(amount, decline_code, retry_count):
    assume(retry_count <= 10)
    event = _event("prop_3", decline_code, retry_count, amount=amount)
    decision, proposed = evaluate_policy(event)
    if proposed.action == "retry_payment" and decision.verdict.value == "ALLOW":
        new_count = retry_count + 1
        assert new_count <= 3, (
            f"Retry count {new_count} exceeds max 3 (original retry_count={retry_count})"
        )


@settings(max_examples=400, deadline=None)
@given(
    amount=st.integers(min_value=500001, max_value=50000000),
    decline_code=st.sampled_from(list(DeclineCode)),
    retry_count=st.integers(min_value=0, max_value=3),
    discount_hint=st.integers(min_value=1, max_value=25),
)
def test_amount_ceiling_at_50k(amount, decline_code, retry_count, discount_hint):
    event = _event(
        "prop_4", decline_code, retry_count, amount=amount,
        metadata=TransactionMetadata(discount_hint=discount_hint),
    )
    decision, proposed = evaluate_policy(event)
    if proposed.discount_percent > 0:
        assert decision.verdict.value != "ALLOW", (
            f"High-value ₹{amount // 100} with {proposed.discount_percent}% discount "
            f"must not ALLOW without human review, got {decision.verdict}"
        )


@settings(max_examples=400, deadline=None)
@given(
    amount=st.integers(min_value=10000, max_value=50000000),
)
def test_transient_decline_allows_retry(amount):
    event = _event("prop_5", DeclineCode.BANK_TIMEOUT, 0, amount=amount)
    decision, _ = evaluate_policy(event)
    assert decision.verdict.value == "ALLOW", (
        f"Transient gateway timeout should ALLOW retry, got {decision.verdict}"
    )


@settings(max_examples=400, deadline=None)
@given(
    amount=st.integers(min_value=10000, max_value=50000000),
    decline_code=st.sampled_from(list(DeclineCode)),
    retry_count=st.integers(min_value=4, max_value=20),
)
def test_exhausted_retries_deny_or_human_review(amount, decline_code, retry_count):
    assume(retry_count >= 4)
    event = _event("prop_6", decline_code, retry_count, amount=amount)
    decision, _ = evaluate_policy(event)
    assert decision.verdict.value in ("DENY", "HUMAN_REVIEW"), (
        f"Exhausted retries (count={retry_count}) must not ALLOW, got {decision.verdict}"
    )


def run_property_tests():
    import sys
    tests = [
        test_high_value_mandate_never_blind_retry,
        test_opted_out_always_deny,
        test_retry_count_never_exceeds_max,
        test_amount_ceiling_at_50k,
        test_transient_decline_allows_retry,
        test_exhausted_retries_deny_or_human_review,
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

    print(f"\n{passed}/{passed + failed} property tests passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_property_tests()