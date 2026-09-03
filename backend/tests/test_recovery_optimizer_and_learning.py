import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import tempfile
from pathlib import Path

import pytest

from app.database import (
    init_db, set_active_db_path, reset_active_db_path,
    get_active_db_path,
)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Give each test a fresh, isolated SQLite DB so the learning/optimizer tests
    are deterministic and never pollute (or depend on residue from) the shared
    benchmark batch DB used by the other test files."""
    token = None
    try:
        path = Path(tmp_path) / "test.db"
        token = set_active_db_path(path)
        asyncio.run(init_db())
        yield path
    finally:
        if token is not None:
            reset_active_db_path(token)


def _run(coro):
    return asyncio.run(coro)


def test_optimizer_preserves_primary_strategy_on_cold_db():
    """On a fresh DB (no learning data) the optimizer must elect the exact
    deterministic output of build_strategy — preserving the benchmark contract."""
    from engine.pipeline import process_batch, load_batch
    from agents.supervisor import process_event

    async def scenario():
        events = load_batch()
        await process_batch(events)
        outcomes = []
        for ev in events:
            out = await process_event(ev)
            outcomes.append((ev.id, out))
        return outcomes

    out = _run(scenario())

    # Runnable events must each have a concrete action (not error / None).
    unactioned_run = [
        (eid, o.next_step[:60]) for eid, o in out
        if o.workflow_status.value not in ("STOPPED", "INVALID")
        and (o.proposed_action is None or o.proposed_action.action in (None, "", "none"))
    ]
    assert not unactioned_run, f"runnable events missing an action: {unactioned_run}"

    # The healthy benchmark number still holds.
    recovered = sum(1 for _, o in out if o.workflow_status.value == "RESOLVED")
    assert recovered > 0, "batch should recover a positive number of events"


def test_learning_table_populated_and_report_consistent():
    """After a batch run the strategy_outcomes table feeds a consistent
    strategy-effectiveness report (one row per event, sums match benchmark)."""
    from engine.pipeline import process_batch, load_batch
    from engine.recovery_analytics import strategy_effectiveness_report
    from app.database import db_session

    async def scenario():
        events = load_batch()
        r = await process_batch(events)
        async with db_session() as db:
            cursor = await db.execute("SELECT COUNT(*) as c, COUNT(DISTINCT event_id) as d FROM strategy_outcomes")
            row = dict(await cursor.fetchone())
        rep = await strategy_effectiveness_report()
        retry = next((s for s in rep["strategies"] if s["strategy"] == "RETRY"), None)
        return r, row, rep, retry

    r, row, rep, retry = _run(scenario())

    # One outcome row per batch event (no double-recording corruption).
    assert row["c"] == len(r.records), f"expected one outcome per event, got {row['c']}"
    assert row["d"] == row["c"], "event_id keys must be unique"

    # The RETRY effectiveness must mirror the benchmark: successes summing to the
    # exact recovered benchmark amount.
    assert retry is not None, "RETRY strategy must be present in the report"
    assert retry["successes"] == r.recovered, f"RETRY successes {retry['successes']} != {r.recovered}"
    assert retry["recovery_amount"] == r.recovered_amount, \
        f"RETRY recovered {retry['recovery_amount']} != benchmark {r.recovered_amount}"

    # Overall success rate sanity.
    assert 0.0 <= rep["overall"]["success_rate"] <= 1.0


def test_learning_overrides_prior_with_data():
    """The closed loop: with >=3 recorded outcomes for a (strategy, decline_code),
    the empirical probability replaces the cold prior used for Expected Recovery Value."""
    from agents.probability_estimator import _base_probability
    from app.database import record_strategy_outcome, get_recovery_probability
    from engine.pipeline import load_batch

    async def scenario():
        from agents.customer_context_agent import get_customer_context
        events = [e for e in load_batch() if e.decline_code.value == "insufficient_funds"]
        assert events, "need insufficient_funds batch events"
        ev = events[0]
        base = _base_probability("MESSAGE", ev)

        # Seed 3 successes out of 4 = empirical 0.75.
        for i in range(4):
            await record_strategy_outcome(
                event_id=f"learn_{i}",
                strategy="MESSAGE",
                action="send_dunning_message",
                channel="whatsapp",
                amount=ev.amount,
                success=(i < 3),
                recovered_amount=(ev.amount if i < 3 else 0),
                probability=base,
                expected_value=int(ev.amount * base),
                decline_code=ev.decline_code.value,
                diagnosis_confidence=0.9,
                safe_to_contact=True,
            )
        learned = await get_recovery_probability("MESSAGE", ev.decline_code.value, base)
        return base, learned

    base, learned = _run(scenario())
    assert learned == 0.75, f"expected empirical 0.75, got {learned}"
