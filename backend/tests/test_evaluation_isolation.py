import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import pytest

from app.database import init_db, db_session


async def _snapshot():
    """Capture every benchmark-relevant row set as an immutable snapshot."""
    async with db_session() as db:
        rows = {}
        for table, cols in (
            ("revenue_events", ["id", "status", "recovered_amount", "retry_count", "last_attempt_at"]),
            ("audit_log", ["id", "event_id", "action", "policy_decision", "result", "workflow_status"]),
        ):
            cursor = await db.execute(f"SELECT {', '.join(cols)} FROM {table} ORDER BY id")
            rows[table] = [tuple(r[c] for c in cols) for r in await cursor.fetchall()]

        cursor = await db.execute("SELECT customer_id, event_id, channel, message_type FROM contact_events ORDER BY customer_id, event_id")
        rows["contact_events"] = [tuple(r) for r in await cursor.fetchall()]

        # Only benchmark events are counted (txn_*). Live/webhook test rows must never
        # leak into or out of the benchmark tables.
        cursor = await db.execute(
            "SELECT id, status, recovered_amount, retry_count, last_attempt_at FROM revenue_events "
            "WHERE id LIKE 'txn_%' ORDER BY id"
        )
        batch = [tuple(r) for r in await cursor.fetchall()]
    return rows, batch


@pytest.fixture(autouse=True)
def setup_db():
    asyncio.get_event_loop().run_until_complete(init_db())


def test_evaluation_does_not_mutate_batch_benchmark_state():
    from engine.pipeline import process_batch, load_batch
    from engine.evaluation import run_scenario_evaluation

    async def scenario():
        events = load_batch()
        result = await process_batch(events)

        before_rows, before_batch = await _snapshot()

        # Run the full evaluation suite exactly as the UI's "Run Evaluation" does.
        results = await run_scenario_evaluation()
        passed = sum(1 for r in results if r.get("pass"))
        total = len(results)

        after_rows, after_batch = await _snapshot()

        return {
            "batch": result,
            "before_rows": before_rows,
            "before_batch": before_batch,
            "after_rows": after_rows,
            "after_batch": after_batch,
            "passed": passed,
            "total": total,
        }

    out = asyncio.get_event_loop().run_until_complete(scenario())

    # Scenario evaluation must remain fully passing (20/20).
    assert out["passed"] == out["total"] >= 20, f"scenario evaluation regressed: {out['passed']}/{out['total']}"
    assert out["total"] == 20

    # Every benchmark table must be byte-for-byte identical after evaluation.
    assert out["after_rows"] == out["before_rows"], "evaluation mutated shared benchmark state"

    # The batch subset (txn_* events) must be identical.
    assert out["after_batch"] == out["before_batch"], "evaluation mutated batch events"

    # No scenario residue in contact_events.
    remaining = {t for t in out["after_rows"]["contact_events"] if t[0].startswith("cust_scenario")}
    assert not remaining, f"evaluation left scenario contact residue: {remaining}"

    # No scenario rows in the benchmark tables.
    scenario_rows = [r for r in out["after_rows"]["revenue_events"] if r[0].startswith("scenario_")]
    scenario_audit = [r for r in out["after_rows"]["audit_log"] if "scenario" in r[0]]
    assert not scenario_rows, f"evaluation inserted scenario revenue rows"
    assert not scenario_audit, f"evaluation inserted scenario audit rows"
