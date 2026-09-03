import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import aiosqlite
import pytest

from app.database import init_db, db_session, get_active_db_path, set_active_db_path


async def _snapshot_all():
    """Capture every DB row set as a stable sorted tuple snapshot."""
    async with db_session() as db:
        rows = {}
        for table, cols, order in [
            ("revenue_events", ["id", "status", "recovered_amount", "retry_count", "last_attempt_at"], "id"),
            ("audit_log", ["id", "event_id", "action", "result", "workflow_status"], "id"),
            ("batch_runs", ["batch_id", "total_records", "recovered", "recovered_amount",
                            "baseline_amount", "blocked_by_policy", "human_review",
                            "pending_webhook", "errors"], "batch_id"),
            ("contact_events", ["customer_id", "event_id", "channel", "status"], "customer_id"),
        ]:
            cursor = await db.execute(f"SELECT {', '.join(cols)} FROM {table} ORDER BY {order}")
            rows[table] = [tuple(r[c] for c in cols) for r in await cursor.fetchall()]
    return rows


async def _snapshot_batch_bars():
    """Mirrors the dashboard hero endpoint's batch-only metric computation."""
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) as c FROM revenue_events WHERE id LIKE 'txn_%' AND status = 'success'")
        recovered = (await cursor.fetchone())["c"]

        cursor = await db.execute(
            "SELECT SUM(recovered_amount) as a FROM revenue_events WHERE id LIKE 'txn_%' AND status = 'success'")
        recovered_amount = (await cursor.fetchone())["a"] or 0

        cursor = await db.execute(
            "SELECT COUNT(*) as c FROM revenue_events WHERE id LIKE 'txn_%' AND status = 'blocked'")
        blocked = (await cursor.fetchone())["c"]

        cursor = await db.execute(
            "SELECT COUNT(*) as c FROM revenue_events WHERE id LIKE 'txn_%' AND status = 'human_review'")
        human_review = (await cursor.fetchone())["c"]

        cursor = await db.execute(
            "SELECT COUNT(*) as c FROM revenue_events WHERE id LIKE 'txn_%' AND status = 'pending_webhook'")
        pending_webhook = (await cursor.fetchone())["c"]

        # Batch recoverable amount baseline per counterfactual
        cursor = await db.execute(
            "SELECT decline_code, SUM(amount) as total FROM revenue_events WHERE id LIKE 'txn_%' GROUP BY decline_code")
        cat_rows = [dict(r) for r in await cursor.fetchall()]
    from data.scenarios import COUNTERFACTUAL_RATES
    baseline = 0
    for cr in cat_rows:
        rate = COUNTERFACTUAL_RATES.get(cr["decline_code"], 0.05)
        baseline += int((cr["total"] or 0) * rate)

    return {
        "batch_recovered": recovered,
        "batch_recovered_amount": recovered_amount,
        "batch_blocked": blocked,
        "batch_human_review": human_review,
        "batch_pending_webhook": pending_webhook,
        "batch_baseline": baseline,
        "batch_incremental": recovered_amount - baseline,
    }


@pytest.fixture(autouse=True)
def setup_db():
    asyncio.get_event_loop().run_until_complete(init_db())


def test_evaluation_does_not_mutate_batch():
    from engine.pipeline import process_batch, load_batch
    from engine.evaluation import run_scenario_evaluation

    async def scenario():
        events = load_batch()
        batch1 = await process_batch(events)

        before_rows = await _snapshot_all()
        before_metrics = await _snapshot_batch_bars()

        results = await run_scenario_evaluation()
        passed = sum(1 for r in results if r.get("pass"))
        total = len(results)

        after_rows = await _snapshot_all()
        after_metrics = await _snapshot_batch_bars()

        # Run batch AGAIN after evaluation (must be deterministic / unaffected)
        batch2 = await process_batch(events)
        post_rerun_metrics = await _snapshot_batch_bars()

        return {
            "batch1": batch1,
            "batch2": batch2,
            "before_rows": before_rows,
            "after_rows": after_rows,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "post_rerun_metrics": post_rerun_metrics,
            "passed": passed,
            "total": total,
        }

    out = asyncio.get_event_loop().run_until_complete(scenario())

    # Scenario evaluation is 20/20.
    assert out["total"] == 20, f"expected 20 scenarios, got {out['total']}"
    assert out["passed"] == 20, f"evaluation regressed: {out['passed']}/20"

    # DB fingerprint byte-for-byte identical after evaluation.
    assert out["after_rows"] == out["before_rows"], "evaluation mutated DB state"

    # Dashboard batch metrics identical after evaluation.
    assert out["after_metrics"] == out["before_metrics"], "evaluation changed batch metrics"

    # A second batch run is deterministic and not polluted by evaluation.
    assert out["post_rerun_metrics"] == out["before_metrics"], "evaluation altered a subsequent batch run"

    # Recovered amount must remain positive (the healthy benchmark number survives).
    assert out["before_metrics"]["batch_recovered_amount"] > 0, "batch recovered amount should be non-zero"
    assert out["batch1"].recovered_amount == out["batch2"].recovered_amount, "batch replay must be deterministic"


def test_batch_is_idempotent_across_repeated_runs():
    """The benchmark batch must be a deterministic replay.

    Root cause of the historical corruption: record_outcome() inserts a new
    contact_events row per contacted event with no de-duplication, so re-running
    the batch compounds per-customer contact counts until the frequency-limit
    guardrail flips the whole benchmark to DENY (₹26k -> ₹0, 92 blocked).
    Running the batch repeatedly must not change its outcome.
    """
    from engine.pipeline import process_batch, load_batch

    async def scenario():
        events = load_batch()
        results = []
        for _ in range(4):
            r = await process_batch(events)
            results.append((r.recovered, r.recovered_amount, r.blocked_by_policy))
        return results

    out = asyncio.get_event_loop().run_until_complete(scenario())

    recovered, amount, blocked = out[0]
    assert recovered > 0, "healthy batch must recover a positive number of events"
    for r in out[1:]:
        assert r == (recovered, amount, blocked), (
            f"batch is not idempotent: {out} "
            "(contact_events are compounding and flipping the benchmark to DENY)"
        )
