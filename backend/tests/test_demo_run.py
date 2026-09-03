"""Acceptance tests for the stage-streaming simulator + deterministic demo.

Covers the two judge-tier improvements that close the demo gap:
  * simulator events now run through the FULL real-time pipeline (seq register ->
    run_live_recovery -> every SSE stage: detect/diagnose/context/optimize/policy/
    execute/await confirmation) instead of a single transaction.updated row.
  * /api/demo/run: the deterministic 5-scenario narrative, proving trusted
    confirmation only counts money once (signed, exactly-once, duplicate ignored)
    and that bounded autonomy (AFA / opt-out / retry exhaustion) never executes.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from pathlib import Path

import pytest

# Force demo mode for deterministic derived signature checks.
os.environ["WEBHOOK_MODE"] = "demo"
os.environ["WEBHOOK_ALLOW_UNSIGNED"] = "false"


@pytest.fixture()
def isolated_db(tmp_path):
    from app.database import set_active_db_path, init_db, reset_active_db_path
    path = Path(tmp_path) / "test.db"
    token = set_active_db_path(path)
    asyncio.run(init_db())
    yield path
    reset_active_db_path(token)


def _run(coro):
    # asyncio.run builds and closes its own loop; restore a fresh main-thread
    # loop afterward so later test modules relying on asyncio.get_event_loop()
    # still see a current loop (alphabetically this file executes first).
    result = asyncio.run(coro)
    asyncio.set_event_loop(asyncio.new_event_loop())
    return result


def _capture(coro):
    from engine.ingestion import broadcaster
    q = broadcaster.subscribe()
    try:
        result = _run(coro)
        cap = list(q)
    finally:
        broadcaster.unsubscribe(q)
    return result, cap


async def _query_all(sql, *args):
    from app.database import db_session
    async with db_session() as db:
        cur = await db.execute(sql, args)
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


GO_LIVE = {
    "event_type": "payment.failed", "event_id": "evt_stage1",
    "transaction_id": "txn_stage1", "amount": 250000, "currency": "INR",
    "decline_code": "insufficient_funds", "customer_id": "cust_stage1",
    "retry_count": 0,
}


class TestSimulatorStreamsFullPipeline:
    def test_simulator_event_emits_each_stage(self, isolated_db):
        from app.main import _run_simulated_failure
        res, cap = _capture(_run_simulated_failure(dict(GO_LIVE), source="simulator"))
        types = {s["type"] for s in cap}
        for required in [
            "event.received", "event.normalized", "diagnosis.completed",
            "customer_context.completed", "strategy.candidates_generated",
            "strategy.ranked", "policy.evaluated", "execution.started",
            "payment.pending", "execution.completed", "recovery.completed",
            "event.completed",
        ]:
            assert required in types, f"missing stage {required} in {sorted(types)}"

        cand = next(s for s in cap if s["type"] == "strategy.candidates_generated")
        assert cand["payload"]["count"] > 0
        assert all({"strategy", "action", "probability", "expected_value"} <= set(c)
                   for c in cand["payload"]["candidates"])
        evs = [c["expected_value"] for c in cand["payload"]["candidates"]]
        assert evs == sorted(evs, reverse=True)

        policy = next(s for s in cap if s["type"] == "policy.evaluated")
        assert policy["payload"]["verdict"] == "ALLOW"

        # stages carry the closed-loop identifiers + the RecentTx feed row.
        assert any(s["type"] == "transaction.updated" for s in cap)
        assert res["status"] == "PENDING_WEBHOOK"
        assert res["policy_verdict"] == "ALLOW"
        assert res["amount_recovered"] == 0  # pending adds nothing


class TestDemoNarrative:
    def _demo(self):
        from app.main import demo_run
        return _run(demo_run())

    def test_demo_runs_all_five_scenarios(self, isolated_db):
        out = self._demo()
        scen = {r["scenario"]: r for r in out["scenarios"]}
        assert set(scen) == {"a", "b", "c", "d", "e"}
        assert not any("error" in r for r in out["scenarios"])

        # A: retry elected, awaited trusted confirmation, then recovered exactly once.
        a = scen["a"]
        assert a["status"] == "PENDING_WEBHOOK"
        assert a["confirm"] == "RESOLVED"
        assert a["duplicate_confirm"] == "duplicate_acknowledged"
        assert a["amount_recovered"] == 250000

        # B: RBI AFA recurring mandate -> reauthorize, never a blind retry.
        b = scen["b"]
        assert b["policy_verdict"] == "ALLOW"
        assert b["status"] == "PENDING_WEBHOOK"

        # C: opted-out customer -> DENY, zero execution.
        c = scen["c"]
        assert c["policy_verdict"] == "DENY"
        assert c["status"] == "STOPPED"
        assert c.get("amount_recovered", 0) == 0

        # D: retry budget exhausted -> no further automatic execution.
        d = scen["d"]
        assert d["policy_verdict"] in ("DENY", "HUMAN_REVIEW")
        assert d.get("amount_recovered", 0) == 0

        # E: identical inbound webhook re-delivered -> exactly-once.
        assert scen["e"]["status"] == "duplicate_acknowledged"

    def test_demo_ledger_confirmation_counts_money_once(self, isolated_db):
        self._demo()
        rows = _run(_query_all(
            "SELECT id, status, recovered_amount FROM revenue_events "
            "WHERE id LIKE 'evt_demo_%' ORDER BY id"))
        by = {r["id"]: r for r in rows}
        assert by["evt_demo_a"]["status"] == "success"
        assert by["evt_demo_a"]["recovered_amount"] == 250000
        assert by["evt_demo_b"]["recovered_amount"] == 0
        assert by["evt_demo_c"]["status"] != "success"
        assert by["evt_demo_c"]["recovered_amount"] == 0
        assert by["evt_demo_d"]["recovered_amount"] == 0
        # Re-delivered failure + duplicate confirmation never double the ledger.
        assert sum(r["recovered_amount"] for r in rows) == 250000

    def test_demo_does_not_touch_benchmark(self, isolated_db):
        from engine.pipeline import load_batch, process_batch
        from app.main import demo_run
        # Seed + replay a benchmark so isolation is measurable.
        _run(process_batch(load_batch()))
        before = _run(_query_all(
            "SELECT COUNT(*) AS c FROM revenue_events WHERE id LIKE 'txn_%'"))
        _run(demo_run())
        after = _run(_query_all(
            "SELECT COUNT(*) AS c FROM revenue_events WHERE id LIKE 'txn_%'"))
        assert before[0]["c"] == after[0]["c"]  # demo never mutates the benchmark