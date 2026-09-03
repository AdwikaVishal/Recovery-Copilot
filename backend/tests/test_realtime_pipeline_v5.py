"""v5.0 acceptance tests: real-time closed-loop webhook pipeline.

Covers (Part 22 acceptance) through the REAL decision path:
  * webhook signature + payload validation
  * normalization (9 canonical fields)
  * atomic exactly-once idempotency (no double audit / attempts)
  * SSE stage emission (candidates -> EV ranking -> policy -> execution)
  * policy boundary (DENY/ALLOW, no blind AFA retry)
  * confirmation-gated revenue (pending adds nothing; only trusted captures count)
  * closed-loop re-optimization (failed -> failed -> confirmed)
  * MAX_RECOVERY_STEPS bound (4th failure blocked, no further re-optimization)
  * live metrics
  * benchmark + evaluation isolation (numbers provably unchanged)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Force demo mode for deterministic derived signature tests.
os.environ["WEBHOOK_MODE"] = "demo"
os.environ["WEBHOOK_ALLOW_UNSIGNED"] = "false"

EXPECTED_BATCH_REVOVERED_COUNT = 27
EXPECTED_BATCH_REVOVERED_AMOUNT = 9406014  # ₹94,060.14


@pytest.fixture()
def isolated_db(tmp_path):
    from app.database import set_active_db_path, init_db, reset_active_db_path
    path = Path(tmp_path) / "test.db"
    token = set_active_db_path(path)
    asyncio.run(init_db())
    yield path
    reset_active_db_path(token)


def _make_payload(**overrides) -> dict:
    # Default: RBI AFA mandate event that yields PENDING_WEBHOOK (awaiting
    # trusted payment confirmation) — the honest closed-loop unit.
    p = {
        "event_type": "recurring_payment_failure",
        "event_id": "evt_web1",
        "transaction_id": "txn_web1",
        "amount": 2500000,
        "currency": "INR",
        "customer_id": "cust_web1",
        "customer_name": "Webhook User",
        "decline_code": "mandate_afa_required",
        "retry_count": 0,
        "correlation_id": "corr_web1",
        "source": "razorpay",
    }
    p.update(overrides)
    return p


def _sign(body: bytes) -> str:
    from engine.webhook import compute_signature, _derive_secret
    return compute_signature(body, _derive_secret())


def _process(payload: dict, capture: list = None) -> dict:
    """Mirror POST /api/webhooks/payment exactly (idempotency gate,
    confirmation routing, sequence registration, outcome mirroring).

    Uses the same helper calls as app.main.payment_webhook so the acceptance
    contract is exercised against a real (isolated) database.
    """
    from engine.webhook import normalize_payment_webhook, new_correlation_id
    from engine.realtime import (
        run_live_recovery, confirm_live_recovery, publish_recovery_blocked,
    )
    from engine.ingestion import broadcaster
    from app.database import (
        mark_webhook_processed, record_webhook_received, get_webhook_processing,
        register_recovery_step, update_recovery_sequence, get_open_recovery_sequence,
        record_recovery_attempt, db_session,
    )

    MAX_STEPS = 3

    async def go():
        event = normalize_payment_webhook(payload)
        correlation_id = new_correlation_id()
        event.correlation_id = correlation_id

        idem_key = (
            f"{payload.get('event_type', 'event')}:"
            f"{payload.get('event_id') or payload.get('transaction_id') or event.id}"
        )
        is_new = await record_webhook_received(
            event_id=idem_key, correlation_id=correlation_id,
            source="live_webhook", signature_verified=True,
        )
        if not is_new:
            existing = await get_webhook_processing(idem_key)
            return {
                "status": "duplicate_acknowledged",
                "message": (existing.get("result_summary") if existing
                            else "Event already processed (idempotent)"),
                "event_id": event.id, "transaction_id": payload.get("transaction_id"),
            }

        recovery_key = payload.get("transaction_id") or event.transaction_id or event.id
        is_confirmation = (
            payload.get("event_type") in ("payment.captured", "subscription.charged")
            or payload.get("status") in ("captured", "authorized", "success", "paid")
        )

        if is_confirmation:
            seq = await get_open_recovery_sequence(recovery_key)
            if not seq:
                await mark_webhook_processed(
                    idem_key, "confirmation_for_unknown_event", "duplicate")
                return {"status": "unknown_event", "event_id": event.id}
            result = await confirm_live_recovery(
                event, correlation_id, recovery_key=recovery_key,
                amount=payload.get("amount", 0),
            )
            await mark_webhook_processed(
                idem_key,
                f"outcome=confirmed, recovered={result['amount_recovered']}",
                "confirmed",
            )
            return result

        seq, attempt_number = await register_recovery_step(
            key=recovery_key, event_id=event.id, correlation_id=correlation_id,
            max_steps=MAX_STEPS,
        )
        event.attempt_number = attempt_number
        event.max_steps = seq.get("max_steps") or MAX_STEPS

        if attempt_number > event.max_steps:
            publish_recovery_blocked(
                event, correlation_id,
                f"MAX_RECOVERY_STEPS ({event.max_steps}) reached for transaction "
                f"{recovery_key} — no further recovery attempts",
                amount_recovered=0,
            )
            await record_recovery_attempt(
                event_id=event.id, correlation_id=correlation_id,
                attempt_number=event.attempt_number,
                strategy="STOP", action="none", channel="none", amount=event.amount,
                probability=0.0, expected_value=0, policy_verdict="BLOCKED",
                execution_result="exec_max_steps", amount_recovered=0,
                outcome="max_steps", decision_ms=0, execution_ms=0, source="live",
            )
            async with db_session() as db:
                await db.execute(
                    "UPDATE revenue_events SET status = 'blocked' WHERE id = ?",
                    (event.id,),
                )
            await update_recovery_sequence(
                recovery_key, status="max_steps", latest_verdict="BLOCKED"
            )
            await mark_webhook_processed(idem_key, "max_recovery_steps_reached", "blocked")
            return {
                "status": "BLOCKED", "outcome": "max_steps", "action": "none",
                "policy_verdict": "BLOCKED", "attempt": attempt_number,
                "max_steps": event.max_steps, "amount_recovered": 0,
            }

        result = await run_live_recovery(
            event, correlation_id, source="webhook",
            recovery_key=recovery_key, attempt_number=attempt_number,
            max_steps=event.max_steps,
        )
        result.setdefault("attempt", attempt_number)
        result["max_steps"] = event.max_steps
        outcome_status = result.get("status", "processed")
        await mark_webhook_processed(
            idem_key,
            f"outcome={outcome_status}, recovered={result.get('amount_recovered', 0)}",
            status="processed" if outcome_status not in ("pending",) else "pending",
        )

        if outcome_status == "recovered":
            await update_recovery_sequence(
                recovery_key, status="succeeded",
                final_amount=result.get("amount_recovered", 0),
                latest_action=result.get("action"), latest_verdict=result.get("policy_verdict"),
            )
        elif outcome_status in ("pending", "human_review"):
            await update_recovery_sequence(
                recovery_key, latest_action=result.get("action"),
                latest_verdict=result.get("policy_verdict"),
            )
        return result

    q = broadcaster.subscribe()
    try:
        result = asyncio.run(go())
    finally:
        if capture is not None:
            capture[:] = list(q)
        broadcaster.unsubscribe(q)
    return result


def _stages(capture: list) -> dict:
    return {s["type"] for s in capture}


class TestSignatureAndValidation:
    def test_valid_signature_accepted(self, isolated_db):
        from engine.webhook import verify_webhook
        body = json.dumps(_make_payload()).encode()
        assert verify_webhook(body, _sign(body), "") is True

    def test_invalid_signature_rejected(self, isolated_db):
        from engine.webhook import verify_webhook
        body = json.dumps(_make_payload()).encode()
        assert verify_webhook(body, "bad-sig", "") is False

    def test_tampered_body_rejected(self, isolated_db):
        from engine.webhook import verify_webhook
        body = json.dumps(_make_payload()).encode()
        tampered = json.dumps(_make_payload(amount=1)).encode()
        assert verify_webhook(tampered, _sign(body), "") is False

    def test_bad_payloads_rejected(self, isolated_db):
        from engine.webhook import validate_payload, WebhookError
        with pytest.raises(WebhookError):
            validate_payload({"event_type": "payment.failed"})  # no amount
        with pytest.raises(WebhookError):
            validate_payload({"event_type": "payment.captured", "amount": 100})  # no id
        with pytest.raises(WebhookError):
            validate_payload({"event_type": "payment.failed", "amount": "x"})


class TestNormalization:
    def test_normalization_sets_canonical_fields(self, isolated_db):
        from engine.webhook import normalize_payment_webhook
        ev = normalize_payment_webhook(_make_payload())
        # 9 canonical normalized fields + identity/classification.
        assert ev.id == "evt_web1"
        assert ev.transaction_id == "txn_web1"
        assert ev.amount == 2500000
        assert ev.currency == "INR"
        assert ev.decline_code.value == "mandate_afa_required"
        assert ev.type.value == "recurring_payment_failure"
        assert ev.source == "razorpay"
        assert ev.correlation_id == "corr_web1"
        assert ev.occurred_at is not None
        assert ev.received_at is not None
        assert ev.customer.id == "cust_web1"
        assert ev.retry_count == 0

    def test_failure_event_defaults(self, isolated_db):
        from engine.webhook import normalize_payment_webhook
        ev = normalize_payment_webhook({"event_type": "payment.failed", "amount": 5000})
        assert ev.id.startswith("evt_")
        assert ev.transaction_id == ev.id  # falls back to the event id

    def test_confirmation_event_targets_original(self, isolated_db):
        from engine.webhook import normalize_payment_webhook
        ev = normalize_payment_webhook({
            "event_type": "payment.captured", "amount": 2500000,
            "event_id": "evt_web1", "transaction_id": "txn_web1",
        })
        assert ev.id == "evt_web1"
        assert ev.transaction_id == "txn_web1"


class TestIdempotencyExactlyOnce:
    def test_atomic_duplicate_gate_single_processing(self, isolated_db):
        """Same webhook delivered twice must produce exactly one pipeline run:
        one attempt row, one sequence step, one live event."""
        from app.database import get_live_metrics, db_session
        cap = []
        r1 = _process(_make_payload(), cap)
        assert r1["workflow_status"] == "PENDING_WEBHOOK" or r1["status"] == "pending"

        cap2 = []
        r2 = _process(_make_payload(), cap2)
        assert r2["status"] == "duplicate_acknowledged"

        async def _counts():
            async with db_session() as db:
                cur = await db.execute(
                    "SELECT COUNT(*) c FROM recovery_attempts WHERE source='live'")
                attempts = (await cur.fetchone())["c"]
                cur = await db.execute(
                    "SELECT current_step FROM recovery_sequences WHERE key='txn_web1'")
                step = (await cur.fetchone())["current_step"]
            return attempts, step
        attempts, step = asyncio.run(_counts())
        assert attempts == 1
        assert step == 1  # no double step registration

        metrics = asyncio.run(get_live_metrics())

        async def _live_rows():
            async with db_session() as db:
                cur = await db.execute(
                    "SELECT COUNT(*) c FROM revenue_events "
                    "WHERE id NOT LIKE 'txn_%' AND id NOT LIKE 'scenario_%'")
                return (await cur.fetchone())["c"]
        assert asyncio.run(_live_rows()) == 1
        assert r2["message"].startswith("outcome=")


class TestSseStages:
    def test_full_stage_sequence_with_candidates_ev_policy(self, isolated_db):
        cap = []
        res = _process(_make_payload(), cap)
        types = _stages(cap)
        for required in [
            "event.received", "event.normalized", "diagnosis.completed",
            "customer_context.completed", "strategy.candidates_generated",
            "strategy.ranked", "policy.evaluated", "execution.started",
            "execution.completed", "event.completed",
        ]:
            assert required in types, f"missing stage {required} in {sorted(types)}"

        # Every broadcast stage carries the closed-loop identifiers.
        for s in cap:
            assert s.get("transaction_id") == "txn_web1"
            assert s.get("recovery_key") == "txn_web1"
            assert s.get("correlation_id")
            assert "timestamp" in s and s.get("stage")

        cand = next(s for s in cap if s["type"] == "strategy.candidates_generated")
        ranked = next(s for s in cap if s["type"] == "strategy.ranked")
        policy = next(s for s in cap if s["type"] == "policy.evaluated")

        assert cand["payload"]["count"] > 0
        candidates = cand["payload"]["candidates"]
        assert all({"strategy", "action", "probability", "expected_value"} <= set(c)
                   for c in candidates)

        # EV ranking must be non-increasing.
        evs = [c["expected_value"] for c in candidates]
        assert evs == sorted(evs, reverse=True)

        assert ranked["payload"]["selected"] in {c["strategy"] for c in candidates}
        assert ranked["payload"]["reason"]

        assert policy["payload"]["verdict"] == "ALLOW"
        assert policy["payload"]["candidates_considered"] == len(candidates)
        assert res["status"] == "pending"

    def test_ranked_elected_is_top_candidate(self, isolated_db):
        cap = []
        _process(_make_payload(amount=250000, decline_code="bank_timeout"), cap)
        cand = next(s for s in cap if s["type"] == "strategy.candidates_generated")
        ranked = next(s for s in cap if s["type"] == "strategy.ranked")
        top = sorted(cand["payload"]["candidates"],
                     key=lambda c: c["expected_value"], reverse=True)
        assert ranked["payload"]["selected"] == top[0]["strategy"]


class TestPolicyBoundary:
    def test_opted_out_denied_and_blocked(self, isolated_db):
        cap = []
        res = _process(_make_payload(opted_out=True), cap)
        policy = next(s for s in cap if s["type"] == "policy.evaluated")
        assert policy["payload"]["verdict"] == "DENY"
        assert res["status"] == "blocked"
        types = _stages(cap)
        assert "recovery.blocked" in types

    def test_high_value_mandate_never_blind_retry(self, isolated_db):
        """RBI: recurring > ₹15,000 with AFA — must reauthorize, never blind-retry."""
        cap = []
        res = _process(_make_payload(
            event_type="recurring_payment_failure",
            event_id="evt_afa1", transaction_id="txn_afa1",
            amount=2500000, decline_code="mandate_afa_required",
        ), cap)
        policy = next(s for s in cap if s["type"] == "policy.evaluated")
        assert policy["payload"]["verdict"] in ("ALLOW", "MODIFY")
        assert res["status"] == "pending"
        assert res["action"] == "re_authorize_mandate"
        assert res["action"] != "retry_payment"


class TestConfirmationGatedRevenue:
    def test_pending_adds_no_revenue_confirmation_counts(self, isolated_db):
        from app.database import get_live_metrics, db_session, get_open_recovery_sequence
        cap = []
        res = _process(_make_payload(), cap)
        assert res["status"] == "pending"

        metrics = asyncio.run(get_live_metrics())
        assert metrics["live_pending"] == 1
        assert metrics["live_money_recovered"] == 0  # nothing until confirmation

        async def _pending_row():
            async with db_session() as db:
                cur = await db.execute(
                    "SELECT status, recovered_amount FROM revenue_events WHERE id='evt_web1'")
                return dict(await cur.fetchone())
        row = asyncio.run(_pending_row())
        assert row["status"] in ("pending_webhook", "pending")
        assert row["recovered_amount"] == 0

        cap2 = []
        conf = _process(_make_payload(
            event_type="payment.captured", event_id="evt_web1",
            transaction_id="txn_web1", amount=2500000, status="captured",
        ), cap2)
        print("CONF:", conf)
        assert conf["status"] == "recovered"
        assert conf["amount_recovered"] == 2500000

        metrics2 = asyncio.run(get_live_metrics())
        assert metrics2["live_confirmed_payments"] == 1
        assert metrics2["live_money_recovered"] == 2500000

        seq_row = asyncio.run(get_open_recovery_sequence("txn_web1"))
        assert seq_row is None  # sequence closed

        async def _seq():
            from app.database import db_session as _db
            async with _db() as db:
                cur = await db.execute(
                    "SELECT status, final_amount FROM recovery_sequences WHERE key='txn_web1'")
                return dict(await cur.fetchone())
        seq = asyncio.run(_seq())
        assert seq["status"] == "succeeded"
        assert seq["final_amount"] == 2500000

    def test_confirmation_for_unknown_event(self, isolated_db):
        res = _process(_make_payload(
            event_type="payment.captured", event_id="evt_nope",
            transaction_id="txn_nope", status="captured",
        ))
        assert res["status"] == "unknown_event"


class TestClosedLoopReoptimization:
    def test_failed_failed_confirmed(self, isolated_db):
        from app.database import db_session
        cap1 = []
        r1 = _process(_make_payload(), cap1)
        assert r1["status"] == "pending"

        cap2 = []
        r2 = _process(_make_payload(
            event_id="evt_web1b", retry_count=1,
        ), cap2)
        assert r2["status"] == "pending"
        assert r2["attempt"] == 2
        assert r2["max_steps"] == 3

        cap3 = []
        r3 = _process(_make_payload(
            event_type="payment.captured", event_id="evt_web1",
            transaction_id="txn_web1", amount=2500000, status="captured",
        ), cap3)
        assert r3["status"] == "recovered"
        assert r3["amount_recovered"] == 2500000

        async def _seq():
            async with db_session() as db:
                cur = await db.execute(
                    "SELECT current_step, status, final_amount, event_ids "
                    "FROM recovery_sequences WHERE key='txn_web1'")
                return dict(await cur.fetchone())
        seq = asyncio.run(_seq())
        assert seq["current_step"] == 2  # two failure inbound events observed
        assert seq["status"] == "succeeded"
        assert seq["final_amount"] == 2500000
        ids = json.loads(seq["event_ids"])
        assert "evt_web1" in ids and "evt_web1b" in ids


class TestMaxRecoveryStepsBlock:
    def test_fourth_failure_blocked_no_further_optimization(self, isolated_db):
        from app.database import db_session, get_live_metrics
        outcomes = []
        for i in range(1, 4):
            cap = []
            outcomes.append(_process(_make_payload(
                event_id=f"evt_m{i}", retry_count=i - 1,
            ), cap).get("status"))
        assert outcomes == ["pending", "pending", "pending"]

        cap4 = []
        r4 = _process(_make_payload(event_id="evt_m4", retry_count=3), cap4)
        assert r4["status"] == "BLOCKED"
        assert r4["outcome"] == "max_steps"
        assert r4["attempt"] == 4
        types = _stages(cap4)
        assert "recovery.blocked" in types

        # No further re-optimization: 4th failure produced a blocked row, not a
        # pipeline attempt — but it must be explicit that only 3 pipeline-runs
        # happened (attempts exec_pendingx3 + 1 exec_max_steps).
        async def _attempts():
            async with db_session() as db:
                cur = await db.execute(
                    "SELECT outcome FROM recovery_attempts WHERE source='live' ORDER BY attempted_at")
                return [r["outcome"] for r in await cur.fetchall()]
        rows = asyncio.run(_attempts())
        assert rows.count("max_steps") == 1
        assert len(rows) == 4  # 3 pipeline attempts + 1 blocked record
        assert rows[:3] == ["pending", "pending", "pending"]

        async def _seq():
            async with db_session() as db:
                cur = await db.execute(
                    "SELECT status, latest_verdict, current_step FROM recovery_sequences "
                    "WHERE key='txn_web1'")
                return dict(await cur.fetchone())
        seq = asyncio.run(_seq())
        assert seq["status"] == "max_steps"
        assert seq["latest_verdict"] == "BLOCKED"
        assert seq["current_step"] == 4

        metrics = asyncio.run(get_live_metrics())
        assert metrics["open_recovery_sequences"] == 0  # boundary closed, not open
        assert metrics["live_money_recovered"] == 0  # nothing confirmed


class TestLiveMetrics:
    def test_metrics_reflect_closed_loop(self, isolated_db):
        from app.database import get_live_metrics
        _process(_make_payload())
        _process(_make_payload(
            event_type="payment.captured", event_id="evt_web1",
            transaction_id="txn_web1", amount=2500000, status="captured",
        ))
        m = asyncio.run(get_live_metrics())
        assert m["live_events"] == 1
        assert m["live_confirmed_payments"] == 1
        assert m["live_money_recovered"] == 2500000
        assert m["open_recovery_sequences"] == 0
        assert m["live_recovery_rate"] == 1.0
        assert m["avg_decision_ms"] >= 0
        assert "time_to_confirmation_sec" in m


class TestHttpEndpoint:
    """Exercise the REAL FastAPI webhook endpoints over HTTP (TestClient).

    The on-the-wire path is what the simulator + Razorpay webhooks hit, and it
    must match the internal pipeline exactly: signed ingress -> PENDING -> trusted
    capture confirmation -> RESOLVED with recovered amount, exactly-once dedup,
    invalid signatures rejected, and live metrics reflecting the closed loop.
    """

    def _signed_post(self, client, payload):
        from tools.simulate_realtime import derive_secret, compute_signature
        body = json.dumps(payload).encode()
        sig = compute_signature(body, derive_secret())
        return client.post(
            "/api/webhooks/payment", content=body,
            headers={"Content-Type": "application/json", "X-Signature": sig},
        )

    def test_full_http_closed_loop_confirm(self, isolated_db):
        from app.main import app
        with TestClient(app) as c:
            r1 = self._signed_post(c, _make_payload())
            assert r1.status_code == 200
            j1 = r1.json()
            assert j1["status"] == "PENDING_WEBHOOK"
            assert j1["outcome"] == "pending"
            assert j1["action"] == "re_authorize_mandate"
            assert j1["amount_recovered"] == 0
            assert j1["mode"] == "demo"

            # Confirmation over the same conduit must resolve + record money.
            # Regression: previously raised NameError (WEBHOOK_MODE) -> HTTP 500.
            r2 = self._signed_post(c, _make_payload(
                event_type="payment.captured", event_id="evt_web1",
                transaction_id="txn_web1", amount=2500000, status="captured",
            ))
            assert r2.status_code == 200, f"confirmation must not 500: {r2.text}"
            j2 = r2.json()
            assert j2["status"] == "RESOLVED"
            assert j2["outcome"] == "recovered"
            assert j2["amount_recovered"] == 2500000
            assert j2["mode"] == "demo"

            # Closed loop: sequence closed, confirmed money in live metrics.
            m = c.get("/api/live/metrics").json()
            assert m["live_money_recovered"] == 2500000
            assert m["live_confirmed_payments"] == 1
            assert m["open_recovery_sequences"] == 0

    def test_http_duplicate_replay_exactly_once(self, isolated_db):
        from app.main import app
        with TestClient(app) as c:
            r1 = self._signed_post(c, _make_payload())
            assert r1.status_code == 200
            r2 = self._signed_post(c, _make_payload())
            assert r2.status_code == 200
            j2 = r2.json()
            assert j2["status"] == "duplicate_acknowledged"
            assert j2["message"].startswith("outcome=")

            # Only one pipeline attempt, one live event row.
            m = c.get("/api/live/metrics").json()
            assert m["live_events"] == 1
            assert m["live_recovery_attempts"] == 1

    def test_http_invalid_signature_rejected(self, isolated_db):
        from app.main import app
        with TestClient(app) as c:
            body = json.dumps(_make_payload()).encode()
            resp = c.post(
                "/api/webhooks/payment", content=body,
                headers={"Content-Type": "application/json", "X-Signature": "bad-sig"},
            )
            assert resp.status_code == 401


class TestIsolation:
    def test_live_pipeline_does_not_mutate_benchmark(self, isolated_db):
        from engine.webhook import normalize_payment_webhook, new_correlation_id
        from engine.realtime import run_live_recovery
        from engine.pipeline import load_batch, process_batch

        batch1 = asyncio.run(process_batch(load_batch()))

        for i in range(3):
            ev = normalize_payment_webhook({
                "event_type": "payment.failed", "amount": 30000 + i * 10000,
                "customer_id": f"cust_l{i}", "decline_code": "insufficient_funds",
            })
            asyncio.run(run_live_recovery(ev, new_correlation_id()))

        batch2 = asyncio.run(process_batch(load_batch()))
        assert batch1.recovered == batch2.recovered == EXPECTED_BATCH_REVOVERED_COUNT
        assert batch1.recovered_amount == batch2.recovered_amount == EXPECTED_BATCH_REVOVERED_AMOUNT

    def test_evaluation_does_not_mutate_benchmark(self, isolated_db):
        from engine.evaluation import run_evaluation
        from engine.pipeline import load_batch, process_batch

        baseline = asyncio.run(process_batch(load_batch()))
        metrics = asyncio.run(run_evaluation())  # isolated eval DB (temp)
        assert metrics.total_events >= 20  # the 20-scenario guardrail suite ran

        after = asyncio.run(process_batch(load_batch()))
        assert after.recovered == baseline.recovered == EXPECTED_BATCH_REVOVERED_COUNT
        assert after.recovered_amount == baseline.recovered_amount == EXPECTED_BATCH_REVOVERED_AMOUNT
        assert after.recovered_amount == 9406014


class TestAuditChainVerification:
    """Regression for the audit-chain hash bug.

    log_supervisor_decision used to hash with a freshly-called timestamp string
    (~17µs AFTER the one that was stored), so /api/audit/verify reported every
    entry broken on a clean chain. The writer and verifier must agree
    byte-for-byte on the STORED row.
    """

    def test_written_chain_verifies_and_detects_tamper(self, isolated_db):
        import hashlib
        from app.database import db_session

        # Drive the real pipeline so the supervisor logs a genuine audit entry.
        res = _process(_make_payload())
        assert res["status"] == "pending"

        async def _latest():
            async with db_session() as db:
                cur = await db.execute(
                    "SELECT * FROM audit_log WHERE event_id='evt_web1' ORDER BY timestamp DESC LIMIT 1")
                return dict(await cur.fetchone())
        row = asyncio.run(_latest())
        assert row is not None
        assert row["rule_version"] == "4.0.0"

        def recompute(r, prev):
            payload = f"{r['id']}|{r['timestamp']}|{r['event_id']}|{r['workflow_status']}|{r['action']}|{r['result']}|{prev}"
            return hashlib.sha256(payload.encode()).hexdigest()

        # Writer hash must match the verifier's recomputation from stored fields.
        assert row["entry_hash"] == recompute(row, row["prev_hash"]), \
            "audit hash does not match the stored timestamp (writer/verifier divergence)"

        # Clean chain verifies over the real HTTP endpoint.
        from app.main import app
        with TestClient(app) as c:
            v1 = c.get("/api/audit/verify").json()
            assert v1["valid"] is True, f"clean chain must verify, got {v1}"

        # Tamper with a stored field -> verifier must now fail on that row.
        async def _tamper():
            async with db_session() as db:
                await db.execute(
                    "UPDATE audit_log SET action='TAMPERED' WHERE event_id='evt_web1'")
        asyncio.run(_tamper())

        row2 = asyncio.run(_latest())
        assert row2["entry_hash"] != recompute(row2, row2["prev_hash"])
        with TestClient(app) as c:
            v2 = c.get("/api/audit/verify").json()
            assert v2["valid"] is False
            assert v2["broken_entry_id"] == row2["id"]


class TestRetryFamilyConfirmationGate:
    """P1 #5: live retry family must NOT finalize money without a trusted
    payment.captured confirmation — while the batch benchmark keeps its
    immediate-capture behaviour on discrete txn_* events."""

    def test_live_retry_waits_for_confirmation_but_batch_recovers(self, isolated_db):
        from app.database import get_live_metrics
        from engine.pipeline import load_batch, process_batch

        # Batch benchmark is untouched: retry_payment/insufficient_funds -> capture.
        batch = asyncio.run(process_batch(load_batch()))
        assert batch.recovered == EXPECTED_BATCH_REVOVERED_COUNT
        assert batch.recovered_amount == EXPECTED_BATCH_REVOVERED_AMOUNT

        # Live event, same decline_code, ₹500 (below AFA threshold): previously
        # this RESOLVED immediately with ₹500; it must now stay pending at ₹0.
        cap = []
        r1 = _process(_make_payload(
            event_type="payment.failed", event_id="evt_retry1",
            transaction_id="txn_retry1", amount=50000,
            decline_code="insufficient_funds", retry_count=0,
        ), cap)
        assert r1["action"] == "retry_payment"
        assert r1["status"] == "pending"
        assert r1["workflow_status"] in ("PENDING_WEBHOOK", "pending")
        assert r1["amount_recovered"] == 0

        m1 = asyncio.run(get_live_metrics())
        assert m1["live_money_recovered"] == 0  # nothing counted yet

        # Only the trusted payment.captured confirmation finalizes the money.
        r2 = _process(_make_payload(
            event_type="payment.captured", event_id="evt_retry1",
            transaction_id="txn_retry1", amount=50000, status="captured",
        ))
        assert r2["status"] == "recovered"
        assert r2["amount_recovered"] == 50000

        m2 = asyncio.run(get_live_metrics())
        assert m2["live_confirmed_payments"] == 1
        assert m2["live_money_recovered"] == 50000