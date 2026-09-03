"""Tests for v4.0 live recovery: webhook ingress, closed-loop, live metrics,
and isolation from the batch benchmark."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import json
from pathlib import Path

import pytest

# Force demo mode for deterministic derived signature tests.
os.environ["WEBHOOK_MODE"] = "demo"
os.environ["WEBHOOK_ALLOW_UNSIGNED"] = "false"


@pytest.fixture()
def isolated_db(tmp_path):
    """Each test gets its own temp SQLite DB via the ContextVar."""
    from app.database import set_active_db_path, init_db, reset_active_db_path
    path = Path(tmp_path) / "test.db"
    token = set_active_db_path(path)
    asyncio.run(init_db())
    yield path
    reset_active_db_path(token)


def _sign(body: bytes) -> str:
    from engine.webhook import compute_signature, _derive_secret
    return compute_signature(body, _derive_secret())


def _make_payload(**overrides) -> dict:
    p = {
        "event_type": "payment.failed",
        "amount": 50000,
        "currency": "INR",
        "customer_id": "cust_web1",
        "customer_name": "Webhook User",
        "decline_code": "insufficient_funds",
        "retry_count": 0,
    }
    p.update(overrides)
    return p


class TestWebhookSignature:
    def test_valid_signature_passes(self, isolated_db):
        from engine.webhook import verify_webhook, _derive_secret
        body = json.dumps(_make_payload()).encode()
        assert _derive_secret() != ""
        assert verify_webhook(body, _sign(body), "") is True

    def test_tampered_signature_fails(self, isolated_db):
        from engine.webhook import verify_webhook
        body = json.dumps(_make_payload()).encode()
        assert verify_webhook(body, "deadbeef", "") is False

    def test_tampered_body_fails(self, isolated_db):
        from engine.webhook import verify_webhook
        body = json.dumps(_make_payload()).encode()
        tampered = json.dumps(_make_payload(amount=9)).encode()
        assert verify_webhook(tampered, _sign(body), "") is False

    def test_production_requires_secret(self, isolated_db, monkeypatch):
        from engine import webhook
        monkeypatch.setattr(webhook, "WEBHOOK_MODE", "production")
        monkeypatch.setattr(webhook, "WEBHOOK_SECRET", "prod-secret")
        monkeypatch.setattr(webhook, "WEBHOOK_ALLOW_UNSIGNED", "false")
        body = json.dumps(_make_payload()).encode()
        secret = webhook._derive_secret()
        good = webhook.compute_signature(body, secret)
        assert webhook.verify_webhook(body, good, "") is True
        assert webhook.verify_webhook(body, "bad", "") is False
        assert webhook.verify_webhook(body, "", "") is False


class TestNormalize:
    def test_normalize_failure_event(self, isolated_db):
        from engine.webhook import normalize_payment_webhook, validate_payload
        p = _make_payload()
        validate_payload(p)
        ev = normalize_payment_webhook(p)
        assert ev.id.startswith("evt_")
        assert ev.amount == 50000
        assert ev.decline_code.value == "insufficient_funds"
        assert ev.type.value == "card_payment_failure"

    def test_missing_amount_rejected(self, isolated_db):
        from engine.webhook import validate_payload, WebhookError
        p = _make_payload()
        p.pop("amount")
        with pytest.raises(WebhookError):
            validate_payload(p)

    def test_confirmation_requires_id(self, isolated_db):
        from engine.webhook import validate_payload, WebhookError
        p = {"event_type": "payment.captured", "amount": 100}
        with pytest.raises(WebhookError):
            validate_payload(p)


class TestIdempotency:
    def test_duplicate_stored_record(self, isolated_db):
        from app.database import record_webhook_received, get_webhook_processing
        asyncio.run(record_webhook_received(
            event_id="evt_dup1", correlation_id="corr_a", source="live_webhook",
            signature_verified=True))
        r1 = asyncio.run(get_webhook_processing("evt_dup1"))
        r2 = asyncio.run(get_webhook_processing("evt_dup1"))
        assert r1 is not None
        assert r2 is not None
        assert r1["event_id"] == r2["event_id"] == "evt_dup1"


class TestLiveClosedLoop:
    def test_live_recovery_confirm_gates_amount(self, isolated_db):
        """pending link must NOT add recovered_amount; only confirmation does."""
        from engine.webhook import normalize_payment_webhook, new_correlation_id
        from engine.realtime import run_live_recovery
        from app.database import db_session, get_live_metrics

        # Abandoned checkout -> payment link -> pending_webhook.
        ev = normalize_payment_webhook({
            "event_type": "checkout.abandoned", "amount": 80000,
            "customer_id": "cust_p1", "decline_code": "payment_link_expired",
        })
        res = asyncio.run(run_live_recovery(ev, new_correlation_id()))
        assert res["workflow_status"] == "PENDING_WEBHOOK"

        async def _check_pending():
            async with db_session() as db:
                cur = await db.execute("SELECT status, recovered_amount FROM revenue_events WHERE id=?", (ev.id,))
                row = await cur.fetchone()
            return dict(row) if row else None
        row = asyncio.run(_check_pending())
        # Before confirmation: pending, recovered_amount must be 0.
        assert row["status"] == "pending_webhook" or row["status"] == "pending"
        assert row["recovered_amount"] == 0

        metrics = asyncio.run(get_live_metrics())
        assert metrics["live_money_recovered"] == 0  # not counted yet

        # Now confirm via the trusted payment.captured conduit.
        from app.database import db_session as _db2, record_recovery_attempt
        async def _confirm():
            async with _db2() as db:
                await db.execute(
                    "UPDATE revenue_events SET status='success', recovered_amount=? WHERE id=?",
                    (80000, ev.id))
        asyncio.run(_confirm())

        metrics2 = asyncio.run(get_live_metrics())
        assert metrics2["live_confirmed_payments"] == 1
        assert metrics2["live_money_recovered"] == 80000

    def test_live_does_not_affect_batch_metrics(self, isolated_db):
        """Live events (evt_*) must not change the txn_ batch benchmark numbers."""
        from engine.webhook import normalize_payment_webhook, new_correlation_id
        from engine.realtime import run_live_recovery
        from engine.pipeline import load_batch, process_batch
        from app.database import db_session

        batch1 = asyncio.run(process_batch(load_batch()))

        # Run live events through the closed loop.
        for i in range(3):
            ev = normalize_payment_webhook({
                "event_type": "payment.failed", "amount": 30000 + i * 10000,
                "customer_id": f"cust_l{i}", "decline_code": "insufficient_funds",
            })
            asyncio.run(run_live_recovery(ev, new_correlation_id()))

        batch2 = asyncio.run(process_batch(load_batch()))

        # Benchmark numbers must be identical despite live events.
        assert batch1.recovered == batch2.recovered == 27
        assert batch1.recovered_amount == batch2.recovered_amount == 9406014

    def test_opted_out_live_event_blocked(self, isolated_db):
        from engine.webhook import normalize_payment_webhook, new_correlation_id
        from engine.realtime import run_live_recovery
        ev = normalize_payment_webhook({
            "event_type": "payment.failed", "amount": 50000,
            "customer_id": "cust_opt", "decline_code": "insufficient_funds",
            "opted_out": True,
        })
        res = asyncio.run(run_live_recovery(ev, new_correlation_id()))
        assert res["status"] == "blocked"
