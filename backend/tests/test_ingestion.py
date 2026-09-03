import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import json
import hashlib
import hmac
import pytest
from datetime import datetime

from app.database import init_db, db_session
from engine.ingestion import (
    EventBroadcaster,
    verify_razorpay_signature,
    normalize_razorpay_webhook,
    normalize_simulator_event,
    is_duplicate_event,
    _classify_event,
    _map_razorpay_error,
)
from app.models import EventType, DeclineCode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_db():
    """Ensure the test database is initialized before each test."""
    asyncio.get_event_loop().run_until_complete(init_db())


@pytest.fixture
def razorpay_payment_failed():
    return {
        "event": "payment.failed",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_abc123",
                    "amount": 100000,
                    "currency": "INR",
                    "error_code": "insufficient_funds",
                    "error_description": "Insufficient funds",
                    "card": {"last4": "4242", "issuer": "HDFC"},
                    "notes": {
                        "event_id": "evt_rzp_001",
                        "customer_id": "cust_rzp_001",
                        "customer_name": "RZP Test User",
                        "customer_phone": "+919876543210",
                        "customer_email": "rzp@test.com",
                    },
                }
            }
        },
    }


@pytest.fixture
def razorpay_subscription():
    return {
        "event": "subscription.authenticated",
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_abc123",
                    "amount": 200000,
                    "currency": "INR",
                    "mandate_id": "mandate_123",
                    "notes": {
                        "event_id": "evt_rzp_sub_001",
                        "customer_id": "cust_rzp_sub_001",
                        "customer_name": "Sub User",
                    },
                }
            }
        },
    }


# ---------------------------------------------------------------------------
# EventBroadcaster tests
# ---------------------------------------------------------------------------

class TestEventBroadcaster:
    def test_subscribe_returns_queue(self):
        bc = EventBroadcaster()
        q = bc.subscribe()
        assert isinstance(q, list)
        assert len(q) == 0

    def test_broadcast_delivers_to_subscribers(self):
        bc = EventBroadcaster()
        q1 = bc.subscribe()
        q2 = bc.subscribe()

        bc.broadcast({"type": "test", "value": 42})

        assert len(q1) == 1
        assert q1[0] == {"type": "test", "value": 42}
        assert len(q2) == 1
        assert q2[0] == {"type": "test", "value": 42}

    def test_unsubscribe_removes_queue(self):
        bc = EventBroadcaster()
        q = bc.subscribe()
        bc.unsubscribe(q)

        bc.broadcast({"type": "test"})
        assert len(q) == 0

    def test_unsubscribe_nonexistent_is_safe(self):
        bc = EventBroadcaster()
        q = []
        bc.unsubscribe(q)  # should not raise

    def test_broadcast_with_no_subscribers(self):
        bc = EventBroadcaster()
        bc.broadcast({"type": "test"})  # should not raise

    def test_multiple_broadcasts_accumulate(self):
        bc = EventBroadcaster()
        q = bc.subscribe()

        bc.broadcast({"i": 1})
        bc.broadcast({"i": 2})
        bc.broadcast({"i": 3})

        assert len(q) == 3
        assert [e["i"] for e in q] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Signature verification tests
# ---------------------------------------------------------------------------

class TestSignatureVerification:
    def test_valid_signature(self):
        secret = "whsec_test123"
        body = b'{"event":"payment.failed"}'
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

        assert verify_razorpay_signature(body, expected, secret) is True

    def test_invalid_signature(self):
        secret = "whsec_test123"
        body = b'{"event":"payment.failed"}'
        bad_sig = "0" * 64

        assert verify_razorpay_signature(body, bad_sig, secret) is False

    def test_empty_secret_returns_false(self):
        assert verify_razorpay_signature(b"body", "sig", "") is False

    def test_tampered_body_fails(self):
        secret = "whsec_test123"
        body = b'{"event":"payment.failed"}'
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

        tampered = b'{"event":"payment.captured"}'
        assert verify_razorpay_signature(tampered, sig, secret) is False


# ---------------------------------------------------------------------------
# Razorpay error mapping tests
# ---------------------------------------------------------------------------

class TestRazorpayErrorMapping:
    def test_known_errors(self):
        assert _map_razorpay_error("bad_expired_card") == DeclineCode.EXPIRED_CARD
        assert _map_razorpay_error("insufficient_funds") == DeclineCode.INSUFFICIENT_FUNDS
        assert _map_razorpay_error("do_not_honor") == DeclineCode.DO_NOT_HONOR
        assert _map_razorpay_error("bank_timeout") == DeclineCode.BANK_TIMEOUT
        assert _map_razorpay_error("incorrect_cvc") == DeclineCode.INCORRECT_CVC
        assert _map_razorpay_error("processing_error") == DeclineCode.PROCESSING_ERROR

    def test_unknown_error_defaults_to_do_not_honor(self):
        assert _map_razorpay_error("something_weird") == DeclineCode.DO_NOT_HONOR

    def test_empty_string_defaults(self):
        assert _map_razorpay_error("") == DeclineCode.DO_NOT_HONOR


# ---------------------------------------------------------------------------
# _classify_event tests
# ---------------------------------------------------------------------------

class TestClassifyEvent:
    def test_payment_failed_default(self):
        et, dc = _classify_event("payment.failed", "", 50000)
        assert et == EventType.CARD_PAYMENT_FAILURE
        assert dc == DeclineCode.INSUFFICIENT_FUNDS

    def test_payment_failed_with_explicit_decline(self):
        et, dc = _classify_event("payment.failed", "expired_card", 50000)
        assert et == EventType.CARD_PAYMENT_FAILURE
        assert dc == DeclineCode.EXPIRED_CARD

    def test_recurring_payment_failure(self):
        et, dc = _classify_event("recurring_payment_failure", "", 100000)
        assert et == EventType.RECURRING_PAYMENT_FAILURE
        assert dc == DeclineCode.MANDATE_SIMPLE_RETRY

    def test_checkout_abandonment(self):
        et, dc = _classify_event("checkout_abandonment", "", 99900)
        assert et == EventType.CHECKOUT_ABANDONMENT
        assert dc == DeclineCode.PAYMENT_LINK_EXPIRED

    def test_overdue_invoice(self):
        et, dc = _classify_event("overdue_invoice", "", 500000)
        assert et == EventType.OVERDUE_INVOICE
        assert dc == DeclineCode.INVOICE_OVERDUE

    def test_unknown_type_defaults_to_card_failure(self):
        et, dc = _classify_event("unknown_event_type", "", 10000)
        assert et == EventType.CARD_PAYMENT_FAILURE
        assert dc == DeclineCode.INSUFFICIENT_FUNDS

    def test_explicit_decline_overrides_default(self):
        et, dc = _classify_event("recurring_payment_failure", "mandate_afa_required", 100000)
        assert dc == DeclineCode.MANDATE_AFA_REQUIRED


# ---------------------------------------------------------------------------
# normalize_razorpay_webhook tests
# ---------------------------------------------------------------------------

class TestNormalizeRazorpayWebhook:
    def test_payment_failed(self, razorpay_payment_failed):
        event = normalize_razorpay_webhook(razorpay_payment_failed)
        assert event is not None
        assert event.id == "evt_rzp_001"
        assert event.type == EventType.CARD_PAYMENT_FAILURE
        assert event.amount == 100000
        assert event.customer.id == "cust_rzp_001"
        assert event.customer.name == "RZP Test User"
        assert event.decline_code == DeclineCode.INSUFFICIENT_FUNDS
        assert event.metadata.card_last4 == "4242"
        assert event.metadata.bank == "HDFC"

    def test_subscription_event(self, razorpay_subscription):
        event = normalize_razorpay_webhook(razorpay_subscription)
        assert event is not None
        assert event.type == EventType.RECURRING_PAYMENT_FAILURE
        assert event.metadata.mandate_id == "mandate_123"

    def test_empty_payload_returns_none(self):
        assert normalize_razorpay_webhook({}) is None

    def test_no_entity_returns_none(self):
        assert normalize_razorpay_webhook({"event": "payment.failed", "payload": {}}) is None

    def test_missing_event_id_generates_one(self, razorpay_payment_failed):
        del razorpay_payment_failed["payload"]["payment"]["entity"]["notes"]["event_id"]
        event = normalize_razorpay_webhook(razorpay_payment_failed)
        assert event is not None
        assert event.id.startswith("evt_")

    def test_opted_out_customer(self, razorpay_payment_failed):
        razorpay_payment_failed["payload"]["payment"]["entity"]["notes"]["opted_out"] = "true"
        event = normalize_razorpay_webhook(razorpay_payment_failed)
        assert event.customer.opted_out is True

    def test_unknown_event_type_maps_to_card_failure(self):
        payload = {
            "event": "some.unknown.event",
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_unknown",
                        "amount": 50000,
                        "currency": "INR",
                        "notes": {"event_id": "evt_unknown_001", "customer_id": "cust_1"},
                    }
                }
            },
        }
        event = normalize_razorpay_webhook(payload)
        assert event is not None
        assert event.type == EventType.CARD_PAYMENT_FAILURE
        assert event.decline_code == DeclineCode.DO_NOT_HONOR


# ---------------------------------------------------------------------------
# normalize_simulator_event tests
# ---------------------------------------------------------------------------

class TestNormalizeSimulatorEvent:
    def test_default_event(self):
        event = normalize_simulator_event({})
        assert event.id.startswith("evt_sim_")
        assert event.type == EventType.CARD_PAYMENT_FAILURE
        assert event.amount == 50000
        assert event.currency == "INR"

    def test_custom_event_type(self):
        event = normalize_simulator_event({
            "event_type": "recurring_payment_failure",
            "amount": 200000,
        })
        assert event.type == EventType.RECURRING_PAYMENT_FAILURE
        assert event.amount == 200000
        assert event.decline_code == DeclineCode.MANDATE_SIMPLE_RETRY

    def test_checkout_abandonment(self):
        event = normalize_simulator_event({"event_type": "checkout_abandonment"})
        assert event.type == EventType.CHECKOUT_ABANDONMENT
        assert event.decline_code == DeclineCode.PAYMENT_LINK_EXPIRED

    def test_overdue_invoice(self):
        event = normalize_simulator_event({"event_type": "overdue_invoice", "amount": 1000000})
        assert event.type == EventType.OVERDUE_INVOICE
        assert event.amount == 1000000

    def test_explicit_customer_id(self):
        event = normalize_simulator_event({"customer_id": "cust_custom_123"})
        assert event.customer.id == "cust_custom_123"

    def test_auto_generated_customer_id(self):
        event = normalize_simulator_event({})
        assert event.customer.id.startswith("cust_sim_")

    def test_explicit_decline_code(self):
        event = normalize_simulator_event({
            "event_type": "payment.failed",
            "decline_code": "expired_card",
        })
        assert event.decline_code == DeclineCode.EXPIRED_CARD

    def test_retry_count_preserved(self):
        event = normalize_simulator_event({"retry_count": 3})
        assert event.retry_count == 3
        assert event.metadata.previous_attempts == 3

    def test_opted_out_flag(self):
        event = normalize_simulator_event({"opted_out": True})
        assert event.customer.opted_out is True

    def test_language_pref(self):
        event = normalize_simulator_event({"language_pref": "en"})
        assert event.customer.language_pref == "en"

    def test_ground_truth_is_uncertain(self):
        event = normalize_simulator_event({})
        assert event.ground_truth == "uncertain"

    def test_status_is_pending(self):
        event = normalize_simulator_event({})
        assert event.status == "pending"


# ---------------------------------------------------------------------------
# Idempotency tests (requires DB)
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_new_event_not_duplicate(self):
        result = asyncio.get_event_loop().run_until_complete(
            is_duplicate_event("evt_nonexistent_999")
        )
        assert result is False

    def test_existing_event_is_duplicate(self):
        async def _test():
            async with db_session() as db:
                await db.execute(
                    """INSERT OR IGNORE INTO revenue_events
                       (id, type, customer_id, customer_name, amount, currency,
                        root_cause, decline_code, failed_at, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("evt_dup_test_001", "card_payment_failure", "cust_1",
                     "Test", 50000, "INR", "insufficient_funds", "insufficient_funds",
                     datetime.utcnow().isoformat(), "pending"),
                )
            return await is_duplicate_event("evt_dup_test_001")

        result = asyncio.get_event_loop().run_until_complete(_test())
        assert result is True

    def test_different_event_not_duplicate(self):
        result = asyncio.get_event_loop().run_until_complete(
            is_duplicate_event("evt_completely_different")
        )
        assert result is False
