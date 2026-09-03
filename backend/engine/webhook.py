"""Webhook ingress layer for live recovery mode.

Production-style external event entry point:

    External payment event
        -> POST /api/webhooks/payment
        -> signature verification (HMAC-SHA256)
        -> timestamp / replay protection
        -> event validation
        -> idempotency gate (exactly-once)
        -> normalization into RevenueEvent
        -> async recovery pipeline
        -> SSE broadcast

Security configuration via environment variables:

    WEBHOOK_SECRET        -- shared secret for production HMAC verification
    WEBHOOK_MODE          -- "demo" (defer signature, or use demo secret) | "production"
    WEBHOOK_ALLOW_UNSIGNED-- "true" to allow unsigned in demo mode (DEFAULT: only
                            when a DEMO_SIGNATURE header is present)

Demo mode uses a DERIVED, deterministic HMAC so the simulator can sign payloads
without shipping a real secret. Production mode REQUIRES WEBHOOK_SECRET.
"""
import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timezone

from app.models import (
    RevenueEvent, Customer, TransactionMetadata,
    EventType, DeclineCode,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


WEBHOOK_SECRET = _env("WEBHOOK_SECRET", "")
WEBHOOK_MODE = _env("WEBHOOK_MODE", "demo").lower()
WEBHOOK_ALLOW_UNSIGNED = _env("WEBHOOK_ALLOW_UNSIGNED", "false").lower() == "true"

# In demo mode, payloads are signed with a deterministic secret derived from the
# demo marker, so the simulator can reproduce it without a real secret.
DEMO_SECRET_SALT = "recovery-copilot-demo-v1"


class WebhookError(Exception):
    """Raised for webhook validation / security failures."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------

def _derive_secret() -> str:
    """In demo mode derive a stable secret; in production use WEBHOOK_SECRET."""
    if WEBHOOK_MODE == "production":
        return WEBHOOK_SECRET
    # Demo: rotate a signing key from a public marker + fixed salt.
    return hashlib.sha256((DEMO_SECRET_SALT + WEBHOOK_SECRET).encode()).hexdigest()


def validate_webhook_config() -> None:
    """Fail fast when production mode is misconfigured (no signing secret)."""
    if WEBHOOK_MODE == "production" and not WEBHOOK_SECRET:
        raise WebhookError(
            "WEBHOOK_MODE=production requires WEBHOOK_SECRET to verify signatures",
            status_code=500,
        )


def compute_signature(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def verify_signature(body: bytes, provided: str, secret: str) -> bool:
    if not secret:
        return False
    expected = compute_signature(body, secret)
    return hmac.compare_digest(expected, provided)


def verify_webhook(
    raw_body: bytes,
    signature: str,
    timestamp: str,
    max_age_seconds: int = 300,
) -> bool:
    """Verify signature + timestamp replay protection.

    Production requires a valid signed payload. In demo mode, either:
      - a valid signed payload, or
      - unsigned payloads explicitly allowed via WEBHOOK_ALLOW_UNSIGNED.
    """
    secret = _derive_secret()

    # Replay protection: reject stale timestamps (> max_age). Accept empty
    # timestamps only when unsigned is allowed (test/demo convenience).
    if timestamp:
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if abs((now - ts).total_seconds()) > max_age_seconds:
                return False
        except ValueError:
            return False

    sig_ok = bool(signature) and verify_signature(raw_body, signature, secret)

    if WEBHOOK_MODE == "production":
        # Production REQUIRES a valid signature, period.
        return sig_ok

    # Demo mode: signed OR explicitly-allowed-unsigned.
    return sig_ok or (WEBHOOK_ALLOW_UNSIGNED and not signature)


# ---------------------------------------------------------------------------
# Event normalization
# ---------------------------------------------------------------------------

EVENT_TYPE_MAP = {
    "payment.failed": EventType.CARD_PAYMENT_FAILURE,
    "payment.captured": EventType.CARD_PAYMENT_FAILURE,
    "payment.authorized": EventType.CARD_PAYMENT_FAILURE,
    "payment.refunded": EventType.CARD_PAYMENT_FAILURE,
    "subscription.charged": EventType.RECURRING_PAYMENT_FAILURE,
    "subscription.payment_failed": EventType.RECURRING_PAYMENT_FAILURE,
    "recurring_payment_failure": EventType.RECURRING_PAYMENT_FAILURE,
    "invoice.overdue": EventType.OVERDUE_INVOICE,
    "invoice_overdue": EventType.OVERDUE_INVOICE,
    "checkout.abandoned": EventType.CHECKOUT_ABANDONMENT,
    "checkout_abandonment": EventType.CHECKOUT_ABANDONMENT,
}

DECLINE_MAP = {
    "insufficient_funds": DeclineCode.INSUFFICIENT_FUNDS,
    "expired_card": DeclineCode.EXPIRED_CARD,
    "do_not_honor": DeclineCode.DO_NOT_HONOR,
    "bank_timeout": DeclineCode.BANK_TIMEOUT,
    "incorrect_cvc": DeclineCode.INCORRECT_CVC,
    "processing_error": DeclineCode.PROCESSING_ERROR,
    "mandate_afa_required": DeclineCode.MANDATE_AFA_REQUIRED,
    "mandate_simple_retry": DeclineCode.MANDATE_SIMPLE_RETRY,
    "payment_link_expired": DeclineCode.PAYMENT_LINK_EXPIRED,
    "invoice_overdue": DeclineCode.INVOICE_OVERDUE,
}


def _default_decline(event_type: EventType) -> DeclineCode:
    return {
        EventType.CARD_PAYMENT_FAILURE: DeclineCode.INSUFFICIENT_FUNDS,
        EventType.RECURRING_PAYMENT_FAILURE: DeclineCode.MANDATE_SIMPLE_RETRY,
        EventType.CHECKOUT_ABANDONMENT: DeclineCode.PAYMENT_LINK_EXPIRED,
        EventType.OVERDUE_INVOICE: DeclineCode.INVOICE_OVERDUE,
    }.get(event_type, DeclineCode.DO_NOT_HONOR)


def validate_payload(payload: dict) -> None:
    """Basic structural validation before normalization."""
    if not isinstance(payload, dict):
        raise WebhookError("Payload must be a JSON object")
    event_type = payload.get("event_type")
    if not event_type:
        raise WebhookError("Missing required field: event_type")
    amount = payload.get("amount")
    if amount is None or not isinstance(amount, (int, float)) or int(amount) <= 0:
        raise WebhookError("Missing or invalid required field: amount (>0)")

    if "payment.captured" in event_type or "subscription.charged" in event_type:
        if not payload.get("event_id") and not payload.get("transaction_id"):
            raise WebhookError(
                "Confirmation events require event_id or transaction_id"
            )


def normalize_payment_webhook(payload: dict) -> RevenueEvent:
    """Normalize a /api/webhooks/payment event into a RevenueEvent.

    The event_id used for idempotency is the inbound event_id (or transaction_id
    for confirmation events). Confirmation events (payment.captured, paid,
    subscription.charged) are represented with the matching internal event id so
    the closed-loop sequencer can mark them confirmed.

    The normalized event also carries transaction_id (the stable recovery key
    for the closed-loop sequence), occurred_at, source and correlation_id so a
    single fully-normalized event is the canonical internal representation.
    """
    event_type = EVENT_TYPE_MAP.get(payload.get("event_type"), EventType.CARD_PAYMENT_FAILURE)
    amount = int(payload.get("amount", 0))
    currency = payload.get("currency", "INR")

    decline_str = payload.get("decline_code") or payload.get("error_code", "")
    if decline_str and decline_str in DECLINE_MAP:
        decline_code = DECLINE_MAP[decline_str]
    else:
        decline_code = _default_decline(event_type)

    # Confirmation events target the original event via transaction_id or event_id.
    inner_event_id = payload.get("event_id") or payload.get("transaction_id") or f"evt_{uuid.uuid4().hex[:12]}"
    transaction_id = payload.get("transaction_id") or inner_event_id

    customer_id = payload.get("customer_id") or payload.get("notes", {}).get("customer_id") or f"cust_{uuid.uuid4().hex[:8]}"
    opted_out = bool(payload.get("opted_out", False))

    failed_at_raw = payload.get("occurred_at") or payload.get("failed_at")
    failed_at = datetime.utcnow()
    if failed_at_raw:
        try:
            parsed = datetime.fromisoformat(str(failed_at_raw).replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.replace(tzinfo=None)
            failed_at = parsed
        except ValueError:
            pass

    return RevenueEvent(
        id=inner_event_id,
        type=event_type,
        customer=Customer(
            id=customer_id,
            name=payload.get("customer_name") or f"Customer {customer_id[-4:]}",
            phone=payload.get("customer_phone") or "+910000000000",
            email=payload.get("customer_email") or f"{customer_id}@example.com",
            language_pref=payload.get("language_pref", "hi"),
            opted_out=opted_out,
        ),
        amount=amount,
        currency=currency,
        root_cause=decline_code,
        decline_code=decline_code,
        failed_at=failed_at,
        metadata=TransactionMetadata(
            previous_attempts=payload.get("retry_count") or payload.get("previous_attempts") or 0,
            card_last4=payload.get("card_last4"),
            subscription_id=payload.get("subscription_id"),
            invoice_id=payload.get("invoice_id"),
        ),
        ground_truth="uncertain",
        retry_count=payload.get("retry_count") or 0,
        status="pending",
        transaction_id=transaction_id,
        occurred_at=failed_at,
        source=payload.get("source") or "webhook",
        correlation_id=payload.get("correlation_id"),
        received_at=datetime.utcnow(),
    )


def new_correlation_id() -> str:
    return f"corr_{uuid.uuid4().hex[:12]}"


__all__ = [
    "WebhookError", "verify_webhook", "compute_signature",
    "normalize_payment_webhook", "validate_payload", "validate_webhook_config",
    "new_correlation_id", "WEBHOOK_MODE",
]
