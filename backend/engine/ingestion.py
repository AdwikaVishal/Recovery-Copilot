"""
Real-time event ingestion layer.

Normalizes inbound events (webhook payloads, simulator events) into
internal RevenueEvent format and processes them through the same
multi-agent pipeline used by batch mode.
"""
import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime
from typing import Optional

from app.models import (
    RevenueEvent, Customer, TransactionMetadata,
    EventType, DeclineCode,
)
from app.database import db_session


# ---------------------------------------------------------------------------
# SSE broadcaster – keeps track of connected clients
# ---------------------------------------------------------------------------

class EventBroadcaster:
    """Maintains a set of SSE queues and broadcasts events to all of them."""

    def __init__(self):
        self._queues: list = []

    def subscribe(self) -> list:
        q: list = []
        self._queues.append(q)
        return q

    def unsubscribe(self, q: list):
        if q in self._queues:
            self._queues.remove(q)

    def broadcast(self, data: dict):
        dead = []
        for q in self._queues:
            try:
                q.append(data)
            except Exception:
                dead.append(q)
        for q in dead:
            self._queues.remove(q)


broadcaster = EventBroadcaster()


# ---------------------------------------------------------------------------
# Webhook signature verification
# ---------------------------------------------------------------------------

def verify_razorpay_signature(body: bytes, signature: str, secret: str) -> bool:
    """Verify Razorpay webhook HMAC-SHA256 signature."""
    if not secret:
        return False
    expected = hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------

def normalize_razorpay_webhook(payload: dict) -> Optional[RevenueEvent]:
    """
    Normalize a Razorpay webhook payload into a RevenueEvent.

    Handles payment.failed, payment.authorized, payment.captured,
    and subscription/mandate related events.
    """
    event_type_raw = payload.get("event", "")

    entity = {}
    if "payload" in payload:
        payment_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
        subscription_entity = payload.get("payload", {}).get("subscription", {}).get("entity", {})
        entity = payment_entity or subscription_entity

    if not entity:
        return None

    notes = entity.get("notes", {})
    event_id = notes.get("event_id") or entity.get("notes", {}).get("event_id")
    customer_id = notes.get("customer_id") or entity.get("notes", {}).get("customer_id") or f"cust_{entity.get('customer_id', 'unknown')}"
    customer_name = notes.get("customer_name", "Unknown Customer")
    customer_phone = notes.get("customer_phone", "+910000000000")
    customer_email = notes.get("customer_email", "unknown@example.com")

    amount = entity.get("amount", 0)
    currency = entity.get("currency", "INR")

    if event_type_raw == "payment.failed":
        failure_reason = entity.get("error_description", "Payment failed")
        decline_code = _map_razorpay_error(entity.get("error_code", ""))
        event_type = EventType.CARD_PAYMENT_FAILURE
    elif event_type_raw in ("subscription.authenticated", "mandate.reauthorized"):
        event_type = EventType.RECURRING_PAYMENT_FAILURE
        decline_code = DeclineCode.MANDATE_SIMPLE_RETRY
    elif event_type_raw == "payment.captured":
        decline_code = DeclineCode.INSUFFICIENT_FUNDS
        event_type = EventType.CARD_PAYMENT_FAILURE
    else:
        event_type = EventType.CARD_PAYMENT_FAILURE
        decline_code = DeclineCode.DO_NOT_HONOR

    if not event_id:
        event_id = f"evt_{entity.get('id', uuid.uuid4().hex[:12])}"

    return RevenueEvent(
        id=event_id,
        type=event_type,
        customer=Customer(
            id=customer_id,
            name=customer_name,
            phone=customer_phone,
            email=customer_email,
            language_pref=notes.get("language_pref", "hi"),
            opted_out=notes.get("opted_out", "false") == "true",
        ),
        amount=amount,
        currency=currency,
        root_cause=decline_code,
        decline_code=decline_code,
        failed_at=datetime.utcnow(),
        metadata=TransactionMetadata(
            card_last4=entity.get("card", {}).get("last4"),
            bank=entity.get("card", {}).get("issuer"),
            mandate_id=entity.get("mandate_id"),
            subscription_id=entity.get("subscription_id"),
        ),
        ground_truth="uncertain",
        status="pending",
    )


def normalize_simulator_event(payload: dict) -> RevenueEvent:
    """
    Normalize a simulator/test event into a RevenueEvent.

    Expects a simplified JSON payload:
    {
        "event_type": "payment.failed" | "recurring_payment_failure" | ...,
        "amount": 50000,
        "currency": "INR",
        "customer_id": "optional",
        "decline_code": "optional"
    }
    """
    raw_type = payload.get("event_type", "payment.failed")
    amount = payload.get("amount", 50000)
    currency = payload.get("currency", "INR")
    decline_code_str = payload.get("decline_code", "")
    customer_id = payload.get("customer_id", "")

    event_type, decline_code = _classify_event(raw_type, decline_code_str, amount)

    if not customer_id:
        customer_id = f"cust_sim_{uuid.uuid4().hex[:8]}"

    # An explicit event_id makes deterministic replay (e.g. the demo narrative's
    # duplicate-webhook leg) possible; otherwise a fresh id is minted.
    raw_id = payload.get("event_id", "")

    return RevenueEvent(
        id=(f"evt_sim_{uuid.uuid4().hex[:12]}" if not raw_id else raw_id),
        type=event_type,
        customer=Customer(
            id=customer_id,
            name=f"Sim Customer {customer_id[-4:]}",
            phone=f"+91{9000000000 + hash(customer_id) % 1000000000}",
            email=f"{customer_id}@sim.example.com",
            language_pref=payload.get("language_pref", "hi"),
            opted_out=payload.get("opted_out", False),
        ),
        amount=amount,
        currency=currency,
        root_cause=decline_code,
        decline_code=decline_code,
        failed_at=datetime.utcnow(),
        metadata=TransactionMetadata(
            previous_attempts=payload.get("retry_count", 0),
        ),
        ground_truth="uncertain",
        retry_count=payload.get("retry_count", 0),
        status="pending",
    )


# ---------------------------------------------------------------------------
# Core event processing (unified path for batch + real-time)
# ---------------------------------------------------------------------------

async def process_single_event(event: RevenueEvent, source: str = "unknown") -> dict:
    """
    Process a single event through the full multi-agent pipeline.

    This is the unified entry point used by:
    - Batch mode (iterates over events)
    - Webhook handler (receives one event at a time)
    - Simulator (sends one event at a time)

    Returns a summary dict for the SSE broadcast.
    """
    from engine.pipeline import _map_workflow_status
    from agents.supervisor import process_event as supervisor_process
    from agents.execution_adapter import execute

    # Persist the inbound event
    async with db_session() as db:
        await db.execute(
            """INSERT OR IGNORE INTO revenue_events
               (id, type, customer_id, customer_name, customer_phone, customer_email,
                language_pref, opted_out, amount, currency, root_cause, decline_code,
                failed_at, metadata_json, ground_truth, recovered_amount, retry_count, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.id, event.type.value, event.customer.id, event.customer.name,
                event.customer.phone, event.customer.email, event.customer.language_pref,
                1 if event.customer.opted_out else 0, event.amount, event.currency,
                event.root_cause.value, event.decline_code.value, event.failed_at.isoformat(),
                event.metadata.model_dump_json(), event.ground_truth, event.recovered_amount,
                event.retry_count, event.status,
            ),
        )

    # Run through supervisor (which calls all agents + policy + execution)
    try:
        supervisor_output = await supervisor_process(event)
    except Exception as exc:
        return {
            "type": "event.error",
            "event_id": event.id,
            "error": str(exc),
            "source": source,
        }

    # Determine final execution status
    exec_status = "failed"
    if supervisor_output.workflow_status.value == "PENDING_WEBHOOK":
        exec_status = "pending"
    elif supervisor_output.workflow_status.value == "RESOLVED":
        exec_status = "success"
    elif supervisor_output.workflow_status.value == "HUMAN_REVIEW":
        exec_status = "pending"
    elif supervisor_output.workflow_status.value == "STOPPED":
        exec_status = "blocked"

    # Build the broadcast payload
    policy_verdict = "UNKNOWN"
    action = "none"
    amount_recovered = 0
    if supervisor_output.proposed_action:
        action = supervisor_output.proposed_action.action
    if supervisor_output.compliance_explanation:
        policy_verdict = supervisor_output.compliance_explanation.verdict

    broadcast_data = {
        "type": "transaction.updated",
        "event_id": event.id,
        "customer_id": event.customer.id,
        "customer_name": event.customer.name,
        "amount": event.amount,
        "currency": event.currency,
        "event_type": event.type.value,
        "status": exec_status,
        "action": action,
        "policy_verdict": policy_verdict,
        "workflow_status": supervisor_output.workflow_status.value,
        "reason": supervisor_output.next_step,
        "channel": supervisor_output.proposed_action.channel if supervisor_output.proposed_action else "none",
        "amount_recovered": amount_recovered,
        "source": source,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Broadcast to SSE clients
    broadcaster.broadcast(broadcast_data)

    return broadcast_data


# ---------------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------------

async def is_duplicate_event(event_id: str) -> bool:
    """Check if an event with this ID has already been processed."""
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT id FROM revenue_events WHERE id = ?",
            (event_id,),
        )
        row = await cursor.fetchone()
        return row is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_razorpay_error(error_code: str) -> DeclineCode:
    mapping = {
        "bad_expired_card": DeclineCode.EXPIRED_CARD,
        "insufficient_funds": DeclineCode.INSUFFICIENT_FUNDS,
        "do_not_honor": DeclineCode.DO_NOT_HONOR,
        "bank_timeout": DeclineCode.BANK_TIMEOUT,
        "incorrect_cvc": DeclineCode.INCORRECT_CVC,
        "processing_error": DeclineCode.PROCESSING_ERROR,
    }
    return mapping.get(error_code, DeclineCode.DO_NOT_HONOR)


def _classify_event(raw_type: str, decline_code_str: str, amount: int) -> tuple[EventType, DeclineCode]:
    """Classify a simulator event into EventType + DeclineCode."""
    type_map = {
        "payment.failed": EventType.CARD_PAYMENT_FAILURE,
        "card_payment_failure": EventType.CARD_PAYMENT_FAILURE,
        "recurring_payment_failure": EventType.RECURRING_PAYMENT_FAILURE,
        "subscription.failed": EventType.RECURRING_PAYMENT_FAILURE,
        "checkout_abandonment": EventType.CHECKOUT_ABANDONMENT,
        "overdue_invoice": EventType.OVERDUE_INVOICE,
    }
    event_type = type_map.get(raw_type, EventType.CARD_PAYMENT_FAILURE)

    decline_map = {
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

    if decline_code_str and decline_code_str in decline_map:
        decline_code = decline_map[decline_code_str]
    else:
        default_map = {
            EventType.CARD_PAYMENT_FAILURE: DeclineCode.INSUFFICIENT_FUNDS,
            EventType.RECURRING_PAYMENT_FAILURE: DeclineCode.MANDATE_SIMPLE_RETRY,
            EventType.CHECKOUT_ABANDONMENT: DeclineCode.PAYMENT_LINK_EXPIRED,
            EventType.OVERDUE_INVOICE: DeclineCode.INVOICE_OVERDUE,
        }
        decline_code = default_map.get(event_type, DeclineCode.DO_NOT_HONOR)

    return event_type, decline_code
