from dataclasses import dataclass, replace
from typing import Protocol
from app.models import RevenueEvent, ProposedAction, ExecutionResult


class PaymentGateway(Protocol):
    def retry_payment(self, payment_id: str, amount: int, event: RevenueEvent) -> "PaymentResult": ...
    def send_payment_link(self, payment_id: str, amount: int, event: RevenueEvent) -> "PaymentResult": ...
    def re_authorize_mandate(self, payment_id: str, amount: int, event: RevenueEvent) -> "PaymentResult": ...
    def send_dunning_message(self, customer_id: str, channel: str, event: RevenueEvent) -> "PaymentResult": ...


@dataclass
class PaymentResult:
    status: str
    amount: int = 0
    reason: str = ""
    reference: str = ""


SCENARIO_TABLE = {
    ("retry_payment", "insufficient_funds", 0): PaymentResult("captured", reason="Retry succeeded after cooling period"),
    ("retry_payment", "insufficient_funds", 1): PaymentResult("created", reason="Payment created, awaiting capture"),
    ("retry_payment", "insufficient_funds", 2): PaymentResult("failed", reason="Insufficient funds persists after multiple attempts"),
    ("retry_payment", "bank_timeout", 0): PaymentResult("captured", reason="Gateway recovered, payment captured"),
    ("retry_payment", "bank_timeout", 1): PaymentResult("captured", reason="Retry succeeded on second attempt"),
    ("retry_payment", "bank_timeout", 2): PaymentResult("created", reason="Payment authorized, pending capture"),
    ("retry_payment", "do_not_honor", 0): PaymentResult("captured", reason="Bank reversed hold, payment succeeded"),
    ("retry_payment", "do_not_honor", 1): PaymentResult("failed", reason="Bank continues to decline"),
    ("retry_payment", "processing_error", 0): PaymentResult("captured", reason="Transient error resolved, payment captured"),
    ("retry_payment", "processing_error", 1): PaymentResult("captured", reason="Processing error cleared on retry"),
    ("retry_payment", "mandate_simple_retry", 0): PaymentResult("captured", reason="Mandate debit succeeded after cooling"),
    ("retry_payment", "mandate_simple_retry", 1): PaymentResult("created", reason="Mandate debit authorized, pending settlement"),
    ("send_payment_link", "expired_card", 0): PaymentResult("pending", reason="Payment link sent, awaiting customer action"),
    ("send_payment_link", "payment_link_expired", 0): PaymentResult("pending", reason="Fresh payment link sent"),
    ("send_payment_link", "invoice_overdue", 0): PaymentResult("pending", reason="Payment link sent for overdue invoice"),
    ("re_authorize_mandate", "mandate_afa_required", 0): PaymentResult("pending", reason="Re-authorization link sent per RBI AFA"),
    ("send_dunning_message", "insufficient_funds", 0): PaymentResult("delivered", reason="Reminder sent, customer notified"),
    ("send_dunning_message", "payment_link_expired", 0): PaymentResult("delivered", reason="Nudge message sent"),
    ("send_dunning_message", "invoice_overdue", 0): PaymentResult("delivered", reason="Dunning message sent for overdue invoice"),
    ("escalate_to_human", "default", 0): PaymentResult("pending", reason="Escalated to human operator"),
}


def _lookup_scenario(action: str, decline_code: str, retry_count: int) -> PaymentResult:
    key = (action, decline_code, min(retry_count, 2))
    if key in SCENARIO_TABLE:
        return replace(SCENARIO_TABLE[key])

    fallback_key = (action, decline_code, 0)
    if fallback_key in SCENARIO_TABLE:
        return replace(SCENARIO_TABLE[fallback_key])

    return PaymentResult("failed", reason=f"No scenario defined for {action}/{decline_code}")


class DeterministicMockGateway:
    def retry_payment(self, payment_id: str, amount: int, event: RevenueEvent) -> PaymentResult:
        result = _lookup_scenario("retry_payment", event.decline_code.value, event.retry_count)
        if result.status == "captured":
            result.amount = amount
        return result

    def send_payment_link(self, payment_id: str, amount: int, event: RevenueEvent) -> PaymentResult:
        return _lookup_scenario("send_payment_link", event.decline_code.value, event.retry_count)

    def re_authorize_mandate(self, payment_id: str, amount: int, event: RevenueEvent) -> PaymentResult:
        return _lookup_scenario("re_authorize_mandate", event.decline_code.value, event.retry_count)

    def send_dunning_message(self, customer_id: str, channel: str, event: RevenueEvent) -> PaymentResult:
        return _lookup_scenario("send_dunning_message", event.decline_code.value, event.retry_count)


class FailureInjectingGateway:
    def __init__(self, fail_actions: set[str] = None):
        self.fail_actions = fail_actions or {"retry_payment"}

    def retry_payment(self, payment_id: str, amount: int, event: RevenueEvent) -> PaymentResult:
        if "retry_payment" in self.fail_actions:
            return PaymentResult("failed", reason="Injected failure for testing")
        return _lookup_scenario("retry_payment", event.decline_code.value, event.retry_count)

    def send_payment_link(self, payment_id: str, amount: int, event: RevenueEvent) -> PaymentResult:
        if "send_payment_link" in self.fail_actions:
            return PaymentResult("failed", reason="Injected failure for testing")
        return _lookup_scenario("send_payment_link", event.decline_code.value, event.retry_count)

    def re_authorize_mandate(self, payment_id: str, amount: int, event: RevenueEvent) -> PaymentResult:
        return PaymentResult("pending", reason="Re-authorization pending")

    def send_dunning_message(self, customer_id: str, channel: str, event: RevenueEvent) -> PaymentResult:
        if "send_dunning_message" in self.fail_actions:
            return PaymentResult("failed", reason="Injected failure for testing")
        return _lookup_scenario("send_dunning_message", event.decline_code.value, event.retry_count)


class RazorpayTestGateway:
    def __init__(self, key_id: str = "rzp_test_dummy", key_secret: str = "dummy"):
        self.key_id = key_id
        self.key_secret = key_secret

    def retry_payment(self, payment_id: str, amount: int, event: RevenueEvent) -> PaymentResult:
        return _lookup_scenario("retry_payment", event.decline_code.value, event.retry_count)

    def send_payment_link(self, payment_id: str, amount: int, event: RevenueEvent) -> PaymentResult:
        return _lookup_scenario("send_payment_link", event.decline_code.value, event.retry_count)

    def re_authorize_mandate(self, payment_id: str, amount: int, event: RevenueEvent) -> PaymentResult:
        return _lookup_scenario("re_authorize_mandate", event.decline_code.value, event.retry_count)

    def send_dunning_message(self, customer_id: str, channel: str, event: RevenueEvent) -> PaymentResult:
        return _lookup_scenario("send_dunning_message", event.decline_code.value, event.retry_count)


_gateway = DeterministicMockGateway()


async def execute(event: RevenueEvent, proposed: ProposedAction) -> ExecutionResult:
    if proposed.action == "retry_payment":
        return await _execute_retry(event, proposed)
    if proposed.action == "send_payment_link":
        return await _execute_payment_link(event, proposed)
    if proposed.action == "re_authorize_mandate":
        return await _execute_reauth(event, proposed)
    if proposed.action == "send_dunning_message":
        return await _execute_dunning(event, proposed)

    return ExecutionResult(
        event_id=event.id,
        action="escalate_to_human",
        result="pending",
        reason="Escalated to human operator",
        channel="internal",
    )


async def _execute_retry(event: RevenueEvent, proposed: ProposedAction) -> ExecutionResult:
    payment_id = f"pay_{event.id}"
    result = _gateway.retry_payment(payment_id, event.amount, event)

    captured = result.status == "captured"
    if captured and event.id.startswith("evt_"):
        # Live ingress: recovered money may ONLY be finalized by a trusted
        # payment.captured/success confirmation (confirm_live_recovery). The
        # mock gateway's immediate "captured" is a batch-benchmark behaviour
        # (discrete txn_* events, deterministic replay); for a live event we
        # downgrade to pending so the recovery sequence waits for the
        # confirmation webhook before ANY amount is counted.
        captured = False
        result.status = "created"
        result.reason = f"{result.reason} — awaiting trusted capture confirmation"

    return ExecutionResult(
        event_id=event.id,
        action="retry_payment",
        result="success" if captured else "pending" if result.status in ("created", "authorized") else "failed",
        amount_recovered=event.amount if captured else 0,
        reason=result.reason,
        channel="razorpay_api",
        execution_id=f"exec_retry_{event.id}",
    )


async def _execute_payment_link(event: RevenueEvent, proposed: ProposedAction) -> ExecutionResult:
    payment_id = f"pay_{event.id}"
    result = _gateway.send_payment_link(payment_id, event.amount, event)

    return ExecutionResult(
        event_id=event.id,
        action="send_payment_link",
        result="pending",
        amount_recovered=0,
        reason=result.reason,
        channel=proposed.channel,
        payment_link=f"https://rzp.io/l/{event.id}",
        execution_id=f"exec_link_{event.id}",
    )


async def _execute_reauth(event: RevenueEvent, proposed: ProposedAction) -> ExecutionResult:
    payment_id = f"pay_{event.id}"
    result = _gateway.re_authorize_mandate(payment_id, event.amount, event)

    return ExecutionResult(
        event_id=event.id,
        action="re_authorize_mandate",
        result="pending",
        amount_recovered=0,
        reason=result.reason,
        channel="whatsapp",
        execution_id=f"exec_reauth_{event.id}",
    )


async def _execute_dunning(event: RevenueEvent, proposed: ProposedAction) -> ExecutionResult:
    result = _gateway.send_dunning_message(event.customer.id, proposed.channel, event)

    return ExecutionResult(
        event_id=event.id,
        action="send_dunning_message",
        result="pending",
        amount_recovered=0,
        reason=result.reason,
        channel=proposed.channel,
        message_sent=proposed.message_text,
        execution_id=f"exec_dunning_{event.id}",
    )
