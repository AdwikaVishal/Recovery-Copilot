from app.models import (
    RevenueEvent, DiagnosisOutput, CustomerContextOutput,
    RecoveryStrategyOutput, ProposedAction
)
from app.config import PolicyConfig


CONFIG = PolicyConfig()
MAX_DISCOUNT = CONFIG.max_discount_percent
RBI_AFA_THRESHOLD = CONFIG.rbi_afa_threshold_paise


def build_strategy(
    event: RevenueEvent,
    diagnosis: DiagnosisOutput,
    context: CustomerContextOutput,
) -> RecoveryStrategyOutput:
    action_family = diagnosis.recommended_action_family

    # ── Hard rule: never engage an opted-out / DND / blocked customer ──
    # Every family — including manual escalation, reauthorization links and
    # fresh payment links — requires customer engagement. When the customer is
    # not safe to contact the transaction is blocked outright; the policy
    # engine still DENYs it so the verdict stays visible in the audit trace.
    if not context.safe_to_contact:
        return RecoveryStrategyOutput(
            strategy="STOP",
            priority="CRITICAL",
            reason=f"Customer not safe to contact (opted out / DND / blocked): {context.risk_flags}",
            proposed_delay_hours=0,
            channel="NONE",
            discount_percent=0,
            requires_human_approval=False,
            expected_value=0,
        )

    # ── Hard rule: never blind retry when AFA required ──
    if action_family == "RETRY" and diagnosis.requires_afa:
        return RecoveryStrategyOutput(
            strategy="REAUTHORIZE",
            priority="CRITICAL",
            reason="RBI: recurring > ₹15,000 requires fresh AFA. Blind retry blocked.",
            proposed_delay_hours=0,
            channel="WHATSAPP",
            discount_percent=0,
            requires_human_approval=False,
            expected_value=event.amount,
        )

    # ── RETRY ──
    if action_family == "RETRY":
        delay = diagnosis.recommended_wait_hours
        discount = event.metadata.discount_hint or 0
        ev = _estimate_value(event, diagnosis, 0.70 if event.retry_count == 0 else 0.45)
        reason = f"Diagnosis: {diagnosis.classification} (confidence {diagnosis.confidence}). Wait {delay}h before retry."
        if discount > 0:
            reason += f" Merchant-initiated incentive: {discount}% discount."
        return RecoveryStrategyOutput(
            strategy="RETRY",
            priority=_priority(event, diagnosis),
            reason=reason,
            proposed_delay_hours=delay,
            channel="RAZORPAY_API",
            discount_percent=discount,
            requires_human_approval=False,
            expected_value=ev,
        )

    # ── REAUTHORIZE ──
    if action_family == "REAUTHORIZE":
        return RecoveryStrategyOutput(
            strategy="REAUTHORIZE",
            priority="HIGH",
            reason=f"RBI AFA required for ₹{event.amount // 100:,}. Fresh authorization link via {context.preferred_channel}.",
            proposed_delay_hours=0,
            channel="WHATSAPP",
            discount_percent=0,
            requires_human_approval=False,
            expected_value=event.amount,
        )

    # ── UPDATE_PAYMENT_METHOD → PAYMENT_LINK ──
    if action_family == "UPDATE_PAYMENT_METHOD":
        ch = _map_channel(context.preferred_channel)
        ev = _estimate_value(event, diagnosis, 0.35)
        return RecoveryStrategyOutput(
            strategy="PAYMENT_LINK",
            priority=_priority(event, diagnosis),
            reason=f"Payment method invalid ({diagnosis.classification}). Fresh link sent via {ch}.",
            proposed_delay_hours=0,
            channel=ch,
            discount_percent=0,
            requires_human_approval=False,
            expected_value=ev,
        )

    # ── SEND_REMINDER / MESSAGE ──
    if action_family in ("SEND_REMINDER", "MESSAGE"):
        if not context.safe_to_contact:
            return RecoveryStrategyOutput(
                strategy="STOP",
                priority="CRITICAL",
                reason=f"Not safe to contact: {context.risk_flags}",
                proposed_delay_hours=0,
                channel="NONE",
                discount_percent=0,
                requires_human_approval=False,
                expected_value=0,
            )

        days = event.metadata.days_overdue or 0
        tone = _dunning_tone(days)
        ch = _map_channel(context.preferred_channel)
        discount = event.metadata.discount_hint or 0

        ev = _estimate_value(event, diagnosis, 0.50 if tone <= 2 else 0.25)

        reason = f"Dunning tone {tone} for {days} days overdue. Recoverability: {diagnosis.likely_recoverability}."
        if discount > 0:
            reason += f" Merchant-initiated incentive: {discount}% discount."

        return RecoveryStrategyOutput(
            strategy="MESSAGE",
            priority=_priority(event, diagnosis),
            reason=reason,
            proposed_delay_hours=0,
            channel=ch,
            discount_percent=discount,
            requires_human_approval=False,
            expected_value=ev,
        )

    # ── PTP_FOLLOWUP ──
    if action_family == "PTP_FOLLOWUP":
        return RecoveryStrategyOutput(
            strategy="PTP_FOLLOWUP",
            priority="MEDIUM",
            reason=f"Promise-to-pay tracking for ₹{event.amount // 100:,}. Follow up on promised date.",
            proposed_delay_hours=24,
            channel=_map_channel(context.preferred_channel),
            discount_percent=0,
            requires_human_approval=False,
            expected_value=event.amount,
        )

    # ── STOP ──
    if action_family == "STOP":
        return RecoveryStrategyOutput(
            strategy="STOP",
            priority="LOW",
            reason="Diagnosis recommends stopping. Customer opted out or ineligible.",
            proposed_delay_hours=0,
            channel="NONE",
            discount_percent=0,
            requires_human_approval=False,
            expected_value=0,
        )

    # ── HUMAN_REVIEW ──
    return RecoveryStrategyOutput(
        strategy="HUMAN_REVIEW",
        priority="HIGH",
        reason=f"Action family '{action_family}' — requires manual review. Confidence: {diagnosis.confidence}.",
        proposed_delay_hours=0,
        channel="NONE",
        discount_percent=0,
        requires_human_approval=True,
        expected_value=event.amount,
    )


def strategy_to_proposed_action(
    event: RevenueEvent,
    strategy: RecoveryStrategyOutput,
) -> ProposedAction:
    action_map = {
        "RETRY": "retry_payment",
        "REAUTHORIZE": "re_authorize_mandate",
        "PAYMENT_LINK": "send_payment_link",
        "MESSAGE": "send_dunning_message",
        "PTP_FOLLOWUP": "send_dunning_message",
        "HUMAN_REVIEW": "escalate_to_human",
        "STOP": "blocked",
    }

    channel_map = {
        "RAZORPAY_API": "razorpay_api",
        "WHATSAPP": "whatsapp",
        "SMS": "sms",
        "EMAIL": "email",
        "NONE": "none",
    }

    action = action_map.get(strategy.strategy, "escalate_to_human")
    channel = channel_map.get(strategy.channel, "none")

    tone = 1
    if action == "send_dunning_message":
        days = event.metadata.days_overdue or 0
        tone = _dunning_tone(days)

    return ProposedAction(
        action=action,
        channel=channel,
        amount=event.amount,
        discount_percent=strategy.discount_percent,
        message_tone_level=tone,
        reason=strategy.reason,
    )


def _dunning_tone(days_overdue: int) -> int:
    if days_overdue <= 1:
        return 1
    elif days_overdue <= 3:
        return 2
    elif days_overdue <= 7:
        return 3
    else:
        return 4


def _map_channel(preferred: str) -> str:
    return {
        "WHATSAPP": "WHATSAPP",
        "SMS": "SMS",
        "EMAIL": "EMAIL",
    }.get(preferred, "WHATSAPP")


def _priority(event: RevenueEvent, diagnosis: DiagnosisOutput) -> str:
    if event.amount > 500000:
        return "HIGH"
    if diagnosis.likely_recoverability == "HIGH":
        return "HIGH"
    if diagnosis.likely_recoverability == "LOW":
        return "LOW"
    return "MEDIUM"


def _estimate_value(event: RevenueEvent, diagnosis: DiagnosisOutput, base_prob: float) -> int:
    conf_factor = diagnosis.confidence
    amount = event.amount
    return int(amount * base_prob * conf_factor)
