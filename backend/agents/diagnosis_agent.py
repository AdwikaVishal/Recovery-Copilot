from app.models import DiagnosisOutput, EvidenceItem


DECLINE_KNOWLEDGE = {
    "insufficient_funds": {
        "base_classification": "temporary_cash_flow_issue",
        "base_confidence": 0.82,
        "base_recoverability": "MEDIUM",
        "base_action": "RETRY",
        "base_wait_hours": 48,
        "rationale": "Insufficient funds typically indicates a temporary balance shortfall, often resolved by salary credit cycles.",
    },
    "expired_card": {
        "base_classification": "expired_payment_instrument",
        "base_confidence": 0.95,
        "base_recoverability": "LOW",
        "base_action": "UPDATE_PAYMENT_METHOD",
        "base_wait_hours": 0,
        "rationale": "An expired card cannot process any transaction. Retry without card update will fail.",
    },
    "do_not_honor": {
        "base_classification": "unspecified_bank_rejection",
        "base_confidence": 0.55,
        "base_recoverability": "MEDIUM",
        "base_action": "RETRY",
        "base_wait_hours": 24,
        "rationale": "Bank declined without disclosing reason. May be transient (anti-fraud hold, velocity check) or permanent (account frozen).",
    },
    "bank_timeout": {
        "base_classification": "gateway_network_failure",
        "base_confidence": 0.88,
        "base_recoverability": "HIGH",
        "base_action": "RETRY",
        "base_wait_hours": 2,
        "rationale": "Bank gateway did not respond in time. This is an infrastructure failure, not a card/account issue.",
    },
    "incorrect_cvc": {
        "base_classification": "authentication_mismatch",
        "base_confidence": 0.90,
        "base_recoverability": "LOW",
        "base_action": "UPDATE_PAYMENT_METHOD",
        "base_wait_hours": 0,
        "rationale": "CVC validation failed. Customer likely entered incorrect code. Retry with same details will fail again.",
    },
    "processing_error": {
        "base_classification": "bank_processing_fault",
        "base_confidence": 0.78,
        "base_recoverability": "MEDIUM",
        "base_action": "RETRY",
        "base_wait_hours": 4,
        "rationale": "Bank-side processing error. Often transient and resolves within hours.",
    },
    "mandate_afa_required": {
        "base_classification": "rbi_afa_compliance_block",
        "base_confidence": 0.97,
        "base_recoverability": "HIGH",
        "base_action": "REAUTHORIZE",
        "base_wait_hours": 0,
        "rationale": "RBI mandates Additional Factor Authentication for recurring debits above ₹15,000. The mandate itself is valid but needs fresh customer authorization.",
    },
    "mandate_simple_retry": {
        "base_classification": "mandate_transient_failure",
        "base_confidence": 0.75,
        "base_recoverability": "MEDIUM",
        "base_action": "RETRY",
        "base_wait_hours": 24,
        "rationale": "Mandate debit failed below the ₹15,000 AFA threshold. No additional authorization needed; retry after bank settles.",
    },
    "payment_link_expired": {
        "base_classification": "checkout_abandonment",
        "base_confidence": 0.85,
        "base_recoverability": "MEDIUM",
        "base_action": "SEND_REMINDER",
        "base_wait_hours": 0,
        "rationale": "Payment link expired without completion. Customer showed intent but dropped off. Fresh link with nudge may recover.",
    },
    "invoice_overdue": {
        "base_classification": "overdue_receivable",
        "base_confidence": 0.70,
        "base_recoverability": "MEDIUM",
        "base_action": "SEND_REMINDER",
        "base_wait_hours": 0,
        "rationale": "B2B invoice past due date. Recoverability depends on customer relationship and outstanding disputes.",
    },
}


def diagnose(event) -> DiagnosisOutput:
    decline_code = event.decline_code.value if hasattr(event.decline_code, 'value') else str(event.decline_code)
    evidence = []
    uncertainties = []
    confidence_penalty = 0.0

    # ── Evidence: decline code ──
    knowledge = DECLINE_KNOWLEDGE.get(decline_code)
    if knowledge is None:
        evidence.append(EvidenceItem(
            field="decline_code",
            value=decline_code,
            interpretation=f"Unknown decline code '{decline_code}'. No diagnosis rule available.",
        ))
        return DiagnosisOutput(
            classification="unknown_decline_code",
            confidence=0.15,
            evidence=evidence,
            likely_recoverability="UNKNOWN",
            recommended_wait_hours=0,
            recommended_action_family="HUMAN_REVIEW",
            uncertainties=[f"Decline code '{decline_code}' is not in the known codebase"],
        )

    evidence.append(EvidenceItem(
        field="decline_code",
        value=decline_code,
        interpretation=knowledge["rationale"],
    ))

    # ── Evidence: payment type ──
    event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
    evidence.append(EvidenceItem(
        field="payment_type",
        value=event_type,
        interpretation=f"Event classified as {event_type}.",
    ))

    # ── Evidence: amount ──
    amount_paise = event.amount
    amount_display = f"₹{amount_paise // 100:,}"
    evidence.append(EvidenceItem(
        field="amount",
        value=amount_display,
        interpretation=f"Transaction value {amount_display}.",
    ))

    if decline_code in ("mandate_afa_required", "mandate_simple_retry"):
        afa_threshold = 1500000
        if amount_paise > afa_threshold:
            evidence.append(EvidenceItem(
                field="afa_threshold_check",
                value=f"{amount_display} > ₹15,000",
                interpretation="Amount exceeds RBI AFA threshold for recurring payments.",
            ))
        else:
            evidence.append(EvidenceItem(
                field="afa_threshold_check",
                value=f"{amount_display} <= ₹15,000",
                interpretation="Amount below AFA threshold. Fresh authorization not required.",
            ))

    # ── Evidence: attempt count ──
    retry_count = event.retry_count
    evidence.append(EvidenceItem(
        field="attempt_count",
        value=str(retry_count),
        interpretation=f"This is attempt #{retry_count + 1}.",
    ))

    if retry_count >= 2:
        confidence_penalty += 0.15
        uncertainties.append(f"High retry count ({retry_count}). Prior attempts failed, reducing confidence in recovery.")
    elif retry_count >= 1:
        confidence_penalty += 0.08

    # ── Evidence: timestamps ──
    if hasattr(event, 'failed_at') and event.failed_at:
        failed_at = event.failed_at
        if hasattr(failed_at, 'timestamp'):
            import datetime
            hours_since = (datetime.datetime.utcnow() - failed_at).total_seconds() / 3600
            evidence.append(EvidenceItem(
                field="time_since_failure",
                value=f"{hours_since:.1f} hours",
                interpretation=f"Payment failed {hours_since:.1f} hours ago.",
            ))
            if hours_since > 168:
                confidence_penalty += 0.10
                uncertainties.append("Event is over 7 days old. Customer engagement may have dropped.")
        else:
            evidence.append(EvidenceItem(
                field="time_since_failure",
                value="unknown",
                interpretation="Timestamp parsing failed.",
            ))
            confidence_penalty += 0.05
            uncertainties.append("Could not parse failure timestamp.")

    # ── Evidence: last attempt ──
    if event.last_attempt_at:
        evidence.append(EvidenceItem(
            field="last_attempt",
            value=str(event.last_attempt_at),
            interpretation="Prior attempt recorded.",
        ))
    else:
        evidence.append(EvidenceItem(
            field="last_attempt",
            value="none",
            interpretation="No prior attempt recorded.",
        ))

    # ── Evidence: customer opts / status ──
    if event.customer.opted_out:
        evidence.append(EvidenceItem(
            field="customer_opt_out",
            value="true",
            interpretation="Customer has opted out of communications.",
        ))
        return DiagnosisOutput(
            classification="customer_opted_out",
            confidence=0.99,
            evidence=evidence,
            likely_recoverability="LOW",
            recommended_wait_hours=0,
            recommended_action_family="STOP",
            uncertainties=[],
        )

    # ── Evidence: metadata-specific fields ──
    if decline_code == "invoice_overdue" and event.metadata.days_overdue is not None:
        days = event.metadata.days_overdue
        evidence.append(EvidenceItem(
            field="days_overdue",
            value=str(days),
            interpretation=f"Invoice is {days} days past due.",
        ))
        if days > 60:
            confidence_penalty += 0.15
            uncertainties.append(f"Invoice {days} days overdue. Recovery probability drops significantly after 60 days.")
        elif days > 30:
            confidence_penalty += 0.08
            uncertainties.append(f"Invoice {days} days overdue. Moderate urgency.")

    if decline_code in ("insufficient_funds", "do_not_honor") and event.metadata.bank:
        evidence.append(EvidenceItem(
            field="issuing_bank",
            value=event.metadata.bank,
            interpretation=f"Decline from {event.metadata.bank}.",
        ))

    if decline_code in ("mandate_afa_required", "mandate_simple_retry") and event.metadata.mandate_id:
        evidence.append(EvidenceItem(
            field="mandate_id",
            value=event.metadata.mandate_id,
            interpretation=f"Linked to mandate {event.metadata.mandate_id}.",
        ))

    if decline_code in ("mandate_afa_required", "mandate_simple_retry") and event.metadata.subscription_id:
        evidence.append(EvidenceItem(
            field="subscription_id",
            value=event.metadata.subscription_id,
            interpretation=f"Linked to subscription {event.metadata.subscription_id}.",
        ))

    # ── Compute final confidence ──
    final_confidence = max(0.10, knowledge["base_confidence"] - confidence_penalty)

    # ── Determine action family ──
    action = knowledge["base_action"]
    if action == "RETRY" and retry_count >= 3:
        action = "HUMAN_REVIEW"
        uncertainties.append("Retries exhausted (>=3). Escalating to human review.")

    if action == "RETRY" and decline_code == "insufficient_funds" and retry_count >= 2:
        action = "SEND_REMINDER"
        uncertainties.append("Multiple insufficient_funds failures. Switching to reminder channel.")

    if action == "RETRY" and decline_code == "do_not_honor" and retry_count >= 1:
        action = "SEND_REMINDER"
        uncertainties.append("do_not_honor persisting. Switching to reminder with payment link.")

    # ── Recoverability adjustment ──
    recoverability = knowledge["base_recoverability"]
    if final_confidence < 0.4:
        recoverability = "LOW"
    elif final_confidence > 0.8 and recoverability == "MEDIUM":
        recoverability = "HIGH"

    # ── Wait hours adjustment ──
    wait_hours = knowledge["base_wait_hours"]
    if retry_count >= 2:
        wait_hours = max(wait_hours, 48)
    if event.metadata.days_overdue and event.metadata.days_overdue > 30:
        wait_hours = 0

    requires_afa = decline_code in ("mandate_afa_required",)
    max_retries = 3
    if action == "UPDATE_PAYMENT_METHOD":
        max_retries = 0
    elif action == "RETRY":
        max_retries = 2
        if decline_code == "bank_timeout":
            max_retries = 3
    elif action == "REAUTHORIZE":
        max_retries = 0

    risk_score = 1.0 - final_confidence
    if action == "STOP":
        risk_score = 0.0

    return DiagnosisOutput(
        classification=knowledge["base_classification"],
        confidence=round(final_confidence, 2),
        evidence=evidence,
        likely_recoverability=recoverability,
        recommended_wait_hours=wait_hours,
        recommended_action_family=action,
        uncertainties=uncertainties,
        requires_afa=requires_afa,
        max_retries=max_retries,
        risk_score=round(risk_score, 2),
    )
