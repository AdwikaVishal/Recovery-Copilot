from app.models import (
    PolicyDecision, PolicyCheckDetail, ComplianceExplanation,
    RevenueEvent, DiagnosisOutput, CustomerContextOutput, ProposedAction
)


def explain(
    event: RevenueEvent,
    diagnosis: DiagnosisOutput,
    context: CustomerContextOutput,
    proposed: ProposedAction,
    policy_decision: PolicyDecision,
) -> ComplianceExplanation:
    failed_rules = [r for r in policy_decision.detailed_results if r.result == "FAIL"]
    passed_rules = [r for r in policy_decision.detailed_results if r.result == "PASS"]

    summary = _build_summary(event, policy_decision, failed_rules)

    customer_explanation = _build_customer_explanation(
        event, context, proposed, failed_rules, passed_rules, policy_decision
    )

    operator_explanation = _build_operator_explanation(
        event, diagnosis, context, proposed, policy_decision, failed_rules
    )

    rules_triggered = [
        PolicyCheckDetail(
            rule=r.rule,
            result=r.result,
            explanation=r.explanation,
        )
        for r in policy_decision.detailed_results
    ]

    return ComplianceExplanation(
        verdict=policy_decision.verdict.value,
        summary=summary,
        rules_triggered=rules_triggered,
        customer_safe_explanation=customer_explanation,
        operator_explanation=operator_explanation,
    )


def _build_summary(event: RevenueEvent, decision: PolicyDecision, failed_rules: list) -> str:
    if decision.verdict.value == "ALLOW":
        return f"All {len(decision.checks_passed)} policy checks passed. Action '{event.type.value}' for ₹{event.amount // 100:,} is authorized."

    if decision.verdict.value == "DENY":
        rule_names = ", ".join(r.rule for r in failed_rules)
        return f"Action blocked by {len(failed_rules)} policy rule(s): {rule_names}. No recovery action will be taken."

    if decision.verdict.value == "HUMAN_REVIEW":
        rule_names = ", ".join(r.rule for r in failed_rules)
        return f"Action requires human approval due to {len(failed_rules)} rule(s): {rule_names}. Escalated to compliance team."

    return f"Verdict: {decision.verdict.value}. {decision.reason}"


def _build_customer_explanation(
    event, context, proposed, failed_rules, passed_rules, decision
) -> str:
    parts = []

    parts.append(f"Customer {event.customer.id} attempted a ₹{event.amount // 100:,} payment on {event.type.value}.")

    if any(r.rule == "opt_out" and r.result == "FAIL" for r in failed_rules):
        parts.append("The customer has opted out of recovery communications. We will not contact them.")
        return " ".join(parts)

    if any(r.rule == "afa_check" and r.result == "FAIL" for r in failed_rules):
        parts.append(
            f"RBI regulations require fresh authentication for recurring payments above ₹15,000. "
            f"Your payment of ₹{event.amount // 100:,} needs a new mandate authorization. "
            f"A secure link will be sent to complete this step."
        )
        return " ".join(parts)

    if proposed.discount_percent > 0:
        parts.append(
            f"A {proposed.discount_percent}% discount has been applied, reducing your amount to "
            f"₹{(event.amount * (100 - proposed.discount_percent) // 100) // 100:,}."
        )

    if decision.verdict.value == "ALLOW":
        parts.append(f"We will retry via {proposed.channel.upper()} as per your preferred channel.")
    elif decision.verdict.value == "DENY":
        parts.append("We are unable to proceed with this recovery action at this time.")
    elif decision.verdict.value == "HUMAN_REVIEW":
        parts.append("A compliance officer will review this case shortly.")

    return " ".join(parts)


def _build_operator_explanation(
    event, diagnosis, context, proposed, decision, failed_rules
) -> str:
    parts = []

    parts.append(
        f"[{event.id}] {event.type.value} | ₹{event.amount // 100:,} | "
        f"customer={event.customer.id} | retry={event.retry_count}"
    )

    parts.append(
        f"Diagnosis: {diagnosis.classification} (confidence={diagnosis.confidence}, "
        f"recoverability={diagnosis.likely_recoverability}, action={diagnosis.recommended_action_family})"
    )

    parts.append(
        f"Context: consent={context.consent_status}, safe_to_contact={context.safe_to_contact}, "
        f"channel={context.preferred_channel}, disputes={context.active_dispute}, ptp={context.active_ptp}"
    )

    parts.append(
        f"Proposed: {proposed.action} via {proposed.channel}, "
        f"discount={proposed.discount_percent}%, tone={proposed.message_tone_level}"
    )

    if decision.verdict.value == "ALLOW":
        parts.append(
            f"Policy: ALLOW — all {len(decision.checks_passed)} checks passed. Execute immediately."
        )
    elif decision.verdict.value == "DENY":
        failed_explanations = "; ".join(r.explanation for r in failed_rules)
        parts.append(
            f"Policy: DENY — {len(failed_rules)} rule(s) failed. "
            f"Do not execute. Failed: {failed_explanations}"
        )
    elif decision.verdict.value == "HUMAN_REVIEW":
        failed_explanations = "; ".join(r.explanation for r in failed_rules)
        parts.append(
            f"Policy: HUMAN_REVIEW — {len(failed_rules)} rule(s) failed with high severity. "
            f"Escalate to compliance lead. {failed_explanations}"
        )

    return " | ".join(parts)
