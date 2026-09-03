from datetime import datetime
from app.models import (
    RevenueEvent, SupervisorOutput, WorkflowStatus,
    SpecialistCall, RiskFlag, PolicyVerdict, DiagnosisOutput
)
from agents.diagnosis_agent import diagnose
from agents.customer_context_agent import get_customer_context
from agents.recovery_strategy_agent import build_strategy, strategy_to_proposed_action
from agents.recovery_optimizer import build_optimizer_output
from agents.message_agent import apply_message_to_action
from agents.ptp_agent import handle_promise, check_broken_promises
from agents.policy_engine import PolicyEngine
from agents.compliance_explainer import explain
from agents.human_approval_gate import HumanApprovalGate
from agents.execution_adapter import execute
from agents.outcome_handler import record_outcome
from engine.audit import log_supervisor_decision


def _diagnosis_to_risk_flags(diag: DiagnosisOutput) -> list[RiskFlag]:
    flags = []
    if diag.likely_recoverability == "UNKNOWN":
        flags.append(RiskFlag(code="UNKNOWN_RECOVERABILITY", severity="HIGH", message="Cannot determine recoverability"))
    if diag.likely_recoverability == "LOW":
        flags.append(RiskFlag(code="LOW_RECOVERABILITY", severity="MEDIUM", message=f"Recoverability: LOW (confidence {diag.confidence})"))
    if diag.recommended_action_family == "STOP":
        flags.append(RiskFlag(code="STOP_ACTION", severity="HIGH", message="Customer opted out or ineligible"))
    if diag.recommended_action_family == "HUMAN_REVIEW":
        flags.append(RiskFlag(code="HUMAN_REVIEW_NEEDED", severity="MEDIUM", message="Retries exhausted or ambiguous case"))
    if diag.confidence < 0.4:
        flags.append(RiskFlag(code="LOW_CONFIDENCE", severity="MEDIUM", message=f"Diagnosis confidence {diag.confidence} — high uncertainty"))
    if diag.recommended_action_family == "REAUTHORIZE":
        flags.append(RiskFlag(code="AFA_REQUIRED", severity="HIGH", message="RBI: recurring > ₹15,000 needs fresh AFA"))
    return flags


async def process_event(event: RevenueEvent, now: datetime = None,
                        ml_estimate: 'MLPrediction' = None,
                        execute_action: bool = True) -> SupervisorOutput:
    specialist_calls = []
    risk_flags = []

    if not event.id or not event.customer or not event.amount:
        return SupervisorOutput(
            event_id=event.id or "unknown",
            workflow_status=WorkflowStatus.STOPPED,
            specialist_calls=[SpecialistCall(
                agent="supervisor",
                input_summary="Invalid event",
                output_summary="Schema validation failed",
            )],
            risk_flags=[RiskFlag(code="INVALID_SCHEMA", severity="CRITICAL", message="Missing required fields")],
            next_step="STOP: Invalid event schema",
        )

    specialist_calls.append(SpecialistCall(
        agent="supervisor",
        input_summary=f"Event {event.id}, type={event.type}, amount=₹{event.amount // 100:,}",
        output_summary=f"Event classified as {event.type}",
    ))

    diagnosis = diagnose(event)
    risk_flags.extend(_diagnosis_to_risk_flags(diagnosis))
    specialist_calls.append(SpecialistCall(
        agent="diagnosis_agent",
        input_summary=f"decline_code={event.decline_code}, retry_count={event.retry_count}, amount=₹{event.amount // 100:,}",
        output_summary=f"classification={diagnosis.classification}, recoverability={diagnosis.likely_recoverability}, action={diagnosis.recommended_action_family}, confidence={diagnosis.confidence}",
    ))

    if diagnosis.recommended_action_family == "STOP":
        await log_supervisor_decision(
            event_id=event.id,
            customer_id=event.customer.id,
            workflow_status="STOPPED",
            specialist_calls=[sc.model_dump() for sc in specialist_calls],
            risk_flags=[rf.model_dump() for rf in risk_flags],
        )
        return SupervisorOutput(
            event_id=event.id,
            workflow_status=WorkflowStatus.STOPPED,
            specialist_calls=specialist_calls,
            risk_flags=risk_flags,
            next_step="STOP: Customer opted out",
        )

    context = await get_customer_context(event)
    specialist_calls.append(SpecialistCall(
        agent="customer_context_agent",
        input_summary=f"customer_id={event.customer.id}",
        output_summary=f"consent={context.consent_status}, safe_to_contact={context.safe_to_contact}, channel={context.preferred_channel}, risk_flags={context.risk_flags}",
    ))

    if not context.safe_to_contact and diagnosis.recommended_action_family not in ("RETRY", "REAUTHORIZE"):
        await log_supervisor_decision(
            event_id=event.id,
            customer_id=event.customer.id,
            workflow_status="STOPPED",
            specialist_calls=[sc.model_dump() for sc in specialist_calls],
            risk_flags=[rf.model_dump() for rf in risk_flags],
        )
        return SupervisorOutput(
            event_id=event.id,
            workflow_status=WorkflowStatus.STOPPED,
            specialist_calls=specialist_calls,
            risk_flags=risk_flags,
            next_step=f"STOP: Not safe to contact — {context.risk_flags}",
        )

    # Recovery Optimizer: generate + rank multiple candidates by Expected Recovery
    # Value, then elect the best policy-eligible one. The Deterministic Policy Engine
    # below remains the hard safety boundary — the optimizer only proposes/optimizes.
    # When an ml_estimate is available the calibrated ML probability is the
    # authoritative P(recovery) for every candidate.
    optimizer = await build_optimizer_output(event, diagnosis, context,
                                             now=now, ml_estimate=ml_estimate)
    strategy = optimizer.strategy if optimizer.elected else build_strategy(event, diagnosis, context)
    specialist_calls.append(SpecialistCall(
        agent="recovery_optimizer",
        input_summary=f"action_family={diagnosis.recommended_action_family}, safe_to_contact={context.safe_to_contact}",
        output_summary=(f"candidates={len(optimizer.candidates)}, elected={strategy.strategy}, "
                        f"ev=₹{strategy.expected_value // 100:,} ({optimizer.selection_reason})"),
    ))

    # Recovery Scoring agent: surface the ranked candidate matrix with expected
    # recovery values + structured decision factors for the audit trace. This is
    # the explainable scoring layer — no probabilities are invented.
    scoring_summary = "; ".join(
        f"{c.strategy}={c.probability:.2f}->₹{c.expected_value // 100:,}"
        f"{'(ineligible:' + c.ineligibility_reason[:40] + ')' if not c.eligible else ''}"
        for c in optimizer.candidates
    )
    specialist_calls.append(SpecialistCall(
        agent="recovery_scoring_agent",
        input_summary=f"event={event.id}, decline={event.decline_code.value}, amount=₹{event.amount // 100:,}",
        output_summary=f"factors={optimizer.decision_factors}; ranked: {scoring_summary}",
    ))

    if strategy.strategy == "STOP":
        await log_supervisor_decision(
            event_id=event.id,
            customer_id=event.customer.id,
            workflow_status="STOPPED",
            specialist_calls=[sc.model_dump() for sc in specialist_calls],
            risk_flags=[rf.model_dump() for rf in risk_flags],
        )
        return SupervisorOutput(
            event_id=event.id,
            workflow_status=WorkflowStatus.STOPPED,
            specialist_calls=specialist_calls,
            risk_flags=risk_flags,
            next_step=f"STOP: {strategy.reason}",
            optimizer=(optimizer.model_dump() if optimizer else None),
        )

    proposed_action = strategy_to_proposed_action(event, strategy)

    if proposed_action.action in ("send_dunning_message", "send_payment_link", "re_authorize_mandate"):
        proposed_action = apply_message_to_action(event, proposed_action)
        specialist_calls.append(SpecialistCall(
            agent="message_agent",
            input_summary=f"action={proposed_action.action}, tone={proposed_action.message_tone_level}",
            output_summary=f"message generated ({len(proposed_action.message_text or '')} chars)",
        ))

    event_type_val = event.type.value if hasattr(event.type, 'value') else str(event.type)
    if event_type_val == "promise_to_pay":
        promise = await handle_promise(event)
        specialist_calls.append(SpecialistCall(
            agent="ptp_agent",
            input_summary=f"customer={event.customer.id}, amount=₹{event.amount // 100:,}",
            output_summary=f"promise recorded: {promise}",
        ))

    broken = await check_broken_promises()
    if broken:
        risk_flags.append(RiskFlag(
            code="BROKEN_PROMISE",
            severity="HIGH",
            message=f"{len(broken)} broken promise(s) — escalate tone",
        ))

    policy_engine = PolicyEngine()
    policy_decision = policy_engine.evaluate(event, diagnosis, proposed_action, now=now)

    specialist_calls.append(SpecialistCall(
        agent="policy_engine",
        input_summary=f"verdict={policy_decision.verdict.value}, failed={policy_decision.checks_failed}",
        output_summary=f"verdict={policy_decision.verdict.value}",
    ))

    compliance_explanation = explain(event, diagnosis, context, proposed_action, policy_decision)
    specialist_calls.append(SpecialistCall(
        agent="compliance_explainer",
        input_summary=f"verdict={policy_decision.verdict.value}",
        output_summary=f"summary={compliance_explanation.summary[:120]}",
    ))

    if policy_decision.verdict == PolicyVerdict.DENY:
        await log_supervisor_decision(
            event_id=event.id,
            customer_id=event.customer.id,
            workflow_status="STOPPED",
            specialist_calls=[sc.model_dump() for sc in specialist_calls],
            proposed_action=proposed_action.model_dump(),
            policy_decision=policy_decision.model_dump(),
            risk_flags=[rf.model_dump() for rf in risk_flags],
        )
        return SupervisorOutput(
            event_id=event.id,
            workflow_status=WorkflowStatus.STOPPED,
            specialist_calls=specialist_calls,
            risk_flags=risk_flags,
            next_step=f"STOP: Policy DENY — {policy_decision.reason}",
            proposed_action=proposed_action,
            diagnosis=diagnosis,
            customer_context=context,
            compliance_explanation=compliance_explanation,
            policy_decision=policy_decision,
            optimizer=(optimizer.model_dump() if optimizer else None),
        )

    if policy_decision.verdict == PolicyVerdict.MODIFY:
        if policy_decision.modified_request:
            proposed_action.discount_percent = policy_decision.modified_request.get("discount_percent", proposed_action.discount_percent)
            proposed_action.amount = policy_decision.modified_request.get("amount", proposed_action.amount)
        specialist_calls.append(SpecialistCall(
            agent="policy_engine",
            input_summary=f"original: discount={proposed_action.discount_percent}%",
            output_summary=f"MODIFIED: discount reduced to {proposed_action.discount_percent}%",
        ))

    if policy_decision.verdict == PolicyVerdict.HUMAN_REVIEW:
        gate = HumanApprovalGate()
        gate_result = gate.evaluate(event, proposed_action, policy_decision)

        if gate_result["needs_approval"]:
            await log_supervisor_decision(
                event_id=event.id,
                customer_id=event.customer.id,
                workflow_status="HUMAN_REVIEW",
                specialist_calls=[sc.model_dump() for sc in specialist_calls],
                proposed_action=proposed_action.model_dump(),
                policy_decision=policy_decision.model_dump(),
                risk_flags=[rf.model_dump() for rf in risk_flags],
            )
            return SupervisorOutput(
                event_id=event.id,
                workflow_status=WorkflowStatus.HUMAN_REVIEW,
                specialist_calls=specialist_calls,
                risk_flags=risk_flags,
                next_step=f"HUMAN_REVIEW: {gate_result['reasons']} — escalate to {gate_result['escalation_path']}",
                proposed_action=proposed_action,
                diagnosis=diagnosis,
                customer_context=context,
                compliance_explanation=compliance_explanation,
                policy_decision=policy_decision,
                optimizer=(optimizer.model_dump() if optimizer else None),
            )

    if not execute_action:
        # Decision-only mode: return the ML-informed decision and the policy
        # verdict WITHOUT executing the action or recording an outcome. Used by
        # the advisory /api/recovery/decision endpoint.
        return SupervisorOutput(
            event_id=event.id,
            workflow_status=WorkflowStatus.READY_FOR_POLICY,
            specialist_calls=specialist_calls,
            risk_flags=risk_flags,
            next_step=(f"DECISION: {policy_decision.verdict.value} — "
                       f"{strategy.strategy} ({policy_decision.reason})"),
            proposed_action=proposed_action,
            diagnosis=diagnosis,
            customer_context=context,
            compliance_explanation=compliance_explanation,
            policy_decision=policy_decision,
            optimizer=(optimizer.model_dump() if optimizer else None),
        )

    execution_result = await execute(event, proposed_action)
    specialist_calls.append(SpecialistCall(
        agent="execution_adapter",
        input_summary=f"action={proposed_action.action}",
        output_summary=f"result={execution_result.result}, recovered=₹{execution_result.amount_recovered // 100:,}",
    ))

    if execution_result.result == "pending":
        workflow_status = WorkflowStatus.PENDING_WEBHOOK
    elif execution_result.result == "success":
        workflow_status = WorkflowStatus.RESOLVED
    else:
        workflow_status = WorkflowStatus.STOPPED

    await record_outcome(event, execution_result, workflow_status.value,
                         ml_probability=_ml_probability(ml_estimate, optimizer),
                         expected_value=(optimizer.elected.expected_value if optimizer and optimizer.elected else None))

    await log_supervisor_decision(
        event_id=event.id,
        customer_id=event.customer.id,
        workflow_status=workflow_status.value,
        specialist_calls=[sc.model_dump() for sc in specialist_calls],
        proposed_action=proposed_action.model_dump(),
        policy_decision=policy_decision.model_dump(),
        execution_result=execution_result.model_dump(),
        risk_flags=[rf.model_dump() for rf in risk_flags],
    )

    return SupervisorOutput(
        event_id=event.id,
        workflow_status=workflow_status,
        specialist_calls=specialist_calls,
        risk_flags=risk_flags,
        next_step=f"EXECUTED: {execution_result.result} — {execution_result.reason}",
        proposed_action=proposed_action,
        diagnosis=diagnosis,
        customer_context=context,
        compliance_explanation=compliance_explanation,
        policy_decision=policy_decision,
        optimizer=(optimizer.model_dump() if optimizer else None),
    )


def _ml_probability(ml_estimate, optimizer):
    """Return the authoritative ML recovery probability for outcome recording.

    Prefers the passed-in ML estimate; falls back to the elected candidate's
    probability (which itself is the ML probability on the ML-enabled path).
    Returns None when no ML signal is present so the caller falls back to the
    documented rule-based cold prior.
    """
    if ml_estimate is not None and getattr(ml_estimate, "available", False):
        p = getattr(ml_estimate, "recovery_probability", None)
        if p is not None:
            return p
    if optimizer is not None and getattr(optimizer, "elected", None) is not None:
        src = getattr(optimizer.elected, "probability_source", "")
        return optimizer.elected.probability
    return None