"""FastAPI router: /api/recovery/* — frozen-artifact recovery predictions.

Follows the project's existing ``/api`` prefix convention. All heavy model
objects are loaded lazily via the process-wide singleton on first request,
never at import time.
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
import pandas as pd

from app.recovery.feature_builder import ALL_FEATURES, build_raw_features
from app.recovery.model_service import get_model_service, risk_band, risk_label
from app.recovery.schemas import (
    RecoveryAnalyzeRequest,
    RecoveryAnalyzeResponse,
    MLSignalBlock,
    AIDecisionBlock,
    PolicyBlock,
    ExecutionBlock,
    OutcomeBlock,
    RecoveryBatchRequest,
    RecoveryBatchResponse,
    RecoveryDecisionRequest,
    RecoveryDecisionResponse,
    RecoveryPredictRequest,
    RecoveryPredictResponse,
    RecoveryProcessRequest,
)

router = APIRouter(prefix="/api/recovery", tags=["recovery"])


def _predict_one(request: RecoveryPredictRequest,
                 service=None) -> RecoveryPredictResponse:
    service = service or get_model_service()
    raw = build_raw_features(request)
    raw_df = pd.DataFrame([raw], columns=ALL_FEATURES)
    encoded = service.encode(raw_df)
    prob_raw, prob_cal = service.predict_proba(encoded)
    pred = int(service.decide(prob_cal)[0])
    return RecoveryPredictResponse(
        transaction_id=request.transaction_id,
        customer_id=request.customer_id,
        recovery_probability=round(float(prob_cal[0]), 6),
        probability_raw=round(float(prob_raw[0]), 6),
        threshold=service.selected_threshold,
        recovery_prediction=pred,
        recovery_risk=risk_label(float(prob_cal[0])),
        risk_band=risk_band(float(prob_cal[0])),
        calibrated=True,
        explanation=service.explain(raw_df),
    )


@router.get("/model-info")
async def model_info():
    return get_model_service().model_info()


@router.get("/metrics")
async def metrics():
    return get_model_service().metrics


@router.get("/features")
async def features():
    return get_model_service().features_spec()


@router.post("/predict", response_model=RecoveryPredictResponse)
async def predict(request: RecoveryPredictRequest):
    """Score one recovery-population transaction.

    Flow: transaction -> 44 raw features -> 49 encoded features ->
    ExtraTreesClassifier -> raw probability -> sigmoid calibration ->
    threshold -> risk decision.
    """
    try:
        result = _predict_one(request)
    except Exception as exc:  # surface data-format errors clearly
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
    try:
        from app.database import record_prediction
        await record_prediction(result)
    except Exception:
        # Persistence is best-effort; inference result is still returned.
        pass
    return result


@router.post("/batch-predict", response_model=RecoveryBatchResponse)
async def batch_predict(batch: RecoveryBatchRequest):
    """Score many transactions; each goes through the same feature pipeline."""
    service = get_model_service()
    predictions: List[RecoveryPredictResponse] = []
    try:
        for request in batch.transactions:
            try:
                predictions.append(_predict_one(request, service))
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Prediction failed for {request.transaction_id}: {exc}",
                )
    except HTTPException:
        raise
    return RecoveryBatchResponse(
        threshold=service.selected_threshold,
        count=len(predictions),
        predictions=predictions,
    )


@router.get("/predictions")
async def recent_predictions(limit: int = 20):
    """Recently persisted single predictions (ranking/search support)."""
    try:
        from app.database import get_recent_predictions
        return await get_recent_predictions(limit=max(1, min(limit, 500)))
    except Exception:
        return []


def _build_event(transaction_id: str, customer_id: str, amount: float,
                 transaction_date: str, decline_code: str = "generic_decline") -> "RevenueEvent":
    from datetime import datetime
    from app.models import Customer, DeclineCode, EventType, RevenueEvent
    if not transaction_date:
        transaction_date = datetime.utcnow().isoformat()
    try:
        code = DeclineCode(decline_code)
    except ValueError:
        code = DeclineCode.GENERIC_DECLINE
    return RevenueEvent(
        id=transaction_id,
        type=EventType.CARD_PAYMENT_FAILURE,
        customer=Customer(id=customer_id, name=customer_id, phone="", email=""),
        amount=int(amount or 0),
        currency="INR",
        root_cause=code,
        decline_code=code,
        failed_at=datetime.fromisoformat(transaction_date),
        status="pending",
        retry_count=0,
        transaction_id=transaction_id,
    )


def _decision_response_from_record(record: dict,
                                   recovery_probability: float = None) -> RecoveryDecisionResponse:
    """Compose the full-chain response from a persisted decision record."""
    reasoning = (record.get("reasoning") or "").split("; ") if record.get("reasoning") else []
    return RecoveryDecisionResponse(
        transaction_id=record["transaction_id"],
        customer_id=record["customer_id"],
        recovery_probability=(
            float(recovery_probability)
            if recovery_probability is not None
            else float(record.get("ml_probability") or 0.0)
        ),
        probability_raw=record.get("probability_raw"),
        threshold=record.get("threshold"),
        risk_band=record.get("risk_band") or "",
        risk_label=record.get("risk_label") or "",
        recovery_prediction=record.get("recovery_prediction") or 0,
        model=record.get("model") or "",
        model_artifact=record.get("model_artifact") or "",
        probability_source=record["probability_source"],
        ai_decision=record["ai_decision"],
        action=record["action"],
        channel=record["channel"],
        policy_verdict=record["policy_verdict"],
        policy_reason=record.get("policy_reason") or "",
        reasoning=reasoning,
        expected_outcome=record.get("expected_outcome") or "",
        action_mode=record.get("action_mode") or "simulated",
        outcome=record.get("outcome") or "pending",
    )


def _ai_decision_from_optimizer(optimizer: dict | None) -> str:
    """Extract the elected strategy label from the optimizer dict trace."""
    if not optimizer:
        return "STOP"
    strat = optimizer.get("strategy") or {}
    if isinstance(strat, str):
        return strat
    return strat.get("strategy") or "STOP"


@router.post("/decision", response_model=RecoveryDecisionResponse)
async def recovery_decision(request: RecoveryDecisionRequest):
    """Advisory buildathon chain (no action executed).

    prediction -> AI decision -> policy verdict. The ML probability comes from
    the trained model bridge and is the authoritative P(recovery); on model
    unavailability it degrades to the documented rule-based fallback with
    ``probability_source`` intact.
    """
    from app.recovery.bridge import recovery_prediction_for_event, RULE_BASED_VERSION
    from agents.supervisor import process_event
    from app.database import record_recovery_decision

    event = _build_event(request.transaction_id, request.customer_id,
                         request.amount, request.transaction_date or "",
                         request.decline_code)
    ml = recovery_prediction_for_event(event, payment_method=request.payment_method)
    supervisor = await process_event(event, ml_estimate=ml, execute_action=False)

    optimizer = supervisor.optimizer or {}
    opt_strategy = _ai_decision_from_optimizer(optimizer)
    pd = supervisor.policy_decision
    verdict = pd.verdict.value if pd else "BLOCKED"
    declined = bool(supervisor.declined_to_act) if hasattr(supervisor, "declined_to_act") else False
    try:
        await record_recovery_decision(
            transaction_id=event.transaction_id,
            event_id=event.id,
            customer_id=event.customer.id,
            ml_probability=(ml.recovery_probability if ml.available else None),
            probability_raw=(ml.probability_raw if ml.available else None),
            threshold=(ml.threshold if ml.available else None),
            recovery_prediction=(ml.recovery_prediction if ml.available else None),
            risk_band=(ml.risk_band if ml.available else None),
            risk_label=(ml.risk_label if ml.available else None),
            model=(ml.model if ml.available else None),
            model_artifact=(ml.model_artifact if ml.available else None),
            probability_source=(optimizer or {}).get("probability_source")
            or (ml.probability_source if ml.available else RULE_BASED_VERSION),
            ai_decision=opt_strategy,
            action=supervisor.proposed_action.action,
            channel=supervisor.proposed_action.channel,
            policy_verdict=verdict,
            policy_reason=pd.reason if pd else supervisor.next_step,
            reasoning="; ".join(optimizer.get("decision_factors") or []) or supervisor.next_step,
            expected_outcome=supervisor.next_step,
            action_mode="advisory",
            outcome="pending",
        )
    except Exception:
        pass

    return RecoveryDecisionResponse(
        transaction_id=event.transaction_id,
        customer_id=event.customer.id,
        recovery_probability=(float(ml.recovery_probability) if ml.available else 0.0),
        probability_raw=(ml.probability_raw if ml.available else None),
        threshold=(ml.threshold if ml.available else None),
        risk_band=ml.risk_band or "",
        risk_label=ml.risk_label or "",
        recovery_prediction=(ml.recovery_prediction or 0),
        model=ml.model if ml.available else "",
        model_artifact=ml.model_artifact if ml.available else "",
        probability_source=ml.probability_source if ml.available else RULE_BASED_VERSION,
        ai_decision=opt_strategy,
        action=supervisor.proposed_action.action,
        channel=supervisor.proposed_action.channel,
        policy_verdict=verdict,
        policy_reason=pd.reason if pd else supervisor.next_step,
        reasoning=optimizer.get("decision_factors") or [],
        expected_outcome=supervisor.next_step,
        action_mode="advisory",
        outcome="pending",
        declined_to_act=declined,
    )


@router.post("/process", response_model=RecoveryDecisionResponse)
async def recovery_process(request: RecoveryProcessRequest):
    """Full buildathon chain INCLUDING (simulated) execution + persistence.

    prediction -> AI decision -> policy verdict -> (simulated) action ->
    outcome record. Runs through the real live engine (run_live_recovery) so
    the SSE log, attempt, sequence and learning bookkeeping all apply, and the
    outcome record is stored in ``recovery_decisions`` for the dashboard.
    """
    from app.recovery.bridge import recovery_prediction_for_event, RULE_BASED_VERSION
    from engine.realtime import run_live_recovery
    from app.database import get_recovery_decision

    event = _build_event(request.transaction_id, request.customer_id,
                         request.amount, request.transaction_date or "",
                         request.decline_code)
    ml = recovery_prediction_for_event(event, payment_method=request.payment_method)
    from uuid import uuid4
    correlation_id = f"api_{uuid4().hex[:12]}"
    run = await run_live_recovery(event, correlation_id, source="api",
                                  ml_estimate=ml)
    record = await get_recovery_decision(event.transaction_id)
    if record:
        resp = _decision_response_from_record(
            record, recovery_probability=(ml.recovery_probability if ml.available else 0.0))
        return resp
    # Fallback: no record persisted -> build the response from the run dict.
    return RecoveryDecisionResponse(
        transaction_id=event.transaction_id,
        customer_id=event.customer.id,
        recovery_probability=(float(ml.recovery_probability) if ml.available else 0.0),
        probability_raw=(ml.probability_raw if ml.available else None),
        threshold=(ml.threshold if ml.available else None),
        risk_band=ml.risk_band or "",
        risk_label=ml.risk_label or "",
        recovery_prediction=(ml.recovery_prediction or 0),
        model=ml.model if ml.available else "",
        model_artifact=ml.model_artifact if ml.available else "",
        probability_source=ml.probability_source if ml.available else RULE_BASED_VERSION,
        ai_decision="STOP",
        action=run.get("action") or "",
        channel="",
        policy_verdict=run.get("policy_verdict") or "BLOCKED",
        policy_reason=run.get("status") or "",
        reasoning="",
        expected_outcome=f"workflow {run.get('workflow_status', '')}",
        action_mode="simulated",
        outcome=run.get("status") or "pending",
    )


@router.get("/outcomes")
async def recovery_outcomes(limit: int = 20):
    """Recent prediction->decision->outcome records (chain linkage visible)."""
    try:
        from app.database import get_recent_recovery_decisions
        return await get_recent_recovery_decisions(limit=max(1, min(limit, 500)))
    except Exception:
        return []


@router.get("/analytics")
async def recovery_analytics():
    """Closed-loop aggregate: source, decision, verdict, outcome + recovered amount."""
    try:
        from app.database import get_recovery_analytics
        return await get_recovery_analytics()
    except Exception:
        return {}


@router.post("/analyze", response_model=RecoveryAnalyzeResponse)
async def recovery_analyze(request: RecoveryAnalyzeRequest):
    """Unified ML -> AI -> Policy -> Action -> Outcome chain with full provenance.

    Returns every layer with its own data source so the dependency is
    transparent:

        ML SIGNAL
            recovery probability: 3.31%
            source: ml:ExtraTreesClassifier-500-final

            ->

        AI DECISION
            diagnosis: temporary_cash_flow_issue
            selected action: RETRY
            expected recovery: ₹310

            ->

        POLICY
            ALLOW

            ->

        EXECUTION
            simulated retry

            ->

        OUTCOME
            awaiting trusted confirmation
    """
    from app.recovery.bridge import recovery_prediction_for_event, MODEL_VERSION, RULE_BASED_VERSION
    from agents.diagnosis_agent import diagnose
    from agents.customer_context_agent import get_customer_context
    from agents.recovery_optimizer import build_optimizer_output, score_candidates
    from agents.supervisor import process_event as supervisor_process

    event = _build_event(request.transaction_id, request.customer_id,
                         request.amount, request.transaction_date or "",
                         request.decline_code)

    ml = recovery_prediction_for_event(event, payment_method=request.payment_method)

    diag = diagnose(event)
    context = await get_customer_context(event)
    optimizer = await build_optimizer_output(event, diag, context, ml_estimate=ml)
    supervisor = await supervisor_process(event, ml_estimate=ml, execute_action=False)

    elected = optimizer.elected
    pd_ = supervisor.policy_decision
    verdict = pd_.verdict.value if pd_ else "BLOCKED"
    pol_reason = pd_.reason if pd_ else supervisor.next_step
    pol_passed = pd_.checks_passed if pd_ else []
    pol_failed = pd_.checks_failed if pd_ else []

    action_success_prob = elected.probability if elected else (ml.recovery_probability if ml.available else 0.0)
    expected_recovery = elected.expected_value if elected else 0

    ranked = score_candidates(optimizer.candidates)
    cand_list = [
        {
            "strategy": c["strategy"],
            "action": c["action"],
            "probability": c["probability"],
            "expected_value": c["expected_value"],
            "eligible": c["eligible"],
            "probability_source": c["probability_source"],
        }
        for c in ranked
    ]

    ml_block = MLSignalBlock(
        recovery_probability=round(ml.recovery_probability, 6) if ml.available else 0.0,
        probability_raw=round(ml.probability_raw, 6) if ml.available else 0.0,
        threshold=ml.threshold if ml.available else 0.04,
        prediction=ml.recovery_prediction if ml.available else 0,
        risk_band=ml.risk_band or "",
        risk_label=ml.risk_label or "",
        model_version=ml.model_version if ml.available else "",
        probability_source=ml.probability_source if ml.available else RULE_BASED_VERSION,
        explanation=ml.explanation if ml.available else [],
    )

    ai_block = AIDecisionBlock(
        diagnosis=diag.classification,
        diagnosis_confidence=round(diag.confidence, 4),
        selected_action=elected.strategy if elected else "STOP",
        action_success_probability=round(action_success_prob, 6),
        expected_recovery=expected_recovery,
        reasoning=optimizer.decision_factors or [],
        candidate_actions=cand_list,
    )

    policy_block = PolicyBlock(
        verdict=verdict,
        reason=pol_reason,
        checks_passed=pol_passed,
        checks_failed=pol_failed,
    )

    execution_block = ExecutionBlock(
        action=supervisor.proposed_action.action if supervisor.proposed_action else "none",
        channel=supervisor.proposed_action.channel if supervisor.proposed_action else "none",
        mode="simulated",
        status="advisory",
    )

    outcome_block = OutcomeBlock(
        status="pending",
        recovered_amount=0,
        source=None,
    )

    return RecoveryAnalyzeResponse(
        transaction_id=event.transaction_id,
        customer_id=event.customer.id,
        ml_signal=ml_block,
        ai_decision=ai_block,
        policy=policy_block,
        execution=execution_block,
        outcome=outcome_block,
        model_version=MODEL_VERSION,
        probability_source=ml.probability_source if ml.available else RULE_BASED_VERSION,
    )