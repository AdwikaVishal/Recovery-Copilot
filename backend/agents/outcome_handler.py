from datetime import datetime
from app.models import RevenueEvent, ExecutionResult
from app.database import get_db, db_session, record_contact_event, record_strategy_outcome


async def record_outcome(event: RevenueEvent, result: ExecutionResult, workflow_status: str = None,
                         ml_probability: float = None, expected_value: int = None) -> dict:
    # Outcome -> learning loop: persist the executed strategy + its result so the
    # recovery analytics can compute empirical effectiveness and feed the
    # probabilities back into future recovery-optimizer decisions.
    #
    # The recorded probability is the authoritative ML recovery probability when
    # it was available at decision time (passed through from the optimizer);
    # otherwise it degrades to the documented rule-based cold prior.
    try:
        prob = (
            float(ml_probability)
            if ml_probability is not None
            else _predicted_probability(event, result.action)
        )
        ev = (
            int(expected_value)
            if expected_value is not None
            else _expected_value(event, result.action)
        )
        await record_strategy_outcome(
            event_id=event.id,
            strategy=_strategy_for_action(result.action),
            action=result.action,
            channel=result.channel or "none",
            amount=event.amount,
            success=result.result == "success",
            recovered_amount=result.amount_recovered,
            probability=prob,
            expected_value=ev,
            decline_code=event.decline_code.value,
            diagnosis_confidence=_diagnosis_confidence(event),
            safe_to_contact=not event.customer.opted_out,
            source="batch" if event.id.startswith("txn_") else "live",
        )
    except Exception:
        pass

    async with db_session() as db:
        new_status = _map_result_to_status(result, workflow_status)
        new_retry_count = event.retry_count + 1 if result.action == "retry_payment" else event.retry_count

        await db.execute(
            """UPDATE revenue_events
               SET status = ?, recovered_amount = ?, retry_count = ?, last_attempt_at = ?
               WHERE id = ?""",
            (
                new_status,
                result.amount_recovered,
                new_retry_count,
                datetime.utcnow().isoformat(),
                event.id,
            ),
        )

    contact_actions = {"retry_payment", "send_payment_link", "send_dunning_message", "re_authorize_mandate"}
    if result.action in contact_actions:
        channel = result.channel or "unknown"
        try:
            await record_contact_event(
                customer_id=event.customer.id,
                event_id=event.id,
                channel=channel,
                status=result.result,
                message_type=result.action,
            )
        except Exception:
            pass

    return {
        "event_id": event.id,
        "new_status": new_status,
        "amount_recovered": result.amount_recovered,
        "retry_count": new_retry_count,
    }


def _strategy_for_action(action: str) -> str:
    return {
        "retry_payment": "RETRY",
        "re_authorize_mandate": "REAUTHORIZE",
        "send_payment_link": "PAYMENT_LINK",
        "send_dunning_message": "MESSAGE",
        "escalate_to_human": "HUMAN_REVIEW",
        "blocked": "STOP",
    }.get(action, action.upper() if action else "UNKNOWN")


def _predicted_probability(event: RevenueEvent, action: str):
    from agents.probability_estimator import _base_probability
    strategy = _strategy_for_action(action)
    return _base_probability(strategy, event)


def _expected_value(event: RevenueEvent, action: str):
    p = _predicted_probability(event, action)
    return int(event.amount * p)


def _diagnosis_confidence(event: RevenueEvent) -> float:
    try:
        from agents.diagnosis_agent import diagnose
        return diagnose(event).confidence
    except Exception:
        return 0.0


def _map_result_to_status(result: ExecutionResult, workflow_status: str = None) -> str:
    if workflow_status == "PENDING_WEBHOOK":
        return "pending_webhook"
    if workflow_status == "HUMAN_REVIEW":
        return "human_review"
    if result.result == "success":
        return "success"
    if result.result == "failed":
        return "failed"
    if result.result == "pending":
        return "pending_webhook"
    if result.result == "blocked":
        return "blocked"
    return "pending"


async def simulate_webhook(event_id: str, webhook_type: str, payload: dict) -> dict:
    async with db_session() as db:
        if webhook_type == "payment.captured":
            amount = payload.get("amount", 0)
            await db.execute(
                "UPDATE revenue_events SET status = 'success', recovered_amount = ? WHERE id = ?",
                (amount, event_id),
            )
        elif webhook_type == "payment.failed":
            await db.execute(
                "UPDATE revenue_events SET status = 'failed' WHERE id = ?",
                (event_id,),
            )
        elif webhook_type == "payment.authorized":
            await db.execute(
                "UPDATE revenue_events SET status = 'pending_webhook' WHERE id = ?",
                (event_id,),
            )
        elif webhook_type == "subscription.activated":
            await db.execute(
                "UPDATE revenue_events SET status = 'success' WHERE id = ?",
                (event_id,),
            )
        elif webhook_type == "mandate.reauthorized":
            await db.execute(
                "UPDATE revenue_events SET status = 'pending_webhook' WHERE id = ?",
                (event_id,),
            )

    return {"event_id": event_id, "webhook_type": webhook_type, "processed": True}


async def get_pending_webhook_events() -> list[dict]:
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT * FROM revenue_events WHERE status = 'pending_webhook'"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
