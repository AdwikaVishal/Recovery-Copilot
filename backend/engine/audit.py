import json
import hashlib
from datetime import datetime
from app.database import get_db, db_session

AUDIT_RULE_VERSION = "4.0.0"


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _safe_json(obj):
    if obj is None:
        return None
    if isinstance(obj, str):
        try:
            json.loads(obj)
            return obj
        except (json.JSONDecodeError, TypeError):
            return json.dumps(obj, default=_json_default)
    return json.dumps(obj, default=_json_default)


async def log_supervisor_decision(
    event_id: str,
    customer_id: str,
    workflow_status: str,
    specialist_calls: list[dict],
    proposed_action: dict = None,
    policy_decision: dict = None,
    execution_result: dict = None,
    risk_flags: list[dict] = None,
) -> dict:
    # Single canonical timestamp drives BOTH the chain hash and the stored row.
    # (Previously a second datetime.utcnow() call ~17µs after the stored one
    # produced a hash that never matched the stored timestamp, so /api/audit/verify
    # reported every entry as broken.)
    now_iso = datetime.utcnow().isoformat()
    entry_id = f"audit_{event_id}_{int(datetime.utcnow().timestamp())}"

    diagnosis_confidence = 0.0
    if specialist_calls:
        for sc in specialist_calls:
            if sc.get("agent") == "diagnosis_agent":
                output = sc.get("output_summary", "")
                if "confidence=" in output:
                    try:
                        confidence_str = output.split("confidence=")[1].split(",")[0].split(" ")[0]
                        diagnosis_confidence = float(confidence_str)
                    except (ValueError, IndexError):
                        pass
                break

    amount = 0
    if proposed_action:
        amount = proposed_action.get("amount", 0)

    action = "none"
    reason = ""
    channel = "none"
    if proposed_action:
        action = proposed_action.get("action", "none")
        reason = proposed_action.get("reason", "")
        channel = proposed_action.get("channel", "none")

    result = "none"
    if execution_result:
        result = execution_result.get("result", "none")

    async with db_session() as db:
        cursor = await db.execute(
            "SELECT entry_hash FROM audit_log ORDER BY timestamp DESC LIMIT 1"
        )
        prev_row = await cursor.fetchone()
        prev_hash = prev_row["entry_hash"] if prev_row else ""

        chain_payload = f"{entry_id}|{now_iso}|{event_id}|{workflow_status}|{action}|{result}|{prev_hash}"
        entry_hash = hashlib.sha256(chain_payload.encode()).hexdigest()

        await db.execute(
            """INSERT OR REPLACE INTO audit_log
               (id, timestamp, event_id, customer_id, workflow_status,
                specialist_calls, risk_flags, proposed_action,
                policy_decision, execution_result,
                action, reason, diagnosis_confidence, channel,
                amount_attempted, result, rule_version,
                prev_hash, entry_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
            entry_id,
            now_iso,
            event_id,
            customer_id,
            workflow_status,
            _safe_json(specialist_calls),
            _safe_json(risk_flags or []),
            _safe_json(proposed_action),
            _safe_json(policy_decision),
            _safe_json(execution_result),
                action,
                reason,
                diagnosis_confidence,
                channel,
                amount,
                result,
                AUDIT_RULE_VERSION,
                prev_hash,
                entry_hash,
            ),
        )

    return {"id": entry_id, "event_id": event_id, "workflow_status": workflow_status}


async def get_audit_trail(event_id: str) -> list[dict]:
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT * FROM audit_log WHERE event_id = ? ORDER BY timestamp",
            (event_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_all_audit_entries() -> list[dict]:
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
