import aiosqlite
import contextvars
from pathlib import Path
from contextlib import asynccontextmanager

DB_PATH = Path(__file__).parent.parent / "recovery_copilot.db"

_active_db_path: contextvars.ContextVar[Path] = contextvars.ContextVar(
    "_active_db_path", default=DB_PATH
)


def get_active_db_path() -> Path:
    return _active_db_path.get()


def set_active_db_path(path: Path) -> contextvars.Token:
    return _active_db_path.set(path)


def reset_active_db_path(token: contextvars.Token) -> None:
    _active_db_path.reset(token)


async def get_db():
    db_path = _active_db_path.get()
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    return db


@asynccontextmanager
async def db_session():
    db = await get_db()
    try:
        yield db
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()


async def init_db():
    async with db_session() as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS revenue_events (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                customer_name TEXT NOT NULL,
                customer_phone TEXT,
                customer_email TEXT,
                language_pref TEXT DEFAULT 'hi',
                opted_out INTEGER DEFAULT 0,
                amount INTEGER NOT NULL,
                currency TEXT DEFAULT 'INR',
                root_cause TEXT NOT NULL,
                decline_code TEXT NOT NULL,
                failed_at TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}',
                ground_truth TEXT DEFAULT 'uncertain',
                recovered_amount INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0,
                last_attempt_at TEXT,
                status TEXT DEFAULT 'pending',
                dnd_registered INTEGER DEFAULT 0,
                transaction_id TEXT,
                occurred_at TEXT,
                source TEXT DEFAULT 'unknown',
                correlation_id TEXT,
                received_at TEXT,
                confirmed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                workflow_status TEXT NOT NULL DEFAULT 'UNKNOWN',
                specialist_calls TEXT DEFAULT '[]',
                risk_flags TEXT DEFAULT '[]',
                proposed_action TEXT,
                policy_decision TEXT,
                execution_result TEXT,
                action TEXT NOT NULL DEFAULT 'none',
                reason TEXT DEFAULT '',
                diagnosis_confidence REAL,
                channel TEXT,
                amount_attempted INTEGER,
                result TEXT NOT NULL DEFAULT 'none',
                rule_version TEXT DEFAULT '4.0.0',
                prev_hash TEXT DEFAULT '',
                entry_hash TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS ptp_promises (
                id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                amount INTEGER NOT NULL,
                promised_date TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                checked_at TEXT,
                notes TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS batch_runs (
                batch_id TEXT PRIMARY KEY,
                total_records INTEGER,
                attempted INTEGER,
                recovered INTEGER,
                recovered_amount INTEGER,
                baseline_amount INTEGER,
                blocked_by_policy INTEGER,
                processed_at TEXT,
                human_review INTEGER DEFAULT 0,
                pending_webhook INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS contact_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                event_id TEXT,
                channel TEXT NOT NULL,
                sent_at TEXT NOT NULL,
                status TEXT NOT NULL,
                message_type TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_contact_customer_time
            ON contact_events(customer_id, sent_at);

            CREATE INDEX IF NOT EXISTS idx_audit_event
            ON audit_log(event_id);

            CREATE INDEX IF NOT EXISTS idx_audit_customer_time
            ON audit_log(customer_id, timestamp);

            CREATE INDEX IF NOT EXISTS idx_payment_status
            ON revenue_events(status);

            CREATE INDEX IF NOT EXISTS idx_revenue_customer
            ON revenue_events(customer_id);

            CREATE TABLE IF NOT EXISTS pipeline_lock (
                lock_key TEXT PRIMARY KEY,
                locked_at TEXT NOT NULL,
                batch_id TEXT
            );

            CREATE TABLE IF NOT EXISTS inbound_events (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL DEFAULT 'unknown',
                raw_payload TEXT,
                received_at TEXT NOT NULL,
                processed_at TEXT,
                status TEXT NOT NULL DEFAULT 'received'
            );

            CREATE TABLE IF NOT EXISTS strategy_outcomes (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                strategy TEXT NOT NULL,
                action TEXT NOT NULL,
                channel TEXT NOT NULL,
                amount INTEGER DEFAULT 0,
                success INTEGER DEFAULT 0,
                recovered_amount INTEGER DEFAULT 0,
                probability REAL DEFAULT 0.0,
                expected_value INTEGER DEFAULT 0,
                decline_code TEXT NOT NULL,
                diagnosis_confidence REAL DEFAULT 0.0,
                safe_to_contact INTEGER DEFAULT 0,
                executed_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'batch'
            );

            CREATE INDEX IF NOT EXISTS idx_strategy_outcomes_strategy
            ON strategy_outcomes(strategy);

            CREATE INDEX IF NOT EXISTS idx_strategy_outcomes_decline
            ON strategy_outcomes(decline_code);

            CREATE INDEX IF NOT EXISTS idx_inbound_status
            ON inbound_events(status);

            CREATE INDEX IF NOT EXISTS idx_inbound_received
            ON inbound_events(received_at);

            CREATE TABLE IF NOT EXISTS webhook_events (
                event_id TEXT PRIMARY KEY,
                correlation_id TEXT NOT NULL,
                source TEXT NOT NULL,
                signature_verified INTEGER DEFAULT 0,
                received_at TEXT NOT NULL,
                processed_at TEXT,
                status TEXT NOT NULL DEFAULT 'received',
                result_summary TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_webhook_correlation
            ON webhook_events(correlation_id);

            CREATE TABLE IF NOT EXISTS recovery_attempts (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                correlation_id TEXT,
                attempt_number INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                action TEXT NOT NULL,
                channel TEXT,
                amount INTEGER NOT NULL,
                probability REAL DEFAULT 0.0,
                expected_value INTEGER DEFAULT 0,
                policy_verdict TEXT,
                execution_result TEXT,
                amount_recovered INTEGER DEFAULT 0,
                outcome TEXT,
                decision_ms INTEGER,
                execution_ms INTEGER,
                attempted_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'live',
                received_at TEXT,
                decision_at TEXT,
                execution_at TEXT,
                confirmed_at TEXT,
                roundtrip_ms INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_recovery_attempts_event
            ON recovery_attempts(event_id);

            CREATE INDEX IF NOT EXISTS idx_recovery_attempts_corr
            ON recovery_attempts(correlation_id);

            CREATE TABLE IF NOT EXISTS recovery_sequences (
                key TEXT PRIMARY KEY,
                correlation_id TEXT NOT NULL,
                event_ids TEXT NOT NULL DEFAULT '[]',
                current_step INTEGER NOT NULL DEFAULT 1,
                max_steps INTEGER NOT NULL DEFAULT 3,
                status TEXT NOT NULL DEFAULT 'open',
                latest_action TEXT,
                latest_verdict TEXT,
                final_amount INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_recovery_sequence_status
            ON recovery_sequences(status);

            CREATE TABLE IF NOT EXISTS recovery_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                model TEXT NOT NULL,
                model_artifact TEXT NOT NULL,
                recovery_probability REAL NOT NULL,
                probability_raw REAL NOT NULL,
                threshold REAL NOT NULL,
                recovery_prediction INTEGER NOT NULL,
                recovery_risk TEXT,
                risk_band TEXT,
                predicted_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_prediction_transaction
            ON recovery_predictions(transaction_id);

            CREATE TABLE IF NOT EXISTS recovery_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL UNIQUE,
                event_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                ml_probability REAL,
                probability_raw REAL,
                threshold REAL,
                recovery_prediction INTEGER,
                risk_band TEXT,
                risk_label TEXT,
                model TEXT,
                model_artifact TEXT,
                probability_source TEXT NOT NULL,
                ai_decision TEXT NOT NULL,
                action TEXT NOT NULL,
                channel TEXT NOT NULL,
                policy_verdict TEXT NOT NULL,
                policy_reason TEXT,
                reasoning TEXT,
                expected_outcome TEXT,
                action_mode TEXT NOT NULL DEFAULT 'simulated',
                outcome TEXT NOT NULL DEFAULT 'pending',
                recovered_amount REAL NOT NULL DEFAULT 0,
                decision_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)

        await _ensure_column(db, "revenue_events", "transaction_id", "TEXT")
        await _ensure_column(db, "revenue_events", "occurred_at", "TEXT")
        await _ensure_column(db, "revenue_events", "source", "TEXT DEFAULT 'unknown'")
        await _ensure_column(db, "revenue_events", "correlation_id", "TEXT")
        await _ensure_column(db, "revenue_events", "received_at", "TEXT")
        await _ensure_column(db, "revenue_events", "confirmed_at", "TEXT")
        await _ensure_column(db, "recovery_attempts", "received_at", "TEXT")
        await _ensure_column(db, "recovery_attempts", "decision_at", "TEXT")
        await _ensure_column(db, "recovery_attempts", "execution_at", "TEXT")
        await _ensure_column(db, "recovery_attempts", "confirmed_at", "TEXT")
        await _ensure_column(db, "recovery_attempts", "roundtrip_ms", "INTEGER")


async def _ensure_column(db, table: str, column: str, ddl: str) -> None:
    """Add a column to an existing table (migration for pre-existing DBs)."""
    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    cols = {r["name"] for r in rows}
    if column not in cols:
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


async def record_contact_event(
    customer_id: str,
    event_id: str,
    channel: str,
    status: str = "sent",
    message_type: str = None,
) -> None:
    from datetime import datetime
    async with db_session() as db:
        await db.execute(
            """INSERT INTO contact_events
               (customer_id, event_id, channel, sent_at, status, message_type)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (customer_id, event_id, channel, datetime.utcnow().isoformat(), status, message_type),
        )


async def count_contacts_since(customer_id: str, since_iso: str) -> int:
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) as c FROM contact_events WHERE customer_id = ? AND sent_at > ?",
            (customer_id, since_iso),
        )
        row = await cursor.fetchone()
        return row["c"] if row else 0


async def record_strategy_outcome(
    *,
    event_id: str,
    strategy: str,
    action: str,
    channel: str,
    amount: int,
    success: bool,
    recovered_amount: int,
    probability: float,
    expected_value: int,
    decline_code: str,
    diagnosis_confidence: float,
    safe_to_contact: bool,
    source: str = "batch",
) -> str:
    """Record one executed recovery attempt for the learning/analytics loop.

    Keyed on event_id so repeated record_outcome calls (the supervisor and the
    pipeline both write, and batch replay DELETEs first) yield one row per event
    rather than corrupting the learning counts.
    """
    from datetime import datetime

    async with db_session() as db:
        await db.execute(
            """INSERT OR REPLACE INTO strategy_outcomes
               (id, event_id, strategy, action, channel, amount, success,
                recovered_amount, probability, expected_value, decline_code,
                diagnosis_confidence, safe_to_contact, executed_at, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id, event_id, strategy, action, channel, amount,
                1 if success else 0, recovered_amount, probability, expected_value,
                decline_code, diagnosis_confidence, 1 if safe_to_contact else 0,
                datetime.utcnow().isoformat(), source,
            ),
        )
    return event_id


async def get_strategy_effectiveness() -> list[dict]:
    """Roll up historical outcomes per strategy."""
    async with db_session() as db:
        cursor = await db.execute(
            """SELECT strategy,
                      COUNT(*) as attempts,
                      SUM(success) as successes,
                      SUM(recovered_amount) as recovery_amount,
                      AVG(probability) as avg_probability
               FROM strategy_outcomes
               GROUP BY strategy
               ORDER BY recovery_amount DESC"""
        )
        rows = await cursor.fetchall()

        cursor = await db.execute(
            "SELECT COUNT(*) as c FROM strategy_outcomes WHERE channel != 'NONE' AND channel != 'none'"
        )
        contacted_total = (await cursor.fetchone())["c"] or 0

    results = []
    for r in rows:
        attempts = r["attempts"] or 0
        successes = r["successes"] or 0
        recovery = r["recovery_amount"] or 0
        results.append({
            "strategy": r["strategy"],
            "attempts": attempts,
            "successes": successes,
            "recovery_amount": recovery,
            "contact_rate": (successes / attempts) if attempts else 0.0,
            "empirical_probability": (successes / attempts) if attempts else 0.0,
            "avg_recovery_per_attempt": (recovery // attempts) if attempts else 0,
        })
    return results


async def get_recovery_probability(
    strategy: str, decline_code: str, default: float, source: str = None
) -> float:
    """Empirical recovery probability for a strategy, learned from outcomes.

    Returns the observed success rate if enough samples exist, otherwise the
    supplied transparent default so the system degrades gracefully pre-data.
    When `source` is given, learning is scoped to that environment (live vs
    batch) so one environment's outcomes never leak into another's decisions.
    """
    async with db_session() as db:
        if source:
            cursor = await db.execute(
                """SELECT COUNT(*) as attempts, SUM(success) as successes
                   FROM strategy_outcomes
                   WHERE strategy = ? AND decline_code = ? AND source = ?""",
                (strategy, decline_code, source),
            )
        else:
            cursor = await db.execute(
                """SELECT COUNT(*) as attempts, SUM(success) as successes
                   FROM strategy_outcomes
                   WHERE strategy = ? AND decline_code = ?""",
                (strategy, decline_code),
            )
        row = await cursor.fetchone()
    attempts = row["attempts"] if row else 0
    successes = row["successes"] if row else 0
    if attempts >= 3:
        return successes / attempts
    return default


async def get_strategy_outcome_counts(
    strategy: str, decline_code: str, source: str = None
) -> tuple[int, int]:
    """Raw historical outcome counts (attempts, successes) for display.

    Feeds the outcome-informed optimizer stats (empirical n/m per candidate)
    without hiding the sample size behind a thresholded probability.
    """
    async with db_session() as db:
        if source:
            cursor = await db.execute(
                """SELECT COUNT(*) as attempts, SUM(success) as successes
                   FROM strategy_outcomes
                   WHERE strategy = ? AND decline_code = ? AND source = ?""",
                (strategy, decline_code, source),
            )
        else:
            cursor = await db.execute(
                """SELECT COUNT(*) as attempts, SUM(success) as successes
                   FROM strategy_outcomes
                   WHERE strategy = ? AND decline_code = ?""",
                (strategy, decline_code),
            )
        row = await cursor.fetchone()
    return (row["attempts"] or 0, row["successes"] or 0)


# ---------------------------------------------------------------------------
# Webhook idempotency
# ---------------------------------------------------------------------------

async def get_webhook_processing(event_id: str):
    """Return the stored processing record for an event_id, if any."""
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT * FROM webhook_events WHERE event_id = ?", (event_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def record_webhook_received(
    *,
    event_id: str,
    correlation_id: str,
    source: str,
    signature_verified: bool,
) -> bool:
    """Atomically register an inbound webhook; True if newly inserted.

    Uses INSERT OR IGNORE keyed on `event_id` as the exactly-once gate. A
    re-delivery (or a concurrently retried webhook) observes rowcount == 0 and
    is short-circuited BEFORE any pipeline work, so recovery never runs twice.
    """
    from datetime import datetime
    async with db_session() as db:
        cursor = await db.execute(
            """INSERT OR IGNORE INTO webhook_events
               (event_id, correlation_id, source, signature_verified, received_at, status)
               VALUES (?, ?, ?, ?, ?, 'received')""",
            (
                event_id, correlation_id, source,
                1 if signature_verified else 0,
                datetime.utcnow().isoformat(),
            ),
        )
        return cursor.rowcount == 1


async def mark_webhook_processed(event_id: str, result_summary: str, status: str = "processed") -> None:
    from datetime import datetime
    async with db_session() as db:
        await db.execute(
            """UPDATE webhook_events
               SET processed_at = ?, status = ?, result_summary = ?
               WHERE event_id = ?""",
            (datetime.utcnow().isoformat(), status, result_summary, event_id),
        )


# ---------------------------------------------------------------------------
# Recovery attempts (closed-loop trace + latency)
# ---------------------------------------------------------------------------

async def record_recovery_attempt(
    *,
    event_id: str,
    correlation_id: str,
    attempt_number: int,
    strategy: str,
    action: str,
    channel: str,
    amount: int,
    probability: float,
    expected_value: int,
    policy_verdict: str,
    execution_result: str,
    amount_recovered: int,
    outcome: str,
    decision_ms: int,
    execution_ms: int,
    source: str = "live",
    received_at: str = None,
    decision_at: str = None,
    execution_at: str = None,
    roundtrip_ms: int = None,
) -> str:
    from datetime import datetime
    import uuid
    attempt_id = f'att_{uuid.uuid4().hex[:12]}'
    async with db_session() as db:
        await db.execute(
            """INSERT INTO recovery_attempts
               (id, event_id, correlation_id, attempt_number, strategy, action, channel,
                amount, probability, expected_value, policy_verdict, execution_result,
                amount_recovered, outcome, decision_ms, execution_ms, attempted_at, source,
                received_at, decision_at, execution_at, roundtrip_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                attempt_id, event_id, correlation_id, attempt_number, strategy, action, channel,
                amount, probability, expected_value, policy_verdict, execution_result,
                amount_recovered, outcome, decision_ms, execution_ms,
                datetime.utcnow().isoformat(), source,
                received_at, decision_at, execution_at, roundtrip_ms,
            ),
        )
    return attempt_id


async def set_recovery_attempt_confirmed(event_id: str, confirmed_at: str = None) -> None:
    """Mark the most recent attempt for an event as confirmed (round-trip done)."""
    from datetime import datetime
    async with db_session() as db:
        await db.execute(
            """UPDATE recovery_attempts
               SET confirmed_at = ?, roundtrip_ms = CAST(
                     (julianday(?) - julianday(COALESCE(received_at, attempted_at))) * 86400000 AS INTEGER)
               WHERE event_id = ?
                 AND id = (SELECT id FROM recovery_attempts
                           WHERE event_id = ? ORDER BY attempted_at DESC LIMIT 1)""",
            (confirmed_at or datetime.utcnow().isoformat(),
             confirmed_at or datetime.utcnow().isoformat(),
             event_id, event_id),
        )


async def get_latest_recovery_attempt(event_id: str) -> dict | None:
    """Return the most recent recovery attempt recorded for an event."""
    async with db_session() as db:
        cursor = await db.execute(
            """SELECT * FROM recovery_attempts
               WHERE event_id = ? ORDER BY attempted_at DESC LIMIT 1""",
            (event_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Closed-loop recovery sequences (per-transaction max recovery steps)
# ---------------------------------------------------------------------------

async def register_recovery_step(*, key: str, event_id: str, correlation_id: str, max_steps: int) -> tuple[dict, int]:
    """Register one more inbound failure attempt for a transaction sequence.

    Creates the sequence on first sight and increments `current_step` each time
    a new failure event arrives with the same recovery key. Returns
    (sequence_row, attempt_number) where attempt_number is 1-based and
    attempt_number > max_steps means the MAX_RECOVERY_STEPS boundary is hit.
    """
    import json
    from datetime import datetime
    async with db_session() as db:
        cursor = await db.execute("SELECT * FROM recovery_sequences WHERE key = ?", (key,))
        row = await cursor.fetchone()
        now = datetime.utcnow().isoformat()
        if not row:
            await db.execute(
                """INSERT OR IGNORE INTO recovery_sequences
                   (key, correlation_id, event_ids, current_step, max_steps, status, created_at, updated_at)
                   VALUES (?, ?, ?, 1, ?, 'open', ?, ?)""",
                (key, correlation_id, json.dumps([event_id]), max_steps, now, now),
            )
            cursor = await db.execute("SELECT * FROM recovery_sequences WHERE key = ?", (key,))
            row = await cursor.fetchone()
            seq = dict(row)
            return seq, seq["current_step"]
        event_ids = json.loads(row["event_ids"] or "[]")
        if event_id not in event_ids:
            event_ids.append(event_id)
        await db.execute(
            """UPDATE recovery_sequences
               SET event_ids = ?, correlation_id = ?, current_step = current_step + 1, updated_at = ?
               WHERE key = ?""",
            (json.dumps(event_ids), correlation_id, now, key),
        )
        cursor = await db.execute("SELECT * FROM recovery_sequences WHERE key = ?", (key,))
        row = await cursor.fetchone()
        seq = dict(row)
        return seq, seq["current_step"]


async def get_open_recovery_sequence(key: str) -> dict | None:
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT * FROM recovery_sequences WHERE key = ? AND status = 'open'",
            (key,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def update_recovery_sequence(
    key: str, *, status=None, latest_action=None, latest_verdict=None, final_amount=None
) -> None:
    from datetime import datetime
    sets = ["updated_at = ?"]
    vals = [datetime.utcnow().isoformat()]
    if status is not None:
        sets.append("status = ?")
        vals.append(status)
    if latest_action is not None:
        sets.append("latest_action = ?")
        vals.append(latest_action)
    if latest_verdict is not None:
        sets.append("latest_verdict = ?")
        vals.append(latest_verdict)
    if final_amount is not None:
        sets.append("final_amount = ?")
        vals.append(final_amount)
    vals.append(key)
    async with db_session() as db:
        await db.execute(
            f"UPDATE recovery_sequences SET {', '.join(sets)} WHERE key = ?", vals
        )


async def close_recovery_sequence(key: str, status: str = "succeeded", final_amount: int = 0) -> None:
    await update_recovery_sequence(key, status=status, final_amount=final_amount)


async def record_prediction(prediction) -> None:
    """Persist a recovery-prediction result for auditability/ranking (best-effort)."""
    from datetime import datetime
    async with db_session() as db:
        await db.execute(
            """INSERT OR REPLACE INTO recovery_predictions
               (transaction_id, customer_id, model, model_artifact,
                recovery_probability, probability_raw, threshold,
                recovery_prediction, recovery_risk, risk_band, predicted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction.transaction_id, prediction.customer_id,
                prediction.model, prediction.model_artifact,
                prediction.recovery_probability, prediction.probability_raw,
                prediction.threshold, int(prediction.recovery_prediction),
                prediction.recovery_risk, prediction.risk_band,
                datetime.utcnow().isoformat(),
            ),
        )


async def get_recent_predictions(limit: int = 20) -> list[dict]:
    """Most recently persisted predictions, ranked order preserved."""
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT * FROM recovery_predictions ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def record_recovery_decision(
    *,
    transaction_id: str, event_id: str, customer_id: str,
    probability_source: str, ai_decision: str, action: str, channel: str,
    policy_verdict: str, policy_reason: str = "", reasoning: str = "",
    expected_outcome: str = "", action_mode: str = "simulated",
    outcome: str = "pending", recovered_amount: float = 0,
    ml_probability: float = None, probability_raw: float = None,
    threshold: float = None, recovery_prediction: int = None,
    risk_band: str = None, risk_label: str = None,
    model: str = None, model_artifact: str = None,
) -> None:
    """Persist the prediction -> decision -> outcome record (best-effort).

    One row per transaction_id (upsert). Confirmed outcomes later flow through
    update_recovery_outcome on trusted payment confirmation.
    """
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    async with db_session() as db:
        await db.execute(
            """INSERT OR REPLACE INTO recovery_decisions
               (transaction_id, event_id, customer_id,
                ml_probability, probability_raw, threshold,
                recovery_prediction, risk_band, risk_label,
                model, model_artifact, probability_source,
                ai_decision, action, channel, policy_verdict,
                policy_reason, reasoning, expected_outcome,
                action_mode, outcome, recovered_amount, decision_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                transaction_id, event_id, customer_id,
                ml_probability, probability_raw, threshold,
                recovery_prediction, risk_band, risk_label,
                model, model_artifact, probability_source,
                ai_decision, action, channel, policy_verdict,
                policy_reason, reasoning, expected_outcome,
                action_mode, outcome, recovered_amount, now, now,
            ),
        )


async def update_recovery_outcome(transaction_id: str, outcome: str,
                                  recovered_amount: float = 0) -> None:
    """Close a transaction's decision loop with its confirmed outcome."""
    from datetime import datetime
    async with db_session() as db:
        await db.execute(
            """UPDATE recovery_decisions
               SET outcome = ?, recovered_amount = ?, updated_at = ?
               WHERE transaction_id = ?""",
            (outcome, recovered_amount, datetime.utcnow().isoformat(), transaction_id),
        )


async def get_recent_recovery_decisions(limit: int = 20) -> list[dict]:
    """Most recent prediction->decision records, newest first."""
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT * FROM recovery_decisions ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_recovery_decision(transaction_id: str) -> dict | None:
    """One transaction's prediction->decision->outcome record."""
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT * FROM recovery_decisions WHERE transaction_id = ?",
            (transaction_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def get_recovery_analytics() -> dict:
    """Aggregate the closed loop across ML source, policy verdict and outcomes.

    Honest counts — labeled simulated because execution in the demo runtime is
    simulated; confirmation outcomes come only from trusted webhook events.
    """
    async with db_session() as db:
        cursor = await db.execute("SELECT COUNT(*) as c FROM recovery_decisions")
        total = (await cursor.fetchone())["c"] or 0

        cursor = await db.execute(
            """SELECT probability_source, COUNT(*) as c
               FROM recovery_decisions GROUP BY probability_source""")
        by_source = {r["probability_source"]: r["c"] for r in await cursor.fetchall()}

        cursor = await db.execute(
            """SELECT ai_decision, COUNT(*) as c
               FROM recovery_decisions GROUP BY ai_decision""")
        by_decision = {r["ai_decision"]: r["c"] for r in await cursor.fetchall()}

        cursor = await db.execute(
            """SELECT policy_verdict, COUNT(*) as c
               FROM recovery_decisions GROUP BY policy_verdict""")
        by_verdict = {r["policy_verdict"]: r["c"] for r in await cursor.fetchall()}

        cursor = await db.execute(
            """SELECT outcome, COUNT(*) as c
               FROM recovery_decisions GROUP BY outcome""")
        by_outcome = {r["outcome"]: r["c"] for r in await cursor.fetchall()}

        cursor = await db.execute(
            """SELECT COUNT(*) as c FROM recovery_decisions
               WHERE recovered_amount > 0""")
        with_recovered_amount = (await cursor.fetchone())["c"] or 0

        cursor = await db.execute(
            """SELECT COUNT(*) as c FROM recovery_decisions WHERE outcome = 'recovered_72h'""")
        recovered = (await cursor.fetchone())["c"] or 0

        cursor = await db.execute("""SELECT SUM(recovered_amount) as a FROM recovery_decisions""")
        recovered_amount = (await cursor.fetchone())["a"] or 0

    return {
        "total_decisions": total,
        "by_probability_source": by_source,
        "by_ai_decision": by_decision,
        "by_policy_verdict": by_verdict,
        "by_outcome": by_outcome,
        "recovered_count": recovered,
        "with_recovered_amount": with_recovered_amount,
        "recovered_amount": recovered_amount,
    }


async def get_live_metrics() -> dict:
    """Live recovery metrics aggregated from inbound live events + outcomes."""
    async with db_session() as db:
        # Live events = non-batch, non-scenario, non-evaluation revenue rows.
        cursor = await db.execute(
            """SELECT COUNT(*) as c FROM revenue_events
               WHERE id NOT LIKE 'txn_%' AND id NOT LIKE 'scenario_%'"""
        )
        live_events = (await cursor.fetchone())["c"] or 0

        cursor = await db.execute(
            """SELECT COUNT(*) as c FROM revenue_events
               WHERE id NOT LIKE 'txn_%' AND id NOT LIKE 'scenario_%' AND status = 'success'"""
        )
        live_confirmed = (await cursor.fetchone())["c"] or 0

        cursor = await db.execute(
            """SELECT SUM(recovered_amount) as a FROM revenue_events
               WHERE id NOT LIKE 'txn_%' AND id NOT LIKE 'scenario_%' AND status = 'success'"""
        )
        live_recovered = (await cursor.fetchone())["a"] or 0

        cursor = await db.execute(
            """SELECT COUNT(*) as c FROM revenue_events
               WHERE id NOT LIKE 'txn_%' AND id NOT LIKE 'scenario_%' AND status = 'blocked'"""
        )
        live_blocked = (await cursor.fetchone())["c"] or 0

        cursor = await db.execute(
            """SELECT COUNT(*) as c FROM revenue_events
               WHERE id NOT LIKE 'txn_%' AND id NOT LIKE 'scenario_%' AND status = 'human_review'"""
        )
        live_human = (await cursor.fetchone())["c"] or 0

        cursor = await db.execute(
            """SELECT COUNT(*) as c FROM revenue_events
               WHERE id NOT LIKE 'txn_%' AND id NOT LIKE 'scenario_%' AND status = 'pending_webhook'"""
        )
        live_pending = (await cursor.fetchone())["c"] or 0

        cursor = await db.execute(
            """SELECT COUNT(*) as c FROM recovery_attempts WHERE source = 'live'"""
        )
        live_attempts = (await cursor.fetchone())["c"] or 0

        # Latency: confirmed live events (trusted-confirmation round trip).
        cursor = await db.execute(
            """SELECT AVG(decision_ms) as d, AVG(execution_ms) as e
               FROM recovery_attempts WHERE source = 'live' AND outcome IS NOT NULL"""
        )
        lat = await cursor.fetchone()
        avg_decision_ms = int(lat["d"] or 0)
        avg_execution_ms = int(lat["e"] or 0)

        # Time-to-confirmation: received_at -> confirmed_at (wall-clock round-trip).
        cursor = await db.execute(
            """SELECT AVG((julianday(confirmed_at) - julianday(received_at)) * 86400.0) as t
               FROM recovery_attempts
               WHERE source = 'live' AND confirmed_at IS NOT NULL AND received_at IS NOT NULL"""
        )
        rowtt = await cursor.fetchone()
        time_to_confirmation_sec = float(rowtt["t"] or 0.0)

        # Closed-loop: open sequences = transactions still being recovered.
        cursor = await db.execute(
            "SELECT COUNT(*) as c FROM recovery_sequences WHERE status = 'open'"
        )
        open_sequences = (await cursor.fetchone())["c"] or 0

    roundtrip_sec = time_to_confirmation_sec if time_to_confirmation_sec > 0 else (
        (avg_decision_ms + avg_execution_ms) / 1000.0
    )
    return {
        "live_events": live_events,
        "live_recovery_attempts": live_attempts,
        "live_confirmed_payments": live_confirmed,
        "live_money_recovered": live_recovered,
        "live_money_recovered_display": f"₹{live_recovered // 100:,}",
        "live_blocked": live_blocked,
        "live_human_reviews": live_human,
        "live_pending": live_pending,
        "live_recovery_rate": round((live_confirmed / live_events) if live_events else 0.0, 4),
        "avg_decision_ms": avg_decision_ms,
        "avg_execution_ms": avg_execution_ms,
        "time_to_confirmation_sec": round(time_to_confirmation_sec, 2),
        "avg_recovery_time_sec": round(roundtrip_sec, 2),
        "open_recovery_sequences": open_sequences,
    }
