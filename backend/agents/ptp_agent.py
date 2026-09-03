from app.models import RevenueEvent
from app.database import get_db
from datetime import datetime


async def check_broken_promises() -> list[dict]:
    db = await get_db()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    cursor = await db.execute(
        """SELECT * FROM ptp_promises
           WHERE status = 'pending' AND promised_date <= ?
           ORDER BY promised_date""",
        (today,),
    )
    rows = await cursor.fetchall()
    await db.close()
    return [dict(row) for row in rows]


async def record_promise(
    customer_id: str,
    event_id: str,
    amount: int,
    promised_date: str,
    notes: str = "",
) -> dict:
    promise_id = f"ptp_{event_id}"
    db = await get_db()
    await db.execute(
        """INSERT OR REPLACE INTO ptp_promises
           (id, customer_id, event_id, amount, promised_date, status, created_at, notes)
           VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
        (promise_id, customer_id, event_id, amount, promised_date,
         datetime.utcnow().isoformat(), notes),
    )
    await db.commit()
    await db.close()
    return {"id": promise_id, "status": "pending", "promised_date": promised_date}


async def mark_fulfilled(promise_id: str) -> dict:
    db = await get_db()
    await db.execute(
        "UPDATE ptp_promises SET status = 'fulfilled', checked_at = ? WHERE id = ?",
        (datetime.utcnow().isoformat(), promise_id),
    )
    await db.commit()
    await db.close()
    return {"id": promise_id, "status": "fulfilled"}


async def mark_broken(promise_id: str) -> dict:
    db = await get_db()
    await db.execute(
        "UPDATE ptp_promises SET status = 'broken', checked_at = ? WHERE id = ?",
        (datetime.utcnow().isoformat(), promise_id),
    )
    await db.commit()
    await db.close()
    return {"id": promise_id, "status": "broken"}


async def get_active_promises() -> list[dict]:
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM ptp_promises WHERE status = 'pending' ORDER BY promised_date"
    )
    rows = await cursor.fetchall()
    await db.close()
    return [dict(row) for row in rows]


async def handle_promise(event: RevenueEvent, promise_date: str = None) -> dict:
    if promise_date is None:
        promise_date = datetime.utcnow().strftime("%Y-%m-%d")

    return await record_promise(
        customer_id=event.customer.id,
        event_id=event.id,
        amount=event.amount,
        promised_date=promise_date,
        notes=f"Customer promised to pay by {promise_date}",
    )
