from datetime import datetime, timedelta
from app.models import RevenueEvent, CustomerContextOutput, ContactFrequency
from app.database import get_db, count_contacts_since
from app.config import PolicyConfig


async def get_customer_context(event: RevenueEvent) -> CustomerContextOutput:
    risk_flags = []
    customer_id = event.customer.id
    config = PolicyConfig()
    db = await get_db()

    consent_status = "OPTED_OUT" if event.customer.opted_out else "CONSENTED"

    dnd_registered = False
    cursor = await db.execute(
        "SELECT dnd_registered FROM revenue_events WHERE customer_id = ? LIMIT 1",
        (customer_id,),
    )
    row = await cursor.fetchone()
    if row and row["dnd_registered"]:
        dnd_registered = True

    if dnd_registered:
        consent_status = "OPTED_OUT"
        risk_flags.append("DND_REGISTERED")

    now = datetime.utcnow()
    since_24h = (now - timedelta(hours=24)).isoformat()
    since_7d = (now - timedelta(days=7)).isoformat()

    contacts_24h = 0
    contacts_7d = 0
    try:
        contacts_24h = await count_contacts_since(customer_id, since_24h)
        contacts_7d = await count_contacts_since(customer_id, since_7d)
    except Exception:
        risk_flags.append("CONTACT_HISTORY_UNAVAILABLE")

    last_contact = None
    try:
        cursor = await db.execute(
            """SELECT sent_at FROM contact_events
               WHERE customer_id = ?
               ORDER BY sent_at DESC LIMIT 1""",
            (customer_id,),
        )
        row = await cursor.fetchone()
        last_contact = row["sent_at"] if row else None
    except Exception:
        pass

    if not last_contact:
        try:
            cursor = await db.execute(
                """SELECT timestamp FROM audit_log
                   WHERE customer_id = ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (customer_id,),
            )
            row = await cursor.fetchone()
            last_contact = row["timestamp"] if row else None
        except Exception:
            risk_flags.append("CONTACT_HISTORY_UNAVAILABLE")

    active_dispute = False
    try:
        cursor = await db.execute(
            """SELECT COUNT(*) as c FROM audit_log
               WHERE customer_id = ?
               AND (reason LIKE '%dispute%' OR reason LIKE '%chargeback%')""",
            (customer_id,),
        )
        dispute_count = (await cursor.fetchone())["c"]
        active_dispute = dispute_count > 0
    except Exception:
        pass

    active_ptp = False
    try:
        cursor = await db.execute(
            """SELECT COUNT(*) as c FROM ptp_promises
               WHERE customer_id = ? AND status = 'pending'""",
            (customer_id,),
        )
        ptp_count = (await cursor.fetchone())["c"]
        active_ptp = ptp_count > 0
    except Exception:
        pass

    preferred_channel = "UNKNOWN"
    if event.customer.language_pref == "hi":
        preferred_channel = "WHATSAPP"
    elif event.customer.language_pref == "en":
        preferred_channel = "EMAIL"

    try:
        cursor = await db.execute(
            """SELECT channel, COUNT(*) as c FROM contact_events
               WHERE customer_id = ? AND status = 'delivered'
               GROUP BY channel ORDER BY c DESC LIMIT 1""",
            (customer_id,),
        )
        row = await cursor.fetchone()
        if row and row["channel"]:
            raw = row["channel"].upper()
            if raw in ("WHATSAPP", "SMS", "EMAIL"):
                preferred_channel = raw
    except Exception:
        pass

    open_complaints = 0
    try:
        cursor = await db.execute(
            """SELECT COUNT(*) as c FROM audit_log
               WHERE customer_id = ?
               AND (reason LIKE '%complaint%' OR reason LIKE '%grievance%'
                    OR reason LIKE '%escalat%')""",
            (customer_id,),
        )
        open_complaints = (await cursor.fetchone())["c"]
    except Exception:
        pass

    if open_complaints > 0:
        risk_flags.append("OPEN_COMPLAINTS")

    has_valid_payment_method = True
    try:
        cursor = await db.execute(
            """SELECT decline_code, COUNT(*) as c FROM revenue_events
               WHERE customer_id = ?
               AND decline_code IN ('expired_card', 'incorrect_cvc')
               GROUP BY decline_code""",
            (customer_id,),
        )
        rows = await cursor.fetchall()
        if rows:
            for row in rows:
                if row["c"] >= 2:
                    has_valid_payment_method = False
                    risk_flags.append("PAYMENT_METHOD_ISSUES")
    except Exception:
        pass

    await db.close()

    safe_to_contact = True
    max_contacts_24h = config.get("max_contacts_per_day", 1)
    max_contacts_7d = config.max_contacts_per_week

    if consent_status == "OPTED_OUT":
        safe_to_contact = False
        risk_flags.append("CUSTOMER_OPTED_OUT")

    if active_dispute:
        safe_to_contact = False
        risk_flags.append("ACTIVE_DISPUTE")

    if "CONTACT_HISTORY_UNAVAILABLE" in risk_flags:
        safe_to_contact = False

    if contacts_24h >= max_contacts_24h:
        safe_to_contact = False
        risk_flags.append("FREQUENCY_LIMIT_24H")

    if contacts_7d >= max_contacts_7d:
        safe_to_contact = False
        risk_flags.append("FREQUENCY_LIMIT_7D")

    if open_complaints > 2:
        safe_to_contact = False
        risk_flags.append("EXCESSIVE_COMPLAINTS")

    return CustomerContextOutput(
        customer_id=customer_id,
        consent_status=consent_status,
        contact_frequency=ContactFrequency(
            contacts_last_24h=contacts_24h,
            contacts_last_7d=contacts_7d,
            last_contact_at=last_contact,
        ),
        active_dispute=active_dispute,
        active_ptp=active_ptp,
        preferred_channel=preferred_channel,
        risk_flags=list(set(risk_flags)),
        safe_to_contact=safe_to_contact,
    )
