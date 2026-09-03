#!/usr/bin/env python3
"""
Streaming real-time event simulator for Recovery Copilot.

Streams simulated payment events through the REAL webhook ingress
(POST /api/webhooks/payment) with a demo HMAC signature, driving the
closed-loop live recovery pipeline over SSE.

Features:
  * Failed payment events (transaction_id-keyed recovery sequences).
  * --confirm-rate: a fraction of transactions receive a DEMO-labelled
    payment.captured confirmation, closing the sequence and recording
    recovered revenue (only trusted confirmations finalize money).
  * --closed-loop: non-confirmed transactions receive follow-up
    payment.failed webhooks (same transaction_id, retry_count++) so the
    bounded re-optimization loop can be observed end-to-end.

Usage:
    python3 -m tools.simulate_realtime
    python3 -m tools.simulate_realtime --interval 3
    python3 -m tools.simulate_realtime --count 20 --confirm-rate 0.7 --closed-loop
    python3 -m tools.simulate_realtime --interval 2 --count 10 --base-url http://localhost:8321
"""
import argparse
import hashlib
import hmac
import json
import os
import random
import sys
import time
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError


EVENT_TYPES = [
    ("payment.failed", [50000, 100000, 150000, 250000, 500000, 99900, 149900]),
    ("recurring_payment_failure", [100000, 200000, 500000, 1000000, 1600000, 2500000]),
    ("checkout_abandonment", [99900, 199900, 499900, 999900, 1999900]),
    ("overdue_invoice", [500000, 1000000, 2500000, 5000000]),
]

DECLINE_CODES = [
    "insufficient_funds", "expired_card", "do_not_honor",
    "bank_timeout", "processing_error", "mandate_simple_retry",
    "mandate_afa_required", "payment_link_expired", "invoice_overdue",
]

DEMO_SALT = "recovery-copilot-demo-v1"

CONFIRMATION_TYPES = ("payment.captured", "subscription.charged")


def derive_secret() -> str:
    """Mirror engine.webhook._derive_secret for demo-mode signing."""
    mode = os.environ.get("WEBHOOK_MODE", "demo").lower()
    secret = os.environ.get("WEBHOOK_SECRET", "")
    if mode == "production":
        return secret
    return hashlib.sha256((DEMO_SALT + secret).encode()).hexdigest()


def compute_signature(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _post(base_url: str, payload: dict) -> dict:
    """POST /api/webhooks/payment with a demo HMAC signature."""
    body = json.dumps(payload).encode("utf-8")
    sig = compute_signature(body, derive_secret())
    req = Request(
        f"{base_url}/api/webhooks/payment",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Signature": sig,
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except URLError as e:
        detail = ""
        if getattr(e, "read", None):
            try:
                detail = json.loads(e.read().decode()).get("detail", "")
            except Exception:
                detail = str(e)
        return {"error": detail or str(e)}


def send_failure(base_url: str, event_id: str, transaction_id: str,
                 event_type: str, amount: int, retry_count: int = 0) -> dict:
    payload = {
        "event_type": event_type,
        "event_id": event_id,
        "transaction_id": transaction_id,
        "amount": amount,
        "currency": "INR",
        "decline_code": random.choice(DECLINE_CODES),
        "retry_count": retry_count,
        "notes": {"simulated": True},
    }
    return _post(base_url, payload)


def send_confirmation(base_url: str, event_id: str, transaction_id: str,
                      amount: int, event_type: str = "payment.captured") -> dict:
    """DEMO-labelled trusted confirmation for a transaction."""
    payload = {
        "event_type": event_type,
        "event_id": event_id,
        "transaction_id": transaction_id,
        "amount": amount,
        "currency": "INR",
        "status": "captured",
        "notes": {"simulated": True, "demo_confirmation": True},
    }
    return _post(base_url, payload)


def describe(result: dict) -> str:
    if "error" in result:
        return f"✗ {result['error']}"
    outcome = result.get("outcome", result.get("status", "?"))
    action = result.get("action", "none")
    verdict = result.get("policy_verdict", "?")
    recovered = result.get("amount_recovered", 0)
    attempt = result.get("attempt")
    max_steps = result.get("max_steps")
    seq = f" attempt={attempt}/{max_steps}" if attempt is not None else ""
    return (f"→ {result.get('status', outcome):20s} outcome={outcome:12s} "
            f"action={action:25s} verdict={verdict:8s} "
            f"₹{recovered // 100:,}{seq}")


def main():
    parser = argparse.ArgumentParser(description="Recovery Copilot Real-Time Simulator")
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between events (default: 3)")
    parser.add_argument("--count", type=int, default=0, help="Number of transactions to send (0 = infinite)")
    parser.add_argument("--base-url", type=str, default="http://localhost:8321", help="API base URL")
    parser.add_argument("--confirm-rate", type=float, default=0.7,
                        help="Probability a failed payment later receives a DEMO confirmation (default: 0.7)")
    parser.add_argument("--closed-loop", action="store_true",
                        help="Send follow-up payment.failed webhooks for unconfirmed transactions (bounded re-optimization)")
    args = parser.parse_args()

    mode = os.environ.get("WEBHOOK_MODE", "demo").lower()
    if mode == "production":
        raise SystemExit(
            "Refusing to simulate confirmations/recovery against WEBHOOK_MODE=production. "
            "Run the server in demo mode for this simulator."
        )

    print("Recovery Copilot — Live Recovery Simulator (webhook ingress)")
    print(f"Target: {args.base_url}/api/webhooks/payment")
    print(f"Interval: {args.interval}s   Confirm rate: {args.confirm_rate}   "
          f"Closed-loop: {args.closed_loop}")
    print(f"Count: {'infinite' if args.count == 0 else args.count}")
    print(f"{'─' * 70}")

    sent = 0
    confirmed = 0
    max_retry_rounds = 3
    try:
        while args.count == 0 or sent < args.count:
            event_type, amounts = random.choice(EVENT_TYPES)
            amount = random.choice(amounts)
            tx_id = f"live_sim_{sent + 1:04d}"
            event_id = f"evt_sim_{sent + 1:04d}"
            now = datetime.now().strftime("%H:%M:%S")

            result = send_failure(args.base_url, event_id, tx_id, event_type, amount)
            print(f"[{now}] {event_type:28s} ₹{amount:>10,} {describe(result)}")
            sent += 1

            if "error" in result:
                continue

            if random.random() < args.confirm_rate:
                time.sleep(args.interval)
                conf = send_confirmation(args.base_url, event_id, tx_id, amount)
                confirmed += 1
                if "error" in conf:
                    print(f"[{datetime.now():%H:%M:%S}] *DEMO* payment.captured ₹{amount // 100:,} ✗ "
                          f"{conf['error']}")
                else:
                    print(f"[{datetime.now():%H:%M:%S}] *DEMO* payment.captured   ₹{amount:>10,} "
                          f"{describe(conf)}")
            elif args.closed_loop:
                round_no = 1
                while round_no < max_retry_rounds:
                    time.sleep(args.interval)
                    retry = send_failure(
                        args.base_url,
                        f"evt_sim_{sent}_r{round_no}",
                        tx_id, event_type, amount, retry_count=round_no,
                    )
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"[{now}] {event_type:28s} ₹{amount:>10,} (retry #{round_no}) "
                          f"{describe(retry)}")
                    round_no += 1
                    if "error" in retry or retry.get("outcome") in ("max_steps", "blocked"):
                        break
                    if random.random() < args.confirm_rate:
                        conf = send_confirmation(args.base_url, event_id, tx_id, amount)
                        confirmed += 1
                        print(f"[{datetime.now():%H:%M:%S}] *DEMO* payment.captured   ₹{amount:>10,} "
                              f"{describe(conf)}")
                        break

            if args.count == 0 or sent < args.count:
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n{'─' * 70}")
        print(f"Simulator stopped after {sent} transactions.")

    print(f"\nTotal transactions: {sent}   DEMO confirmations: {confirmed}")


if __name__ == "__main__":
    main()