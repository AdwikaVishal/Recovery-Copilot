import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from app.models import (
    RevenueEvent, Customer, TransactionMetadata,
    EventType, DeclineCode
)

FIRST_NAMES = [
    "Rahul", "Priya", "Amit", "Sneha", "Vikram", "Pooja", "Arjun", "Meera",
    "Rohit", "Ananya", "Karan", "Deepa", "Saurabh", "Nisha", "Aditya",
    "Kavita", "Nikhil", "Shreya", "Varun", "Tanvi", "Gaurav", "Ritu",
    "Manish", "Pallavi", "Sachin", "Divya", "Tarun", "Simran", "Aakash", "Neha"
]

LAST_NAMES = [
    "Sharma", "Patel", "Kumar", "Singh", "Gupta", "Reddy", "Nair", "Joshi",
    "Mishra", "Tiwari", "Verma", "Choudhary", "Kapoor", "Mehta", "Iyer",
    "Desai", "Mukherjee", "Chauhan", "Rao", "Pandey"
]

BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "PNB", "BoB", "Yes Bank"]

CARD_PREFIXES = ["4000", "5000", "6000"]

CART_ITEMS = [
    "Premium Plan (Annual)", "Business Suite", "Enterprise License",
    "Pro Subscription (Monthly)", "Team Plan", "Starter Pack"
]


def _generate_customer(id_num: int) -> Customer:
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    phone = f"+91{random.randint(6000000000, 9999999999)}"
    email = f"{first.lower()}.{last.lower()}{id_num}@example.com"
    return Customer(
        id=f"cust_{id_num:03d}",
        name=f"{first} {last}",
        phone=phone,
        email=email,
        language_pref=random.choice(["hi", "en"]),
        opted_out=random.random() < 0.05
    )


def _card_decline(id_num: int, customer: Customer, when: datetime, force_discount: bool = False) -> RevenueEvent:
    code = random.choices(
        [DeclineCode.INSUFFICIENT_FUNDS, DeclineCode.EXPIRED_CARD,
         DeclineCode.DO_NOT_HONOR, DeclineCode.BANK_TIMEOUT],
        weights=[37, 20, 25, 18]
    )[0]

    amount = random.choice([
        50000, 100000, 150000, 250000, 500000, 750000, 99900, 149900
    ])

    discount_hint = None
    if force_discount:
        # 5% discount on high-value: passes discount_ceiling (5<=10) but
        # fails amount_ceiling (>₹5000 + discount>0) → HUMAN_REVIEW
        discount_hint = 5
    elif amount >= 500000 and code in (DeclineCode.INSUFFICIENT_FUNDS, DeclineCode.DO_NOT_HONOR) and random.random() < 0.15:
        discount_hint = 5

    return RevenueEvent(
        id=f"txn_{id_num:03d}",
        type=EventType.CARD_PAYMENT_FAILURE,
        customer=customer,
        amount=amount,
        root_cause=code,
        decline_code=code,
        failed_at=when,
        metadata=TransactionMetadata(
            card_last4=str(random.randint(1000, 9999)),
            bank=random.choice(BANKS),
            discount_hint=discount_hint,
        ),
        ground_truth=random.choices(
            ["recoverable", "not_recoverable"],
            weights=[65, 35]
        )[0]
    )


def _mandate_failure(id_num: int, customer: Customer, when: datetime) -> RevenueEvent:
    is_high_value = random.random() < 0.4
    amount = random.randint(1600000, 5000000) if is_high_value else random.randint(100000, 1400000)

    code = DeclineCode.MANDATE_AFA_REQUIRED if is_high_value else DeclineCode.MANDATE_SIMPLE_RETRY

    return RevenueEvent(
        id=f"txn_{id_num:03d}",
        type=EventType.RECURRING_PAYMENT_FAILURE,
        customer=customer,
        amount=amount,
        root_cause=code,
        decline_code=code,
        failed_at=when,
        metadata=TransactionMetadata(
            mandate_id=f"MD{random.randint(10000, 99999)}",
            subscription_id=f"sub_{random.randint(100, 999)}",
            bank=random.choice(BANKS)
        ),
        ground_truth=random.choices(
            ["recoverable", "not_recoverable"],
            weights=[55 if is_high_value else 70, 45 if is_high_value else 30]
        )[0]
    )


def _checkout_abandon(id_num: int, customer: Customer, when: datetime, force_discount: bool = False) -> RevenueEvent:
    cart_value = random.choice([99900, 199900, 499900, 999900, 1999900])

    discount_hint = None
    if force_discount or (cart_value >= 499900 and random.random() < 0.35):
        # 15% discount: fails discount_ceiling (15>10) → MODIFY
        discount_hint = 15

    return RevenueEvent(
        id=f"txn_{id_num:03d}",
        type=EventType.CHECKOUT_ABANDONMENT,
        customer=customer,
        amount=cart_value,
        root_cause=DeclineCode.PAYMENT_LINK_EXPIRED,
        decline_code=DeclineCode.PAYMENT_LINK_EXPIRED,
        failed_at=when,
        metadata=TransactionMetadata(
            cart_value=cart_value,
            discount_hint=discount_hint,
        ),
        ground_truth=random.choices(
            ["recoverable", "not_recoverable"],
            weights=[40, 60]
        )[0]
    )


def _overdue_invoice(id_num: int, customer: Customer, when: datetime) -> RevenueEvent:
    days = random.choices([7, 15, 30, 45, 60, 90], weights=[20, 20, 25, 15, 10, 10])[0]
    amount = random.choice([
        500000, 1000000, 2500000, 5000000, 7500000, 10000000
    ])

    return RevenueEvent(
        id=f"txn_{id_num:03d}",
        type=EventType.OVERDUE_INVOICE,
        customer=customer,
        amount=amount,
        root_cause=DeclineCode.INVOICE_OVERDUE,
        decline_code=DeclineCode.INVOICE_OVERDUE,
        failed_at=when - timedelta(days=days),
        metadata=TransactionMetadata(
            invoice_id=f"inv_{random.randint(1000, 9999)}",
            days_overdue=days
        ),
        ground_truth=random.choices(
            ["recoverable", "not_recoverable", "uncertain"],
            weights=[50 if days < 30 else 30, 30 if days < 30 else 50, 20]
        )[0]
    )


def generate_batch(size: int = 100, seed: int = 42) -> list[RevenueEvent]:
    random.seed(seed)
    events = []
    now = datetime.utcnow()

    distribution = {
        "card_payment_failure": int(size * 0.40),
        "recurring_payment_failure": int(size * 0.25),
        "checkout_abandonment": int(size * 0.20),
        "overdue_invoice": int(size * 0.15),
    }

    # Pre-seed specific events to guarantee all 4 policy verdicts appear.
    # Card: 1-40, Recurring: 41-65, Checkout: 66-85, Overdue: 86-100
    _human_review_ids = {5, 15, 35}          # card events, high value + 5% discount → HUMAN_REVIEW
    _modify_ids = {70, 75, 82}               # checkout events, 15% discount → MODIFY
    _deny_retry_ids = {20, 30}               # card events, retry_count=3 → max_retries DENY
    _deny_optout_ids = {38, 42}              # opted_out=True → opt_out DENY

    id_counter = 1
    for event_type, count in distribution.items():
        for _ in range(count):
            customer = _generate_customer(id_counter)
            days_ago = random.randint(0, 14)
            when = now - timedelta(days=days_ago, hours=random.randint(0, 23))

            if id_counter in _deny_optout_ids:
                customer.opted_out = True

            if id_counter in _deny_retry_ids:
                customer.opted_out = False
                retry = 3
            else:
                retry = 0

            force_discount_hr = id_counter in _human_review_ids
            force_discount_mod = id_counter in _modify_ids

            if event_type == "card_payment_failure":
                event = _card_decline(id_counter, customer, when, force_discount=force_discount_hr)
            elif event_type == "recurring_payment_failure":
                event = _mandate_failure(id_counter, customer, when)
            elif event_type == "checkout_abandonment":
                event = _checkout_abandon(id_counter, customer, when, force_discount=force_discount_mod)
            else:
                event = _overdue_invoice(id_counter, customer, when)

            event.retry_count = retry

            # Force high amount for HUMAN_REVIEW events to exceed ₹5,000 ceiling
            if force_discount_hr and event.amount <= 500000:
                event.amount = random.choice([750000, 1000000, 1500000])

            events.append(event)
            id_counter += 1

    random.shuffle(events)
    return events


def save_batch(events: list[RevenueEvent], path: str = None):
    if path is None:
        path = Path(__file__).parent / "sample_batch.json"
    data = [e.model_dump(mode="json") for e in events]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


if __name__ == "__main__":
    events = generate_batch(100)
    path = save_batch(events)
    print(f"Generated {len(events)} events → {path}")

    types = {}
    for e in events:
        types[e.type.value] = types.get(e.type.value, 0) + 1
    print("Distribution:", json.dumps(types, indent=2))
