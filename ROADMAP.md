# Recovery Copilot — Buildathon Roadmap

## Project Summary
An agent that detects revenue at risk, determines the right intervention, and executes bounded recovery workflows — with compliance guardrails, audit trails, and measured outcomes.

---

## Phase 0: Foundation (Day 1 — ~3 hours)

### 0.1 Project Scaffolding
```
recovery-copilot/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry
│   │   ├── models.py            # Pydantic schemas
│   │   ├── database.py          # SQLite setup
│   │   └── config.py            # Settings, API keys
│   ├── data/
│   │   ├── generator.py         # Synthetic data generator
│   │   └── sample_batch.json    # Pre-generated batch
│   ├── engine/
│   │   ├── diagnosis.py         # Root cause classifier
│   │   ├── policy.py            # Guardrail engine
│   │   ├── action.py            # Action executor
│   │   ├── messaging.py         # Hinglish message generator
│   │   ├── ptp_tracker.py       # Promise-to-Pay tracker
│   │   └── audit.py             # Audit log writer
│   ├── razorpay/
│   │   ├── client.py            # Razorpay API wrapper
│   │   └── webhooks.py          # Webhook simulator
│   └── tests/
│       ├── test_policy.py
│       └── test_diagnosis.py
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── Dashboard.tsx        # Main dashboard
│   │   ├── AuditTrail.tsx       # Decision log table
│   │   ├── BatchView.tsx        # Per-record reasoning view
│   │   └── components/
│   │       ├── RecoveryChart.tsx # Baseline vs Agent bar chart
│   │       └── GuardrailStatus.tsx
│   └── package.json
└── docs/
    └── what-broke.md            # Running failure log
```

### 0.2 Tech Stack Decisions
| Layer | Choice | Why |
|-------|--------|-----|
| Backend | Python + FastAPI | Fast to build, Razorpay SDK exists |
| LLM | Claude API (anthropic) | Best reasoning for diagnosis |
| Database | SQLite | Zero setup, sufficient for demo |
| Frontend | React + Recharts | Lightweight, chart-friendly |
| Razorpay | Test-mode SDK (razorpay-python) | Real API calls, no money moved |

### 0.3 Razorpay Test Account Setup
- Create Razorpay test account
- Generate API keys (key_id, key_secret)
- Understand test card decline codes:
  - `insufficient_funds` — card test: 4000 000 0000 0002
  - `expired_card` — card test: 4000 000 0000 0069
  - `incorrect_cvc` — card test: 4000 000 0000 0127
  - `processing_error` — card test: 4000 000 0000 0119
  - `do_not_honor` — generic bank rejection
- Understand subscription/mandate APIs in test mode
- Document webhook event formats for: `payment.failed`, `subscription.charged.failed`, `payment_link.expired`

**Deliverable:** Empty project with structure, Razorpay test keys configured, README.

---

## Phase 1: Synthetic Data + Diagnosis Engine (Day 1 — ~4 hours)

### 1.1 Synthetic Data Generator (`data/generator.py`)
Generate 100 at-risk revenue records with varied root causes:

| Category | Count | Root Causes |
|----------|-------|-------------|
| Card Declines | 40 | insufficient_funds (15), expired_card (8), do_not_honor (10), bank_timeout (7) |
| Recurring Mandate Failures | 25 | AFA_required (10 — amount > ₹15,000), simple_retry (15 — amount < ₹15,000) |
| Abandoned Checkouts | 20 | payment_link.expired, partial_form_filled |
| Overdue B2B Invoices | 15 | 7_days_overdue (5), 30_days_overdue (5), 60_days_plus (5) |

Each record schema:
```python
{
    "id": "txn_001",
    "type": "card_decline" | "mandate_failure" | "checkout_abandon" | "overdue_invoice",
    "customer": {
        "id": "cust_001",
        "name": "Rahul Sharma",
        "phone": "+919876543210",
        "email": "rahul@example.com",
        "language_pref": "hi" | "en",
        "opted_out": false
    },
    "amount": 5000,  # in paise
    "currency": "INR",
    "root_cause": "insufficient_funds",
    "decline_code": "insufficient_funds",
    "failed_at": "2026-08-20T10:30:00Z",
    "metadata": {
        "card_last4": "4242",
        "bank": "HDFC",
        "mandate_id": "MD1234",  # for recurring
        "subscription_id": "sub_001",  # for recurring
        "invoice_id": "inv_001",  # for B2B
        "cart_value": 15000,  # for abandoned checkout
        "days_overdue": 15  # for invoices
    },
    "ground_truth": "recoverable" | "not_recoverable" | "uncertain",
    "recovered_amount": 0  # filled after agent runs
}
```

### 1.2 Diagnosis Engine (`engine/diagnosis.py`)

Two-tier classifier:

**Tier 1: Rule-based (deterministic, fast)**
```python
RULES = {
    "insufficient_funds": {
        "action": "retry_with_delay",
        "reason": "Customer likely has temporary cash flow issue",
        "optimal_delay_hours": 48,
        "max_retries": 2,
        "requires_afa": False
    },
    "expired_card": {
        "action": "send_update_card_link",
        "reason": "Card expired, retry will fail again",
        "max_retries": 0,
        "requires_afa": False
    },
    "do_not_honor": {
        "action": "retry_with_delay",
        "reason": "Bank rejection, may be transient",
        "optimal_delay_hours": 24,
        "max_retries": 1,
        "requires_afa": False
    },
    "bank_timeout": {
        "action": "retry_immediate",
        "reason": "Network/bank timeout, not a real decline",
        "optimal_delay_hours": 2,
        "max_retries": 3,
        "requires_afa": False
    },
    "mandate_afa_required": {
        "action": "re_authorize",
        "reason": "RBI mandate: recurring > ₹15,000 needs fresh AFA",
        "max_retries": 0,  # Must re-authorize, can't retry
        "requires_afa": True
    },
    "mandate_simple_retry": {
        "action": "retry_with_delay",
        "reason": "Mandate failure under ₹15,000, can retry",
        "optimal_delay_hours": 24,
        "max_retries": 2,
        "requires_afa": False
    }
}
```

**Tier 2: LLM reasoning (for ambiguous cases)**
- When decline code is `do_not_honor` with repeated failures
- When customer has history of multiple decline types
- When B2B invoice has partial payment history

LLM prompt template:
```
You are a payment recovery analyst for an Indian SaaS company.

Transaction details:
- Type: {type}
- Amount: ₹{amount}
- Decline code: {decline_code}
- Customer history: {history_summary}
- Days since failure: {days_elapsed}

Classify this into one of:
1. RETRY_NOW - transient failure, safe to retry immediately
2. RETRY_DELAYED - temporary issue, retry after {suggested_delay}
3. SEND_PAYMENT_LINK - customer should complete payment via new link
4. RE_AUTHORIZE_MANDATE - RBI compliance requires fresh AFA
5. ESCALATE_TO_HUMAN - needs manual review
6. DO_NOT_CONTACT - customer opted out or dispute raised

Provide reasoning in 1-2 sentences.
```

**Deliverable:** Generator produces 100 records. Diagnosis engine classifies all 100 with confidence scores. Unit tests pass.

---

## Phase 2: Policy Engine + Guardrails (Day 2 — ~4 hours)

### 2.1 Policy Engine (`engine/policy.py`) — THE COMPLIANCE WOW FACTOR

This is deterministic, non-LLM, fully unit-testable:

```python
class PolicyEngine:
    def __init__(self, config: PolicyConfig):
        self.config = config
    
    def evaluate(self, record: RevenueEvent, proposed_action: ProposedAction) -> PolicyDecision:
        """Returns ALLOW/DENY/MODIFY with reason."""
        
        checks = [
            self._check_opt_out,
            self._check_dispute_status,
            self._check_max_retries,
            self._check_cooling_period,
            self._check_afa_threshold,
            self._check_discount_ceiling,
            self._check_contact_frequency,
            self._check_time_of_day,
        ]
        
        for check in checks:
            result = check(record, proposed_action)
            if result.verdict == "DENY":
                return result  # Stop immediately
        
        return PolicyDecision(verdict="ALLOW", reason="All checks passed")
```

**Hard Guardrails (non-negotiable):**

| # | Rule | Implementation | Why It Matters |
|---|------|----------------|----------------|
| 1 | Max retries per transaction | `retry_count < config.max_retries` | Stopping rule |
| 2 | Cooling-off period | `time_since_last_attempt > config.min_cooling_hours` | Prevents harassment |
| 3 | RBI AFA threshold | `if amount > 15000 and type == recurring: BLOCK retry, require re-auth` | RBI compliance |
| 4 | Opt-out honor | `if customer.opted_out: BLOCK all contact` | Legal compliance |
| 5 | Dispute freeze | `if dispute_raised: BLOCK all action` | Legal compliance |
| 6 | Discount ceiling | `if proposed_discount > 10%: FLAG for human approval` | Financial control |
| 7 | Contact frequency | `contacts_in_last_7_days < 3` | Decency guardrail |
| 8 | Time-of-day | `8:00 AM <= current_time <= 9:00 PM IST` | RBI communication norms |
| 9 | AFA re-auth required | `if mandate_above_15k and no_fresh_afa: DENY retry` | RBI e-mandate rule |
| 10 | Max batch recovery rate | `if recovery_rate > 80%: PAUSE for review` | Sanity check |

### 2.2 Policy Config (`config/policy.yaml`)
```yaml
max_retries_per_transaction: 3
min_cooling_hours: 24
rbi_afa_threshold_paise: 1500000  # ₹15,000
max_discount_percent: 10
max_contacts_per_week: 3
contact_window_start: "08:00"
contact_window_end: "21:00"
channels:
  priority: ["whatsapp", "sms", "email"]
  backup: ["voice"]
escalation_threshold_paise: 500000  # ₹5,000 — above this, human reviews discount
```

### 2.3 Audit Logger (`engine/audit.py`)

Every action logged:
```python
{
    "id": "audit_001",
    "timestamp": "2026-08-20T10:30:00Z",
    "txn_id": "txn_001",
    "customer_id": "cust_001",
    "action": "retry_payment",
    "reason": "decline_code: insufficient_funds, temporary cash flow issue",
    "diagnosis_confidence": 0.92,
    "policy_decision": "ALLOW",
    "policy_checks_passed": ["opt_out", "max_retries", "cooling_period", "afa_check"],
    "channel": "razorpay_api",
    "amount_attempted": 5000,
    "result": "success" | "failed" | "pending",
    "rule_version": "1.0"
}
```

**Deliverable:** Policy engine blocks 100% of opt-out cases, enforces AFA threshold, stops retries at limit. 20+ unit tests covering edge cases. Audit log captures every decision.

---

## Phase 3: Action Execution (Day 2 — ~4 hours)

### 3.1 Razorpay API Integration (`razorpay/client.py`)

Test-mode operations:
```python
class RazorpayClient:
    def retry_payment(self, payment_id: str) -> PaymentResult:
        """Retry failed payment via Razorpay test API."""
        
    def create_payment_link(self, amount: int, customer: dict) -> str:
        """Generate new payment link for abandoned checkout."""
        
    def reauthorize_mandate(self, mandate_id: str) -> str:
        """Initiate fresh AFA for high-value recurring."""
        
    def check_subscription_status(self, sub_id: str) -> SubStatus:
        """Check if subscription is still active."""
```

### 3.2 Action Router (`engine/action.py`)

Routes diagnosis → policy → execution:
```python
async def process_record(record: RevenueEvent) -> RecoveryResult:
    # Step 1: Diagnose
    diagnosis = await diagnose(record)
    
    # Step 2: Propose action
    proposed = propose_action(record, diagnosis)
    
    # Step 3: Policy check
    decision = policy_engine.evaluate(record, proposed)
    
    if decision.verdict == "DENY":
        return RecoveryResult(action="none", reason=decision.reason)
    
    # Step 4: Execute
    if proposed.action == "retry_payment":
        result = await razorpay_client.retry_payment(record.payment_id)
    elif proposed.action == "send_payment_link":
        link = await razorpay_client.create_payment_link(record.amount, record.customer)
        result = RecoveryResult(action="payment_link_sent", link=link)
    elif proposed.action == "re_authorize_mandate":
        # Don't retry — send re-auth request
        result = RecoveryResult(action="re_auth_initiated", 
                                message="RBI AFA required, customer notified")
    
    # Step 5: Audit
    audit_log.record(record, diagnosis, proposed, decision, result)
    
    return result
```

### 3.3 Batch Processor

Process entire batch sequentially, tracking cumulative results:
```python
async def process_batch(batch: List[RevenueEvent]) -> BatchResult:
    results = []
    for record in batch:
        result = await process_record(record)
        results.append(result)
    
    return BatchResult(
        total_records=len(batch),
        attempted=sum(1 for r in results if r.action != "none"),
        recovered=sum(1 for r in results if r.result == "success"),
        recovered_amount=sum(r.amount_recovered for r in results),
        blocked_by_policy=sum(1 for r in results if r.action == "none"),
        audit_trail=results
    )
```

**Deliverable:** Batch of 100 records processes end-to-end. Each record gets a diagnosis, policy check, action, and audit entry. Results stored in SQLite.

---

## Phase 4: Hinglish Messaging + Tone Ladder (Day 3 — ~3 hours)

### 4.1 Hinglish Message Generator (`engine/messaging.py`)

Tone ladder for B2B/subscription dunning:

| Stage | Tone | Template Style | Example |
|-------|------|----------------|---------|
| 1 (Day 0-1) | Friendly nudge | Casual, helpful | "Hi Rahul ji, aapka ₹5,000 ka payment pending hai. Koi issue hua tha kya? Yahan pay karein: {link}" |
| 2 (Day 2-3) | Reminder + offer | Slightly formal, partial-payment option | "Rahul ji, reminder — ₹5,000 due hai. Agar abhi pura nahi de sakte, 50% bhi chalega. Link: {link}" |
| 3 (Day 4-7) | Firm but polite | Formal, deadline | "Rahul ji, ye last reminder hai. ₹5,000 7 din se pending hai. Please aaj pay karein to avoid account suspension." |
| 4 (Day 7+) | Human handoff | System message | "Escalating to account manager. Customer unresponsive after 3 touchpoints." |

### 4.2 LLM-Powered Custom Messages

For non-template cases, use Claude:
```
Generate a WhatsApp recovery message in Hinglish (mix of Hindi and English) for:

Customer: {name}
Amount: ₹{amount}
Days overdue: {days}
Previous contact: {prev_messages_count}
Tone level: {tone_level} (1=friendly, 2=reminder, 3=firm, 4=escalation)

Rules:
- Keep under 50 words
- Include payment link placeholder {link}
- Don't threaten or use aggressive language
- If tone >= 3, mention consequences but stay professional
- Sound like a real person, not a bot
```

### 4.3 Promise-to-Pay Tracker (`engine/ptp_tracker.py`)

```python
class PTPTracker:
    def record_promise(self, customer_id: str, amount: int, promised_date: str):
        """Customer says 'I'll pay Friday' → store it."""
        
    def check_promises(self) -> List[PTPStatus]:
        """Run daily: who promised and hasn't paid?"""
        
    def handle_broken_promise(self, customer_id: str):
        """Auto-escalate tone level for this customer."""
```

State machine:
```
PROMISED → (on due date) → CHECKED → PAID (done) / BROKEN → ESCALATED → HUMAN_HANDOFF
```

**Deliverable:** Hinglish messages generated for each B2B/subscription case. PTP tracker stores promises and flags broken ones. Tone escalation works across 4 levels.

---

## Phase 5: Dashboard (Day 3 — ~4 hours)

### 5.1 Backend API Endpoints

```
GET  /api/batch/{batch_id}/summary     → Recovery stats, baseline comparison
GET  /api/batch/{batch_id}/records     → All records with status
GET  /api/record/{id}/audit-trail      → Full decision chain for one record
GET  /api/guardrails/status            → Active guardrails and trigger counts
POST /api/batch/run                    → Process a batch
GET  /api/ptp/active                   → Active promises-to-pay
```

### 5.2 Frontend Dashboard

**Screen 1: Recovery Overview**
```
┌─────────────────────────────────────────────────┐
│  Recovery Copilot — Batch #1                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ ₹2,45,000│  │   41%    │  │   12%    │      │
│  │ Recovered │  │ Recovery │  │ Baseline │      │
│  │          │  │  Rate    │  │  Rate    │      │
│  └──────────┘  └──────────┘  └──────────┘      │
│                                                  │
│  [Bar Chart: Agent vs Baseline per category]     │
│  Card Declines:   ████████████ 52% vs ███ 15%   │
│  Mandate Failures: █████████ 45% vs ██ 10%      │
│  Abandoned Cart:   ████████ 38% vs ██ 8%        │
│  B2B Invoices:     ██████ 30% vs █ 5%           │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Screen 2: Audit Trail Table**
```
┌─────────────────────────────────────────────────────────────┐
│ Filter: [All Actions ▼] [All Policies ▼] [Success/Failed ▼]│
├──────┬──────────┬────────────┬──────────┬──────────┬────────┤
│ Time │ Txn ID   │ Action     │ Policy   │ Reason   │ Result │
├──────┼──────────┼────────────┼──────────┼──────────┼────────┤
│ 10:30│ txn_001  │ RETRY      │ ALLOW    │ insuffi..│ ✅     │
│ 10:31│ txn_002  │ RE_AUTH    │ ALLOW    │ AFA req..│ 📧     │
│ 10:31│ txn_003  │ BLOCKED    │ DENY     │ Opt-out  │ ⛔     │
│ 10:32│ txn_004  │ SEND_LINK  │ ALLOW    │ Expired..│ ✅     │
└──────┴──────────┴────────────┴──────────┴──────────┴────────┘
```

**Screen 3: Record Detail View**
```
┌─────────────────────────────────────────────────┐
│ txn_001 — Rahul Sharma — ₹5,000                 │
├─────────────────────────────────────────────────┤
│ Diagnosis:                                       │
│   Decline code: insufficient_funds               │
│   Classification: Temporary cash flow issue      │
│   Confidence: 92%                                │
│   LLM reasoning: "Customer's card was charged    │
│   successfully 2 weeks ago. This is likely a     │
│   temporary balance issue."                       │
│                                                  │
│ Policy Check:                                    │
│   ✅ Not opted out                               │
│   ✅ No dispute raised                           │
│   ✅ Retry 1 of 3                                │
│   ✅ 48h cooling period elapsed                  │
│   ✅ Amount < ₹15,000 (no AFA needed)           │
│                                                  │
│ Action Taken:                                    │
│   Retry payment via Razorpay API                 │
│   Result: Success                                │
│   Amount recovered: ₹5,000                       │
│                                                  │
│ Audit Trail: [3 events]                          │
└─────────────────────────────────────────────────┘
```

**Deliverable:** Working dashboard with baseline comparison chart, audit trail table, and record detail view. API serves all data from SQLite.

---

## Phase 6: End-to-End Integration + Demo Prep (Day 4 — ~4 hours)

### 6.1 Full Pipeline Test
- Run 100-record batch end-to-end
- Verify every record has: diagnosis → policy decision → action → audit entry
- Calculate and display:
  - Total recovered amount
  - Recovery rate vs baseline
  - Policy blocks breakdown
  - Channel effectiveness

### 6.2 Baseline Calculation
```python
def calculate_baseline(batch: List[RevenueEvent]) -> BaselineResult:
    """
    Industry-standard do-nothing baseline:
    - Card declines: 5-8% auto-recovery (bank auto-retry)
    - Mandate failures: 10-15% (some succeed on next cycle)
    - Abandoned checkout: 3-5% (some come back)
    - B2B invoices: 60-70% (eventually pay, but slowly)
    """
    # Simulate conservative baseline
    baseline_recovered = sum(
        record.amount * BASELINE_RATE[record.type]
        for record in batch
    )
    return BaselineResult(recovered_amount=baseline_recovered, rate=...)
```

### 6.3 Demo Video Script (5 minutes)

| Time | What to Show | Key Point |
|------|-------------|-----------|
| 0:00-0:30 | Problem statement + one real failed payment | "Revenue degrades silently" |
| 0:30-1:30 | Diagnosis trail for 3 contrasting cases | Reasoning chain visible |
| 1:30-2:30 | AFA compliance case + policy block | RBI awareness wow factor |
| 2:30-3:30 | Hinglish message generation + PTP | Live message demo |
| 3:30-4:30 | Dashboard: baseline vs agent numbers | ₹ recovered front and center |
| 4:30-5:00 | "What broke" story + guardrail layer | Credibility through honesty |

### 6.4 "What Broke" Log

Document these real failures during the build:
1. **Initial retry logic ignored ₹15,000 AFA threshold** — mandates above ₹15,000 silently failed 30% of retries. Caught by comparing against RBI rule docs.
2. **No cooling-off period** — agent retried 3 times in 2 minutes, flagged as spam by test webhook. Added minimum 24h gap.
3. **Hinglish messages sounded robotic** — first prompt templates produced "Dear customer" style. Fixed by adding "sound like a real person, not a bot" to prompt.

**Deliverable:** Full pipeline works. Dashboard shows real numbers. Demo video recorded. "What broke" documented.

---

## Phase 7: Polish + Edge Cases (Day 4 — optional, if time)

### 7.1 Edge Cases to Handle
- Customer with multiple failed transactions (aggregate view)
- Partial payment received (track remaining)
- Currency conversion edge cases (display in ₹)
- Network timeout on Razorpay API call (graceful retry)
- Concurrent batch processing (lock mechanism)

### 7.2 Nice-to-Haves
- Export audit trail as CSV/PDF
- Email notification to human when escalation triggers
- Per-customer recovery history view
- "What if" simulator: "What would happen if we increased retries to 5?"

---

## Wow Factors Checklist

| # | Wow Factor | Phase | Status |
|---|-----------|-------|--------|
| 1 | RBI/NPCI-aware Mandate Retry Sequencer | Phase 2 (Policy) | ☐ |
| 2 | Root-cause → Action reasoning chain | Phase 1 (Diagnosis) | ☐ |
| 3 | Hinglish tone-ladder messaging | Phase 4 | ☐ |
| 4 | Promise-to-Pay tracker | Phase 4 | ☐ |
| 5 | Bounded autonomy guardrail layer | Phase 2 (Policy) | ☐ |
| 6 | Money recovered dashboard with baseline | Phase 5 (Dashboard) | ☐ |

---

## Time Estimates

| Phase | Hours | Priority |
|-------|-------|----------|
| Phase 0: Foundation | 3h | Must-have |
| Phase 1: Data + Diagnosis | 4h | Must-have |
| Phase 2: Policy + Guardrails | 4h | Must-have |
| Phase 3: Action Execution | 4h | Must-have |
| Phase 4: Hinglish + PTP | 3h | High |
| Phase 5: Dashboard | 4h | High |
| Phase 6: Integration + Demo | 4h | Must-have |
| Phase 7: Polish | 2h | Optional |
| **Total** | **~28h** | |

---

## Key Technical Decisions to Make Early

1. **LLM Provider:** Claude API (recommended for reasoning quality)
2. **State Management:** SQLite (sufficient, zero setup)
3. **Razorpay Integration:** Test-mode Python SDK
4. **Frontend Framework:** React + Vite (fast setup)
5. **Charts:** Recharts (simple, React-native)
6. **Message Templates:** Jinja2 for Hinglish templates + Claude for custom generation

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Razorpay test API rate limits | Add retry with exponential backoff, cache responses |
| LLM costs during iteration | Use rule-based first, LLM only for ambiguous cases |
| Demo fails live | Have pre-recorded batch results as fallback |
| "What broke" story too polished | Intentionally keep early failures in git history |
| Dashboard takes too long | Use CLI output as fallback for demo |
