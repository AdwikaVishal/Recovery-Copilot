"""Recovery ML model integration — acceptance tests (Model Integration feature).

The ExtraTrees recovery model trained in ``recovery_model_final.ipynb`` is
served as a frozen production service (``app.recovery``). These tests pin the
16-point contract of the integration:

  1. Frozen artifacts load (model / imputer / encoder / calibrator / metadata)
  2. Exactly 44 raw features; keys == ALL_FEATURES in notebook order
  3. Exactly 49 encoded features
  4. Encoded column order == model_feature_names.json
  5. payment_method one-hot exactly matches the saved encoder's categories
  6. previous_recovery_date converted from datetime to days-since-ref-date
  7. NaN row values (missing history) are imputed — no NaNs after encode
  8. Unknown payment method does not crash (all-zero one-hot)
  9. Raw and calibrated probabilities are valid [0, 1]
 10. Threshold decision rule: recovered iff calibrated prob >= threshold
 11. POST /api/recovery/predict works end-to-end through FastAPI
 12. POST /api/recovery/batch-predict works for multiple (incl. unknown pm)
 13. No future-information leakage in recovery-history features
 14. Existing application functionality stays intact (/api/health, startup)
 15. Predictions are persisted and retrievable (recovery_predictions table)
 16. Risk bands/labels follow the documented application thresholds
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["WEBHOOK_MODE"] = "demo"
os.environ["WEBHOOK_ALLOW_UNSIGNED"] = "false"

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import asyncio  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.recovery.model_service import (  # noqa: E402
    get_model_service,
    risk_band,
    risk_label,
)
from app.recovery.feature_builder import (  # noqa: E402
    build_raw_features,
    ALL_FEATURES,
    BASE_FEATURES,
    HISTORICAL_FEATURES,
    BEHAVIORAL_FEATURES,
    ADVANCED_FEATURES,
    TEMPORAL_FEATURES,
    RECOVERY_HISTORY_FEATURES,
)

# Load artifacts once per test session (single 102 MB model read).
SVC = get_model_service()

EXPECTED_PAYMENT_METHODS = [
    "apple_pay", "bank_transfer", "credit_card", "debit_card", "google_pay", "paypal",
]


def _payload(transaction_id: str = "txn_test", **overrides) -> dict:
    p = {
        "transaction_id": transaction_id,
        "customer_id": "cust_test",
        "transaction_date": "2024-10-10 10:00:00",
        "quantity": 1,
        "unit_price": 1200,
        "total_amount": 1200,
        "discount_applied": 0,
        "shipping_cost": 0,
        "payment_method": "credit_card",
        "status": "refunded",
        "customer_signup_date": "2022-01-01",
        "history": [],
    }
    p.update(overrides)
    return p


def _raw_df(payload: dict) -> pd.DataFrame:
    from app.recovery.schemas import RecoveryPredictRequest
    raw = build_raw_features(RecoveryPredictRequest(**payload))
    return pd.DataFrame([raw]).reindex(columns=ALL_FEATURES)


def _encoded(payload: dict) -> pd.DataFrame:
    return SVC.encode(_raw_df(payload))


# ---------------------------------------------------------------------------
# 1–4. Artifacts, raw & encoded shapes, order
# ---------------------------------------------------------------------------

def test_1_artifacts_load_consistently():
    svc = SVC
    assert svc.model is not None
    assert svc.imputer is not None
    assert svc.encoder is not None
    assert svc.calibrator is not None
    assert svc.model.n_features_in_ == 49
    assert svc.model.__class__.__name__ == "ExtraTreesClassifier"
    assert len(svc.imputer.statistics_) == 43
    assert list(svc.encoder.categories_[0]) == EXPECTED_PAYMENT_METHODS
    assert svc.selected_threshold == pytest.approx(0.04)
    assert svc.ref_date == pd.Timestamp("2023-01-01 11:58:01") or svc.ref_date.year == 2023
    assert svc.model_feature_names.__len__() == 49


def test_2_exactly_44_raw_features_match_all_features():
    assert len(ALL_FEATURES) == 44
    assert len(SVC.feature_lists["ALL_FEATURES"]) == 44
    assert sorted(ALL_FEATURES) == sorted(SVC.feature_lists["ALL_FEATURES"])
    raw = build_raw_features_raw(_payload())
    assert raw.keys().__len__() == 44
    assert sorted(raw.keys()) == sorted(ALL_FEATURES)
    assert len(set(raw.keys())) == 44


def build_raw_features_raw(payload: dict) -> dict:
    from app.recovery.schemas import RecoveryPredictRequest
    return build_raw_features(RecoveryPredictRequest(**payload))


def test_3_exactly_49_encoded_features():
    enc = _encoded(_payload())
    assert enc.shape == (1, 49)


def test_4_encoded_order_matches_model_feature_names():
    enc = _encoded(_payload())
    assert list(enc.columns) == SVC.model_feature_names
    spec = SVC.features_spec()
    assert spec["raw_feature_count"] == 44
    assert spec["encoded_feature_count"] == 49
    assert spec["encoded_feature_names"] == SVC.model_feature_names


# ---------------------------------------------------------------------------
# 5–8. Categorical encoding, date conversion, imputation, robustness
# ---------------------------------------------------------------------------

def test_5_payment_method_onehot_matches_encoder():
    enc = _encoded(_payload(payment_method="credit_card"))
    onehot = list(enc.columns[43:])
    assert sorted(onehot) == sorted(f"payment_method_{m}" for m in EXPECTED_PAYMENT_METHODS)
    assert float(enc["payment_method_credit_card"].iloc[0]) == 1.0
    assert float(enc["payment_method_paypal"].iloc[0]) == 0.0
    assert SVC.model_feature_names[43:] == onehot


def test_6_previous_recovery_date_datetime_to_days_since_ref():
    history = [
        {"transaction_id": "h1", "transaction_date": "2024-08-01 09:00:00",
         "status": "refunded", "total_amount": 500, "recovered_72h": 1},
    ]
    raw = build_raw_features_raw(_payload(history=history))
    assert isinstance(raw["previous_recovery_date"], pd.Timestamp)
    days = (raw["previous_recovery_date"] - SVC.ref_date).total_seconds() / 86400
    assert days > 0

    enc = SVC.encode(pd.DataFrame([raw]).reindex(columns=ALL_FEATURES))
    val = float(enc["previous_recovery_date"].iloc[0])
    assert np.isfinite(val)
    assert val == pytest.approx(days)


def test_7_imputation_no_nans_after_encode():
    enc = _encoded(_payload(customer_signup_date=None, history=[]))
    assert not enc.isna().any().any()
    assert not np.isnan(enc.to_numpy()).any()


def test_8_unknown_payment_method_does_not_crash():
    enc = _encoded(_payload(payment_method="upi"))
    assert enc.shape == (1, 49)
    assert not enc.isna().any().any()
    onehot = [c for c in enc.columns if c.startswith("payment_method_")]
    assert float(enc[onehot].to_numpy().sum()) == 0.0


# ---------------------------------------------------------------------------
# 9–10. Probabilities and threshold decision
# ---------------------------------------------------------------------------

def test_9_probabilities_in_unit_interval():
    enc = _encoded(_payload())
    raw, cal = SVC.predict_proba(enc)
    assert 0.0 <= float(raw[0]) <= 1.0
    assert 0.0 <= float(cal[0]) <= 1.0


def test_10_threshold_decision_rule():
    enc = _encoded(_payload())
    raw, cal = SVC.predict_proba(enc)
    decided = SVC.decide(cal)
    assert decided[0] == (1 if float(cal[0]) >= SVC.selected_threshold else 0)


# ---------------------------------------------------------------------------
# 11–13. API endpoints and leakage safety
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_db(tmp_path):
    from app.database import set_active_db_path, init_db, reset_active_db_path
    path = Path(tmp_path) / "test.db"
    token = set_active_db_path(path)
    asyncio.run(init_db())
    yield path
    reset_active_db_path(token)


@pytest.fixture()
def api(isolated_db):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        yield client


def test_11_predict_endpoint(api):
    r = api.post("/api/recovery/predict", json=_payload("txn_api1"))
    assert r.status_code == 200
    j = r.json()
    for key in ("transaction_id", "recovery_probability", "probability_raw",
                "threshold", "recovery_prediction", "recovery_risk", "risk_band"):
        assert key in j
    assert j["transaction_id"] == "txn_api1"
    assert 0.0 <= j["recovery_probability"] <= 1.0
    assert j["recovery_prediction"] == (1 if j["recovery_probability"] >= j["threshold"] else 0)
    assert j["calibrated"] is True
    assert isinstance(j["explanation"], list) and len(j["explanation"]) <= 5


def test_12_batch_predict_endpoint(api):
    body = {
        "transactions": [
            _payload("txn_b1"),
            _payload("txn_b2", payment_method="upi", status="pending"),
            _payload("txn_b3", payment_method="paypal", status="cancelled",
                     transaction_date="2024-11-05 20:00:00"),
        ]
    }
    r = api.post("/api/recovery/batch-predict", json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["count"] == 3
    assert [p["transaction_id"] for p in j["predictions"]] == ["txn_b1", "txn_b2", "txn_b3"]
    assert all(p["recovery_prediction"] in (0, 1) for p in j["predictions"])


def test_13_no_future_information_leakage():
    from app.recovery.schemas import RecoveryPredictRequest

    # Two genuine prior recovery-population rows (one shared date, different id)
    # plus one row AFTER the current transaction that must NOT be counted.
    history = [
        {"transaction_id": "h_a", "transaction_date": "2024-05-01 09:00:00",
         "status": "refunded", "total_amount": 400, "recovered_72h": 1},
        {"transaction_id": "h_b", "transaction_date": "2024-06-10 09:00:00",
         "status": "pending", "total_amount": 600, "recovered_72h": 1},
        {"transaction_id": "h_future", "transaction_date": "2024-06-12 09:00:00",
         "status": "refunded", "total_amount": 900, "recovered_72h": 1},
    ]
    req = RecoveryPredictRequest(
        **_payload("cur_txn", transaction_date="2024-06-10 10:00:00", history=history))
    raw = build_raw_features(req)

    assert raw["customer_transactions_before"] == 2      # strictly-prior rows only
    assert raw["customer_recoveries_before"] == 2        # future row excluded
    assert raw["customer_recovery_rate_before"] == 1.0
    assert raw["recovery_streak_before"] == 2
    assert raw["recoveries_last_30d"] == 1               # h_b within 30d of current
    assert raw["recoveries_last_90d"] == 2
    assert np.isfinite(raw["days_since_previous_recovery"])
    assert raw["days_since_previous_recovery"] == pytest.approx(
        (pd.Timestamp("2024-06-10 10:00:00") - pd.Timestamp("2024-06-10 09:00:00")).total_seconds() / 86400)

    # recovered_72h omitted => conservative 0 (never counted as a recovery).
    history2 = [{"transaction_id": "h_c", "transaction_date": "2024-05-01 09:00:00",
                 "status": "refunded", "total_amount": 400}]  # no recovered_72h
    raw2 = build_raw_features(
        RecoveryPredictRequest(**_payload("cur2", history=history2)))
    assert raw2["customer_transactions_before"] == 1
    assert raw2["customer_recoveries_before"] == 0
    assert raw2["customer_recovery_rate_before"] == 0.0


# ---------------------------------------------------------------------------
# 14–16. Existing app intact, persistence, risk taxonomy
# ---------------------------------------------------------------------------

def test_14_existing_app_health_intact(api):
    r = api.get("/api/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_15_predictions_persisted_and_retrievable(api):
    r = api.post("/api/recovery/predict", json=_payload("txn_persist"))
    assert r.status_code == 200
    got = api.get("/api/recovery/predictions")
    assert got.status_code == 200
    rows = got.json()
    assert len(rows) >= 1
    assert any(x["transaction_id"] == "txn_persist" for x in rows)
    row = next(x for x in rows if x["transaction_id"] == "txn_persist")
    assert "recovery_risk" in row and "recovery_probability" in row


def test_16_risk_bands_and_labels():
    assert risk_band(0.005) == "<1%"
    assert risk_band(0.015) == "1-2%"
    assert risk_band(0.04) == "4-5%"          # 0.04 is not < 0.04 -> next band
    assert risk_band(0.09) == "7.5-10%"
    assert risk_band(0.50) == ">=10%"

    assert risk_label(0.005) == "Very Low"
    assert risk_label(0.03) == "Low"
    assert risk_label(0.04) == "Medium"
    assert risk_label(0.08) == "High"
    assert risk_label(0.30) == "Very High"


# ---------------------------------------------------------------------------
# 17. Regression: same-day (duplicate timestamp) transactions must not explode
# ---------------------------------------------------------------------------

def test_17_duplicate_timestamps_do_not_explode_feature_builder():
    """Multiple same-day transactions are realistic; the rolling-window join
    must not degenerate into a many-to-many cross-product and hang."""
    import datetime

    history = []
    for i in range(1, 13):
        dt = datetime.datetime(2024, 2, (i % 7) + 1, 10, 0)  # days 2..8 repeat
        history.append({
            "transaction_id": f"h{i}",
            "transaction_date": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "refunded" if i % 3 == 0 else ("pending" if i % 3 == 1 else "cancelled"),
            "total_amount": 800 + i * 50,
            "recovered_72h": 1 if i % 2 == 0 else 0,
        })

    payload = _payload("V_DUP", history=history)
    raw = build_raw_features_raw(payload)
    assert len(raw) == 44

    enc = SVC.encode(pd.DataFrame([raw]).reindex(columns=ALL_FEATURES))
    assert enc.shape == (1, 49)
    assert not enc.isna().any().any()

    raw_p, cal = SVC.predict_proba(enc)
    assert 0.0 <= float(raw_p[0]) <= 1.0
    assert 0.0 <= float(cal[0]) <= 1.0