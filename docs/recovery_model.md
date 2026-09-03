# Recovery Model — Production Integration Spec

This document describes the production integration of the trained recovery
model (`recovery_model_final.ipynb`) into the Recovery Copilot FastAPI backend.
It is the operational twin of the research notebook: the notebook trains, this
service serves. Nothing here retrains, refits, or re-executes the notebook.

## Scope

- **Goal:** for a refunded/cancelled/pending transaction, produce a
  calibrated probability that the same customer recovers within 72 hours,
  plus a threshold decision and a risk band.
- **Target:** `recovered_72h` (1 = recovered within 72h).
- **Model:** `ExtraTreesClassifier`, 500 estimators, `class_weight=balanced`.
- **Training population:** recovery-population events only (refunded,
  cancelled, pending). Requests outside that population are rejected at the
  API boundary (`RecoveryPredictRequest.status` validator).

## Inference pipeline

```
transaction
  -> 44 raw features              app.recovery.feature_builder.build_raw_features
  -> previous_recovery_date       datetime -> days since ref_date (NOT before imputation)
  -> imputer (median)             43 numeric features: NaN -> median (imputer.joblib)
  -> OneHotEncoder                payment_method -> 6 columns (encoder.joblib)
  -> 49 encoded features          column order == model_feature_names.json
  -> ExtraTreesClassifier         predict_proba[:, 1] -> raw probability
  -> sigmoid calibrator           LogisticRegression (logit link) -> calibrated p
  -> threshold 0.04               recovered iff calibrated p >= 0.04
  -> risk label + band            application-level taxonomy (below)
```

### Encoded feature layout

| Block                | Count | Notes                                        |
|----------------------|-------|----------------------------------------------|
| Base                 | 5     | quantity, prices, discount, shipping (numeric) |
| Historical           | 10    | cumulative counts via `shift(1)`             |
| Behavioral           | 9     | rolling windows `closed="left"`, streaks     |
| Advanced             | 5     | recency/momentum final pass                  |
| Temporal             | 6     | year/month/day/dayofweek/hour/is_weekend     |
| Recovery History     | 8     | strictly-prior recovery-population rows only |
| payment_method onehot | 6    | apple_pay, bank_transfer, credit_card, debit_card, google_pay, paypal |
| **Total encoded**    | **49**| order pinned by `model_feature_names.json`   |

`payment_method` is the only categorical; everything else is numeric. The
raw count of 44 includes `payment_method`; the 49 encoded features are the
43 numeric features + 6 one-hot columns.

## Leakage safety

Every temporal feature is computed **before** the current transaction:

- Cumulative historical counts use `groupby(customer).cumsum().shift(1)`.
- Rolling windows use `closed="left"` (current row never in its own window).
- Streaks are `*_streak_before` (run-length immediately preceding the row).
- Recovery-history features (`customer_recoveries_before`,
  `customer_recovery_rate_before`, `prev_*`, `recoveries_last_30d/90d`,
  `recovery_streak_before`) come only from the customer's PRIOR
  recovery-population rows (refunded/cancelled/pending) strictly before the
  current transaction. `recovered_72h` for prior rows is supplied by the caller
  and defaults to `0` (conservative) when unknown.

This is verified by `tests/test_recovery_model.py::test_13_no_future_information_leakage`.

## Artifacts

| Artifact                  | File                          | Role                                  |
|---------------------------|-------------------------------|---------------------------------------|
| Model                     | `final_model.joblib`          | ExtraTreesClassifier (n_features_in_=49) |
| Imputer                   | `imputer.joblib`              | median imputation, 43 features        |
| Encoder                   | `encoder.joblib`              | payment_method OneHot (6 categories)  |
| Calibrator                | `sigmoid_calibrator.joblib`   | LogisticRegression over raw prob      |
| Feature lists             | `feature_lists.json`          | ALL/BASE/HISTORICAL/… feature names   |
| Encoded feature order     | `model_feature_names.json`    | 49 column names, pinning encode order |
| Reference date            | `ref_date.json`               | `"2023-01-01 11:58:01"`               |
| Metrics + threshold       | `final_metrics.json`          | `selected_threshold` = 0.04 + test metrics |
| Explanations              | `feature_importance.csv`      | static impurity importance (top-20)   |

The service asserts loading invariants (44 raw, 49 encoded,
`model.n_features_in_ == 49`, 43 imputer stats, encoder categories matching
the last 6 model feature names) and fails loudly on any mismatch rather than
serving a silently-inconsistent pipeline (`RecoveryModelService.load`).

## Calibration & decision

- Raw model output is the ExtraTrees `predict_proba` positive class.
- The calibrator maps raw probability → calibrated probability via the saved
  logistic regression (coef ≈ 3.01, intercept ≈ -3.43).
- `recovery_prediction = 1` iff calibrated probability ≥ `threshold` (0.04).
- Risk **band** (application taxonomy, not trained):

  | Range        | Band        |
  |--------------|-------------|
  | < 1%         | <1%         |
  | 1–2%         | 1-2%        |
  | 2–3%         | 2-3%        |
  | 3–4%         | 3-4%        |
  | 4–5%         | 4-5%        |
  | 5–7.5%       | 5-7.5%      |
  | 7.5–10%      | 7.5-10%     |
  | ≥ 10%        | >=10%       |

- Risk **label**: Very Low < 2%, Low < 4%, Medium < 7.5%, High < 10%,
  Very High ≥ 10%.

## API

Router prefix `/api/recovery` (included in `app.main`).

| Method | Path               | Description                                       |
|--------|--------------------|---------------------------------------------------|
| GET    | `/api/recovery/model-info` | static model metadata + test metrics       |
| GET    | `/api/recovery/metrics`    | raw `final_metrics.json`, `selected_threshold` |
| GET    | `/api/recovery/features`   | 44 raw / 49 encoded, groups, encoded names   |
| POST   | `/api/recovery/predict`    | score one transaction (persists a row)       |
| POST   | `/api/recovery/batch-predict` | score many transactions (no persistence) |
| GET    | `/api/recovery/predictions`| last 20 persisted predictions               |

`POST /api/recovery/predict` body:

```json
{
  "transaction_id": "txn_123",
  "customer_id": "cust_9",
  "transaction_date": "2024-10-10 10:00:00",
  "quantity": 1,
  "unit_price": 1200,
  "total_amount": 1200,
  "discount_applied": 0,
  "shipping_cost": 0,
  "payment_method": "credit_card",
  "status": "refunded",
  "customer_signup_date": "2022-01-01",
  "history": [
    {"transaction_date": "2024-08-01 09:00:00", "status": "refunded",
     "total_amount": 500, "recovered_72h": 1}
  ]
}
```

`history` is the customer's PRIOR transactions (the model derives all
historical/behavioral/recovery features from it). `recovered_72h` is only
meaningful for recovery-population rows; omit or set 0 when unknown.

Response (per transaction): `recovery_probability` (calibrated),
`probability_raw`, `threshold`, `recovery_prediction` (0/1), `recovery_risk`,
`risk_band`, `model`, `model_artifact`, `calibrated`, `timestamp`, and
`explanation` (top-5 features by saved impurity importance with their values).

## Persistence

Single predictions are persisted best-effort to the `recovery_predictions`
table (`app.database.record_prediction`, `get_recent_predictions`). A database
failure never fails inference: the API returns the prediction and only skips
the persistence step.

## Performance (artifacts, test split)

| Metric          | Value   |
|-----------------|---------|
| ROC-AUC         | 0.5795  |
| PR-AUC          | varies  |
| Brier (cal)     | 0.0365  |
| LogLoss (cal)   | varies  |
| Precision       | 0.0610  |
| Recall          | 0.1433  |
| Lift            | 1.603   |
| Threshold       | 0.04    |

Baseline recovery rate on the training population is the rate implied by
`final_metrics.json` (`baseline_recovery_rate`). Values shown here are the
authoritative artifact values (the trained model is a weak-but-positive lift
model; the threshold is tuned for catch-rate over precision).

## Testing

`backend/tests/test_recovery_model.py` pins the 16-point integration contract
(artifact load, raw/encoded shapes and order, categorical encoding, date
conversion, imputation, unknown-method robustness, probability validity,
threshold rule, `/predict` + `/batch-predict`, leakage safety, existing-app
intactness, persistence, risk taxonomy). Run:

```bash
cd backend
PYTHONPATH=. python3 -m pytest tests/test_recovery_model.py -v
```

## Operations

- **Cold start:** the first `/api/recovery/*` request loads the 102 MB
  `final_model.joblib` once; subsequent requests reuse the in-process
  singleton (`get_model_service`).
- **Singletons & locks:** load is guarded by a thread lock; predict paths are
  read-only after load.
- **No retraining:** deploying a new model = replacing files in
  `recovery_model_artifacts/` and restarting. `main.py` imports the router
  lazily (no model load at import time).
- **Limitations:** trained on the focused research population; expected lift
  over baseline is modest (≈1.6x). The model is advisory — the Policy Engine
  remains the action authority in the broader product.