# FINAL MODEL REPORT — E-Commerce Transaction Recovery Prediction

## DATA
- Raw transactions: 120,000
- Recovery rows (refunded / cancelled / pending): 51,300
- Raw features: 44 (6 base, 10 historical, 9 behavioral, 5 advanced, 6 temporal, 8 recovery-history)
- Encoded model features: 49 (43 numeric + converted `previous_recovery_date` + 6 one-hot `payment_method`)
- Positive recoveries (`recovered_72h`=1): 1,899
- Recovery rate: 3.7018%

## SPLIT (chronological, no shuffling)
- Train: 35,910 rows | 1,331 positives | 2023-01-01 → 2024-05-24
- Validation: 7,695 rows | 275 positives | 2024-05-24 → 2024-09-10
- Test: 7,695 rows | 293 positives | 2024-09-10 → 2024-12-30

## MODEL
- Algorithm: ExtraTreesClassifier
- Estimators: 500
- Class imbalance handling: `class_weight="balanced"`
- Selection performed on validation only; test set touched exactly once for final evaluation

## PERFORMANCE
- Validation ROC-AUC: 0.5786
- Validation PR-AUC: 0.0704
- Test ROC-AUC: 0.5795
- Test PR-AUC: 0.0686
- Test Brier (raw): 0.0366
- Test LogLoss (raw): 0.1621

## CALIBRATION (sigmoid, fit on validation only, evaluated on test)
- Raw ROC-AUC: 0.5795 → Calibrated ROC-AUC: 0.5795 (unchanged, as expected)
- Raw PR-AUC: 0.0686 → Calibrated PR-AUC: 0.0686 (unchanged, as expected)
- Raw Brier: 0.0366 → Calibrated Brier: 0.0365
- Raw LogLoss: 0.1621 → Calibrated LogLoss: 0.1606
- Ranking preservation (Spearman, raw vs. calibrated): 1.0000 ✓

## THRESHOLD (selected on validation via F1, evaluated once on test)
- Selected threshold: 0.04
- Test precision: 0.0610
- Test recall: 0.1433
- Test F1: 0.0856
- Test lift: 1.60x
- Flagged transactions: 688 (8.94% of test set)
- TP=42  FP=646  TN=6,756  FN=251
- Baseline recovery rate: 3.81% | Flagged recovery rate: 6.10%

## TOP-K TARGETING (test set, ranked by calibrated probability)
| Top-K | N     | Recoveries captured | Precision | Recall | Lift  |
|-------|-------|----------------------|-----------|--------|-------|
| 1%    | 76    | 10                   | 0.132     | 0.034  | 3.46x |
| 2%    | 153   | 15                   | 0.098     | 0.051  | 2.57x |
| 5%    | 384   | 27                   | 0.070     | 0.092  | 1.85x |
| 10%   | 769   | 46                   | 0.060     | 0.157  | 1.57x |
| 15%   | 1,154 | 67                   | 0.058     | 0.229  | 1.52x |
| 20%   | 1,539 | 87                   | 0.057     | 0.297  | 1.48x |
| 25%   | 1,923 | 103                  | 0.054     | 0.352  | 1.41x |
| 30%   | 2,308 | 117                  | 0.051     | 0.399  | 1.33x |
| 40%   | 3,078 | 146                  | 0.047     | 0.498  | 1.25x |
| 50%   | 3,847 | 180                  | 0.047     | 0.614  | 1.23x |

## EXPLAINABILITY
**Top model (impurity) importance features:** `customer_tenure_days`, `transaction_hour`, `transaction_day`, `transaction_month`, `transaction_dayofweek`

**Top permutation importance features (TEST, PR-AUC/average precision):** see `permutation_importance.csv` — importance is measured by the drop in PR-AUC when a feature is shuffled, so it reflects real predictive contribution rather than tree-split frequency.

**Top SHAP features (TreeExplainer, 200 test samples, positive class):** `transaction_month`, `transaction_year`, `days_since_previous_failure`, `historical_total_spend`, `transaction_day`

Notably, impurity-based importance is dominated by high-cardinality continuous/temporal features (a known bias of that method), while permutation and SHAP importance — both computed against actual predictive contribution — surface a more varied mix including recovery-history and behavioral features. This divergence is itself a useful diagnostic and is preserved rather than papered over.

## HONEST READ ON MODEL QUALITY
PR-AUC of ~0.069 against a 3.7% base rate means the model has modest but real signal (roughly 1.6–3.5x lift depending on how aggressively you target), not strong discriminative power. Recovery within 72h of a refund/cancellation/pending transaction appears to be driven substantially by factors not captured in this feature set (payment-processor timing, external retry logic, customer-service intervention, etc.). The model is usable for prioritization at scale (e.g., "call the top 5–10% first") but should not be treated as a confident per-transaction predictor.

## ARTIFACTS SAVED (`recovery_model_artifacts/`)
- `final_model.joblib` — trained ExtraTreesClassifier (compressed)
- `imputer.joblib` — median imputer, fit on train only
- `encoder.joblib` — one-hot encoder for `payment_method`, fit on train only
- `sigmoid_calibrator.joblib` — logistic calibrator, fit on validation only
- `feature_lists.json` — the 44 raw feature names by group
- `model_feature_names.json` — the 49 encoded feature names, in model order
- `ref_date.json` — reference date used to convert `previous_recovery_date` to numeric
- `final_metrics.json` — all headline metrics in one file
- `feature_importance.csv` — impurity-based importance (49 encoded features)
- `permutation_importance.csv` — TEST-set permutation importance (PR-AUC based)
- `importance_comparison.csv` — the two importances joined side by side
- `shap_importance.csv` — SHAP global importance (200 test samples)
- `shap_summary_plot.png` — SHAP summary beeswarm plot
- `topk_analysis.csv` — top-K targeting table
- `error_analysis.csv` — per-transaction TP/FP/TN/FN with calibrated probability
