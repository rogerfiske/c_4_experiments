# C4 Parts Forecast - Metrics Collection Framework

**Version**: 1.0
**Created**: 2025-12-24
**Purpose**: Systematic tracking of model performance, stability, and feature health to enable data-driven optimization decisions.

---

## 1. Framework Overview

### 1.1 Design Principles

1. **Observe before optimizing** — Don't chase improvements without understanding stability
2. **Track both absolute and relative metrics** — Accuracy levels AND rankings
3. **Detect drift early** — Alert when patterns shift before they impact operations
4. **Enable root cause analysis** — Trace performance changes to specific features/parts/positions

### 1.2 Collection Cadence

| Metric Category | Collection Frequency | Retention |
|-----------------|---------------------|-----------|
| Daily predictions | Daily | 2 years |
| Accuracy metrics | Daily (rolling windows) | 2 years |
| Stability metrics | Weekly | 1 year |
| Feature health | Weekly | 1 year |
| Model retraining | Monthly | Indefinite |

---

## 2. Metric Definitions

### 2.1 Accuracy Metrics (Layer 1 - Core)

#### 2.1.1 Overall Performance

| Metric | Definition | Target | Alert Threshold |
|--------|------------|--------|-----------------|
| `pooled_coverage_top5` | % of actual parts covered by Top-5 across all positions | >91% | <88% |
| `pooled_coverage_top7` | % of actual parts covered by Top-7 | >97% | <94% |
| `perfect_day_rate_top5` | % of days with 100% Top-5 coverage | >70% | <60% |
| `perfect_day_rate_top7` | % of days with 100% Top-7 coverage | >90% | <85% |

#### 2.1.2 Per-Part Accuracy (Parts 0-9)

| Metric | Definition | Alert Threshold |
|--------|------------|-----------------|
| `part_{X}_top5_accuracy` | Top-5 accuracy when actual part is X | <35% (hard parts) |
| `part_{X}_avg_rank` | Average rank of actual part X in predictions | >6.5 |
| `part_{X}_sample_count` | Number of occurrences in window | <10 (insufficient data) |

**Special tracking for hard parts (2, 7):**
- Lower alert thresholds (these are expected to be harder)
- Track improvement over baseline

#### 2.1.3 Per-Position Accuracy (QS1-QS4)

| Metric | Definition | Alert Threshold |
|--------|------------|-----------------|
| `pos_{X}_top5_accuracy` | Top-5 accuracy for position X | <45% |
| `pos_{X}_top1_accuracy` | Top-1 accuracy for position X | <8% |
| `pos_{X}_avg_confidence` | Average Top-1 probability for position X | Drift >5% from baseline |

---

### 2.2 Stability Metrics (Layer 2 - Trend Analysis)

#### 2.2.1 Accuracy Stability

| Metric | Definition | Healthy Range | Alert |
|--------|------------|---------------|-------|
| `part_accuracy_cv` | Coefficient of variation for part accuracy | <25% | >30% |
| `pos_accuracy_cv` | Coefficient of variation for position accuracy | <15% | >20% |
| `overall_accuracy_cv` | CV for overall pooled coverage | <10% | >15% |

#### 2.2.2 Ranking Stability

| Metric | Definition | Healthy Range | Alert |
|--------|------------|---------------|-------|
| `part_rank_correlation` | Spearman rho between current and previous window | >0.5 | <0.3 |
| `pos_rank_correlation` | Spearman rho for position rankings | >0.7 | <0.5 |
| `ranking_inversion_count` | # of rank changes >3 positions | <2 | >4 |

#### 2.2.3 Drift Detection

| Metric | Definition | Alert Threshold |
|--------|------------|-----------------|
| `accuracy_drift_7d` | Change in Top-5 accuracy vs 7 days ago | >5% drop |
| `accuracy_drift_30d` | Change in Top-5 accuracy vs 30 days ago | >8% drop |
| `part_drift_max` | Max accuracy change for any single part | >15% |

---

### 2.3 Feature Health Metrics (Layer 3 - Diagnostics)

#### 2.3.1 Feature Importance

| Metric | Definition | Purpose |
|--------|------------|---------|
| `feature_{X}_importance` | Model importance score for feature X | Track contribution |
| `feature_{X}_importance_rank` | Rank of feature X by importance | Detect rank changes |
| `new_features_total_importance` | Sum of importance for new features (lags 3,21,56 + rolling) | Validate additions |

#### 2.3.2 Feature Drift

| Metric | Definition | Alert Threshold |
|--------|------------|-----------------|
| `feature_importance_correlation` | Spearman rho between current and baseline importance | <0.7 |
| `top10_feature_stability` | % of top-10 features that remain in top-10 | <70% |
| `feature_dropout_count` | # of features with importance drop >50% | >3 |

#### 2.3.3 Feature Category Performance

| Category | Features | Track |
|----------|----------|-------|
| `lag_features` | ca_lag1, ca_lag2, ..., ca_lag56 | Total importance, top contributor |
| `rolling_features` | ca_roll_mean_7/14/28 | Total importance |
| `aggregate_features` | agg_count_*, agg_prop_* | Total importance |
| `temporal_features` | dow, is_monday (if used) | Individual importance |

---

### 2.4 Model Health Metrics (Layer 4 - Calibration)

#### 2.4.1 Probability Calibration

| Metric | Definition | Healthy Range |
|--------|------------|---------------|
| `calibration_error` | Mean absolute difference between predicted prob and actual freq | <0.05 |
| `brier_score` | Brier score for probability predictions | <0.15 |
| `reliability_slope` | Slope of reliability diagram (predicted vs actual) | 0.9-1.1 |

#### 2.4.2 Confidence Distribution

| Metric | Definition | Purpose |
|--------|------------|---------|
| `avg_top1_confidence` | Mean probability assigned to top-1 prediction | Track overconfidence |
| `confidence_entropy` | Entropy of probability distribution | Low = overconfident |
| `confidence_spread` | Std of top-1 confidence across predictions | Track consistency |

#### 2.4.3 Prediction Diversity

| Metric | Definition | Alert |
|--------|------------|-------|
| `prediction_entropy` | Entropy of predicted part distribution | Low = bias |
| `prediction_bias_max` | Max over/under-prediction for any part | >5% |
| `unique_top1_count` | # of unique parts predicted as top-1 in window | <7 (should be ~10) |

---

### 2.5 Data Quality Metrics (Layer 5 - Input Health)

| Metric | Definition | Alert Threshold |
|--------|------------|-----------------|
| `row_sum_exceptions` | Count of rows violating sum invariant | >1% of data |
| `missing_dates` | Count of missing dates in period | >0 |
| `class_imbalance_ratio` | Max/min class frequency ratio | >2.0 |
| `data_freshness` | Days since last data update | >1 day |

---

## 3. Collection Architecture

### 3.1 Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw CSV Data   │────▶│  Feature Engine  │────▶│  Model Predict  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Metrics Store  │◀────│  Metrics Calc    │◀────│  Predictions    │
│  (JSON/CSV)     │     │  Engine          │     │  + Actuals      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│  Alert Engine   │────▶ Notifications
│  + Dashboard    │
└─────────────────┘
```

### 3.2 Storage Schema

#### 3.2.1 Daily Predictions Log (`artifacts/predictions_log.jsonl`)

```json
{
  "date": "2025-12-24",
  "model_version": "v1.2",
  "predictions": [
    {
      "position": 1,
      "actual": 4,
      "top5": [8, 4, 1, 6, 3],
      "probabilities": [0.08, 0.12, 0.05, ...],
      "top1_confidence": 0.18,
      "actual_rank": 2
    },
    ...
  ],
  "pooled_coverage": {"top1": 0.25, "top3": 0.75, "top5": 1.0, "top7": 1.0}
}
```

#### 3.2.2 Daily Metrics (`artifacts/metrics_daily.csv`)

| Column | Type | Description |
|--------|------|-------------|
| date | date | Prediction date |
| pooled_top5 | float | Pooled Top-5 coverage |
| pooled_top7 | float | Pooled Top-7 coverage |
| part_0_accuracy ... part_9_accuracy | float | Per-part Top-5 |
| pos_1_accuracy ... pos_4_accuracy | float | Per-position Top-5 |
| avg_confidence | float | Mean Top-1 confidence |
| model_version | string | Model identifier |

#### 3.2.3 Weekly Stability Report (`artifacts/stability_weekly.json`)

```json
{
  "week_ending": "2025-12-24",
  "window_days": 30,
  "accuracy": {
    "pooled_top5_mean": 0.918,
    "pooled_top5_std": 0.05,
    "pooled_top5_cv": 0.054
  },
  "part_stability": {
    "rank_correlation_vs_prev": 0.45,
    "hard_parts": [2, 7],
    "easy_parts": [4, 5, 8]
  },
  "feature_health": {
    "importance_correlation": 0.85,
    "top10_stability": 0.9,
    "new_features_contribution": 0.22
  },
  "alerts": []
}
```

---

## 4. Alert Definitions

### 4.1 Critical Alerts (Immediate Action)

| Alert | Condition | Action |
|-------|-----------|--------|
| `ACCURACY_CRASH` | Top-5 < 85% for 3 consecutive days | Investigate model, check data |
| `DATA_MISSING` | No predictions for >1 day | Check pipeline |
| `MODEL_FAILURE` | Prediction errors/exceptions | Check model integrity |

### 4.2 Warning Alerts (Investigate)

| Alert | Condition | Action |
|-------|-----------|--------|
| `ACCURACY_DRIFT` | Top-5 dropped >5% from 30-day baseline | Review recent changes |
| `PART_DEGRADATION` | Any part accuracy dropped >15% | Check part-specific patterns |
| `FEATURE_DRIFT` | Feature importance correlation <0.7 | Review feature engineering |
| `CALIBRATION_DRIFT` | Calibration error >0.08 | Consider recalibration |

### 4.3 Informational Alerts (Track)

| Alert | Condition | Action |
|-------|-----------|--------|
| `RANKING_SHIFT` | Part rankings changed significantly | Note in log, no action |
| `CONFIDENCE_CHANGE` | Average confidence shifted >3% | Monitor trend |
| `NEW_PATTERN` | Previously easy part becoming hard | Flag for review |

---

## 5. Dashboard Views

### 5.1 Executive Summary (Daily)

```
C4 FORECAST PERFORMANCE - 2025-12-24
=====================================
Overall Status: ✓ HEALTHY

Today's Coverage:
  Top-5: 100% (4/4 parts)  [Target: >91%]
  Top-7: 100% (4/4 parts)  [Target: >97%]

Rolling 30-Day:
  Top-5 Avg: 92.1%  (▲ +0.9% vs baseline)
  Top-7 Avg: 98.3%  (▲ +1.1% vs baseline)
  Perfect Days: 73%

Alerts: None
```

### 5.2 Part Performance View (Weekly)

```
PART DIFFICULTY MATRIX - Week of 2025-12-24
===========================================
                    Top-5 Accuracy
Part    This Week   Prev Week   Trend   Status
----    ---------   ---------   -----   ------
4       64%         61%         ▲       Easy
5       58%         55%         ▲       Easy
8       55%         57%         ▼       Average
9       54%         52%         ▲       Average
0       51%         50%         ─       Average
1       49%         51%         ▼       Average
3       47%         44%         ▲       Average
6       44%         46%         ▼       Hard
2       42%         40%         ▲       Hard
7       39%         41%         ▼       Hard ⚠

Hard Parts (2, 7): Stable, no action needed
```

### 5.3 Stability Dashboard (Weekly)

```
STABILITY REPORT - Week of 2025-12-24
=====================================

Ranking Stability:
  Part rank correlation:     0.67 (Moderate)
  Position rank correlation: 0.89 (Stable)

Accuracy Stability:
  Overall CV:    8.2%  ✓ Healthy (<10%)
  Part CV range: 12-22% ✓ Healthy (<25%)

Feature Health:
  Importance correlation: 0.91 ✓ Stable
  Top-10 features stable: 9/10 ✓
  New features contribution: 21%
```

---

## 6. Implementation Checklist

### Phase 1: Core Collection (Week 1)
- [ ] Daily predictions logging
- [ ] Daily accuracy metrics
- [ ] Basic alert system

### Phase 2: Stability Tracking (Week 2)
- [ ] Weekly stability calculations
- [ ] Rank correlation tracking
- [ ] Drift detection

### Phase 3: Feature Health (Week 3)
- [ ] Feature importance logging
- [ ] Feature drift detection
- [ ] Category performance tracking

### Phase 4: Dashboard (Week 4)
- [ ] Executive summary view
- [ ] Part performance matrix
- [ ] Alert notifications

---

## 7. Usage Examples

### 7.1 Daily Collection Command

```bash
python scripts/collect_metrics.py \
  --date 2025-12-24 \
  --predictions artifacts/predictions_log.jsonl \
  --metrics artifacts/metrics_daily.csv
```

### 7.2 Weekly Stability Report

```bash
python scripts/stability_report.py \
  --end_date 2025-12-24 \
  --window_days 30 \
  --output artifacts/stability_weekly.json
```

### 7.3 Check for Alerts

```bash
python scripts/check_alerts.py \
  --metrics artifacts/metrics_daily.csv \
  --thresholds config/alert_thresholds.yaml
```

---

## 8. Appendix: Baseline Values

Established from 365-day evaluation (2024-12-17 to 2025-12-16):

| Metric | Baseline Value | Source |
|--------|---------------|--------|
| Top-5 Pooled Coverage | 91.2% | Original model |
| Top-7 Pooled Coverage | 97.2% | Original model |
| Part 2 Top-5 Accuracy | 40% | Temporal analysis |
| Part 7 Top-5 Accuracy | 40% | Temporal analysis |
| Part 4 Top-5 Accuracy | 62% | Temporal analysis |
| Rank Correlation (parts) | 0.07 | Temporal analysis |
| Feature Importance Top-1 | ca_lag56 | Model tuning |
