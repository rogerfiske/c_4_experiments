# Architecture — C4 Next‑Day Parts Forecasting System
_Last updated: 2025-12-18 20:05:54_

## 1) Overview
System produces next-day forecasts for CA machine positions (QS1..QS4) as multiclass probabilities over labels 0..9,
then converts those probabilities into operational recommendations (top‑K and bottom‑M).

Key principles:
- Leakage-safe supervised framing: features at day t predict target at day t+1
- Blocked / walk-forward evaluation only (no shuffled splits)
- Predictor and decision policy are separate modules

## 2) Data Flow
### Inputs
- `CA_4_predict_mid_aggregate.csv`
- `CA_4_predict_eve_aggregate.csv` (must be trimmed to 2008‑06‑09)
- `CA_4_predict_daily_aggregate.csv`

### Preprocessing / Validation
- Parse `date`, sort, enforce daily continuity
- Validate:
  - CA_QS* in 0..9
  - QS{pos}_{0..9} non-negative integers
  - Row-sum invariants per file type (mid=16, eve=21, daily=37) with exception logging

### Feature Engineering (for each position)
Supervised dataset rows represent (date=t, position=p):
- Target: y = CA_QS{p}(t+1)
- Features from day t and earlier:
  - CA lag features: CA_QS{p}(t-1), (t-2), (t-7), ...
  - Aggregate count features at day t (network demand distribution)
  - Aggregate lag features
  - Aggregate proportion features
  - Rolling stats (computed from <= t)

### Modeling
Phase 1:
- Baselines: majority, persistence, Markov, aggregate-ranker
- Pooled model: one classifier trained on stacked positions with feature `position ∈ {1..4}`
- Compare with per-position models (4 separate) as an experiment branch

Calibration (Phase 2 or late Phase 1):
- Temperature scaling or isotonic regression on validation folds

## 3) Evaluation
### Splits
- Blocked end-of-series split: train / validation / test by date
- Optional walk-forward folds for stability estimates

### Metrics (per position)
- Top‑1 accuracy
- Top‑K accuracy curve (K=1..9)
- Bottom‑M exclusion risk curve (M=1..9)
- Log loss (calibration)

## 4) Decision Policy Layer
Inputs:
- Calibrated probability distribution over {0..9} per position

Outputs:
- Top‑1 recommended part to ship
- Optional top‑K list (for staging/backup)
- Bottom‑M least likely list (exclusion)
- Confidence scores (entropy, margin between top‑1 and top‑2)

Policy selection:
- Start with “ship top‑1”
- Extend to cost-aware policy using Ops cost model:
  - minimize expected cost (shipping + downtime + expedite − salvage)

## 5) Artifacts & Contracts
All artifacts written to `artifacts/`:
- `invariant_row_sum_exceptions.csv`
- `metrics_topk_by_position.csv`
- `metrics_exclusion_by_position.csv`
- `next_day_predictions.json`
- `next_day_predictions_aggregate_ranker.json`
- `run_summary.json`

## 6) Monitoring (future)
- Daily drift monitor on aggregate distributions
- Alert on:
  - unusual row-sum exceptions
  - confidence collapse (high entropy)
  - sustained accuracy degradation
