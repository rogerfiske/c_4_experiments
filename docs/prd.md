# PRD — C4 Next‑Day Parts Forecasting (CA 4‑Machine Positions)
_Last updated: 2025-12-18 20:05:54_

## 1) Background
We operate 4 identical machine positions at a California (CA) site. Each day, for each position, exactly one “part outcome” occurs:
- 0 = no part required
- 1..9 = part number required

A daily shipment decision is made **once per day** for each machine position.

We also have aggregate counts for the same machine positions across the full network:
- 21 sites total (16 have two shifts, 5 are evening-only)
- We produce **mid**, **eve**, and **daily** aggregate datasets
- CA site runs evening shift only
- Aggregates include CA

Aggregates for day **t** are available when predicting CA for day **t+1**.

## 2) Problem Statement
Predict the required part label (0..9) for each CA machine position for the next day (t+1),
and provide operational recommendations:
- **Top‑K** likely parts (service curves)
- **Least‑likely list** (bottom‑M) to support “exclusion” logic
- Calibrated probabilities for decision support

## 3) Scope
### In scope
- Next-day multiclass forecasting for positions:
  - CA_QS1, CA_QS2, CA_QS3, CA_QS4
- Use historical CA labels + aggregate distributions as features (lagged; leakage-safe)
- Evaluation:
  - Top‑1 accuracy
  - Top‑K accuracy curve (K=1..9)
  - Bottom‑M exclusion risk curve (M=1..9)
  - Log loss (probability calibration)
- Produce a daily recommendation payload:
  - per position: top‑1, top‑3, bottom‑7, prob distribution, confidence metrics

### Not in scope (initial)
- Intraday forecasting (no hour/season effects assumed)
- Multi‑day horizons > 1 day (phase 2)
- Part substitution (exact match required)
- Causal inference (focus is predictive utility)

## 4) Data Sources
Three CSVs:
1. `CA_4_predict_mid_aggregate.csv` (mid shift aggregates)
2. `CA_4_predict_eve_aggregate.csv` (evening shift aggregates)
3. `CA_4_predict_daily_aggregate.csv` (daily = mid + eve)

Key columns:
- `date`
- Targets: `CA_QS1..CA_QS4` in {0..9}
- Aggregates: `QS{pos}_{0..9}` counts per position

Expected aggregate row sums per position (typical):
- mid: 16
- eve: 21
- daily: 37
Exceptions can occur due to holidays/outages; exceptions should be recorded.

### Data alignment requirement
All three files must share the same start date for stable feature engineering.
Target start date: **2008‑06‑09** (trim eve to match).

## 5) Users & Stakeholders
- Operations planner (daily part shipment decision)
- Maintenance leadership (service level / downtime risk)
- Data science / ML engineering team (model lifecycle)

## 6) Operational Constraints
- One shipment decision per position per day
- Cost of wrong part includes:
  - wasted shipping
  - downtime
  - expedited replacement
- Wrong part remains usable later (salvage value)

## 7) Success Metrics
Primary:
- Improve decision quality with measurable curves:
  - Top‑1 accuracy (ship exactly one)
  - Top‑K coverage curve (K=1..9)
  - Bottom‑M exclusion risk curve (M=1..9)
  - Log loss (probability calibration)
Secondary:
- Stability across time splits (no overfit spikes)
- Interpretability of drivers (feature importance / error analysis)

## 8) Acceptance Criteria (Phase 1)
1. Reproducible pipeline with leakage-safe evaluation
2. Baselines implemented and reported per position:
   - majority, persistence, Markov, aggregate-ranker
3. Pooled model implemented (Option 2) and compared to per-position models (Option 1)
4. Output artifacts:
   - `metrics_topk_by_position.csv`
   - `metrics_exclusion_by_position.csv`
   - `next_day_predictions.json`
5. Documented decision policy recommendation:
   - recommended operational K and M (based on curves and cost model)

## 9) Risks / Assumptions
- Assumption: aggregates for day t are available prior to shipping decision for day t+1
- Aggregates include CA; only safe if used as features for next day (t -> t+1)
- Holiday/outage exceptions may degrade model; must be surfaced, not hidden
- Potential concept drift during Covid and other disruptions

## 10) Deliverables
- PRD (this doc)
- Architecture doc (`architecture.md`)
- New BMAD agents (or capability updates) + handoff instructions
- Baseline + pooled model pipeline scripts and artifacts
