# BMAD Team Handoff — C4 Experiments (Parts Forecasting)
_Last updated: 2025-12-18 20:05:54_

## 0) What you received in this bundle
Files included:
- New agent persona sources:
  - `bmad_agent_data_engineer_manufacturing_forecast.md`
  - `bmad_agent_ml_scientist_parts_forecast.md`
  - `bmad_agent_leakage_auditor_time_series.md`
  - `bmad_agent_ops_cost_modeler.md`
  - `bmad_updates_existing_agents_c4.md`
- Working pipeline:
  - `c4_parts_forecast_topk_pipeline.py`
  - `requirements_c4_parts_forecast.txt`
- Core docs:
  - `prd.md`
  - `architecture.md`
  - This handoff doc

## 1) Mandatory data fix
**Action:** Trim `CA_4_predict_eve_aggregate.csv` to start at **2008‑06‑09**.

Why:
- daily and mid currently start at 2008‑06‑09; eve starts earlier
- alignment simplifies feature creation and prevents inconsistent early-history lags

Deliverable:
- overwrite `CA_4_predict_eve_aggregate.csv` in the repo (or create a new file and update pipeline paths)

Suggested method:
- Use `scripts/trim_eve_to_2008_06_09.py` provided in this bundle.

## 2) Recommended BMAD sequence (high signal order)
### Step 1 — Data Engineer: dataset contract + invariants
- Validate dates are continuous and aligned
- Validate target ranges 0..9
- Validate row-sum invariants per file type:
  - mid=16, eve=21, daily=37 (allow exceptions; export exceptions)

Outputs:
- `artifacts/invariant_row_sum_exceptions.csv`
- `artifacts/data_quality_report.md` (short)

### Step 2 — Leakage Auditor: certify supervised framing
- Confirm modeling uses features at day t to predict target at day t+1
- Run leak probes:
  - target shuffle -> chance performance
  - feature shift +1 -> detect future leakage

Outputs:
- `artifacts/leakage_audit_report.md`
- `artifacts/leak_probe_results.json`

### Step 3 — ML Scientist: baselines first
Run baselines per position:
- majority
- persistence
- Markov transition P(next|today)
- aggregate-ranker (use day t aggregates to score t+1)

Output:
- baseline metrics per position
- establish a minimum bar for any model

### Step 4 — ML Scientist: pooled model (Option 2)
Train pooled model on stacked positions with `position` feature.
Compare:
- pooled vs baselines
- pooled vs per-position models (Option 1 branch)

Outputs:
- `metrics_topk_by_position.csv`
- `metrics_exclusion_by_position.csv`

### Step 5 — Ops Cost Modeler: decision policy
Using probability outputs:
- evaluate:
  - ship top‑1
  - stage top‑K (if allowed)
  - exclusion bottom‑M risk (safety)

Deliver:
- recommended K and M based on service level and cost

### Step 6 — PM/Architect/Dev: package for operations
- define a stable daily output contract
- implement `predict_next_day.py` and/or scheduled job
- document retraining cadence

## 3) How to use bmad-builder with these agent sources
- Feed each `bmad_agent_*.md` file as source material to `bmad-builder`
- Ask the builder to:
  - create new agents where missing (Data Engineer, ML Scientist, Leakage Auditor, Ops Cost Modeler)
  - or incorporate additions into existing agents using `bmad_updates_existing_agents_c4.md`

## 4) Operational outputs required
Daily planner needs, per position:
- top‑1 part
- top‑3 parts
- least‑likely 7 parts
- probabilities for labels 0..9
- confidence signals (entropy, top‑1 margin)

## 5) Definition of done (Phase 1)
- Data alignment fixed (eve trimmed)
- Pipeline runs end-to-end
- Baselines + pooled model results saved
- Top‑K curves + exclusion curves produced
- Policy recommendation delivered (K and M)
