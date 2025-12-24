# C4 Parts Forecasting Project

**Next-Day Parts Prediction for CA 4-Machine Positions**

_Last updated: 2025-12-22_

---

## Project Overview

This project implements a multiclass next-day forecasting system for CA machine positions (QS1..QS4). Each day, for each position, exactly one part outcome (0-9) occurs. The system:

- Predicts tomorrow's required part label using historical data + network aggregate distributions
- Provides Top-K likely parts and Bottom-M exclusion lists
- Outputs calibrated probabilities for decision support

**Key Principle**: Leakage-safe supervised framing - features at day `t` predict target at day `t+1`.

---

## Directory Structure

```
c_4_experiments/
├── _bmad/                      # BMAD Framework (v6.0.0-alpha.19)
│   ├── core/                   # Core module (bmad-master agent, party-mode)
│   ├── bmm/                    # Method Module (full agent suite, workflows)
│   ├── bmb/                    # Builder Module (create agents/workflows)
│   └── _config/                # Configuration files
│
├── c4_bmad_handoff/            # NEW AGENT SOURCE MATERIALS
│   ├── bmad_agent_data_engineer_manufacturing_forecast.md
│   ├── bmad_agent_ml_scientist_parts_forecast.md
│   ├── bmad_agent_leakage_auditor_time_series.md
│   ├── bmad_agent_ops_cost_modeler.md
│   └── bmad_updates_existing_agents_c4.md
│
├── data/
│   └── raw/                    # Source CSV files
│       ├── CA_4_predict_daily_aggregate.csv   (6/9/2008 - 12/17/2025)
│       ├── CA_4_predict_mid_aggregate.csv     (6/9/2008 - 12/17/2025)
│       ├── CA_4_predict_eve_aggregate.csv     (5/19/2008 - 12/17/2025) ⚠️ NEEDS TRIM
│       └── c-4_RESULTS.txt
│
├── docs/                       # Project Documentation
│   ├── prd.md                  # Product Requirements Document
│   └── architecture.md         # System Architecture
│
├── scripts/                    # Pipeline Scripts
│   ├── c4_parts_forecast_topk_pipeline.py    # Main ML pipeline
│   ├── trim_eve_to_2008_06_09.py             # Data alignment utility
│   └── requirements_c4_parts_forecast.txt    # Python dependencies
│
├── artifacts/                  # OUTPUT (created by pipeline) - not yet created
├── HANDOFF_TODO.md             # Full handoff documentation
└── README.md                   # This file
```

---

## Data Overview

### Source Files

| File | Start Date | End Date | Aggregate Row Sum |
|------|------------|----------|-------------------|
| `CA_4_predict_daily_aggregate.csv` | 2008-06-09 | 2025-12-17 | 37 (typical) |
| `CA_4_predict_mid_aggregate.csv` | 2008-06-09 | 2025-12-17 | 16 (typical) |
| `CA_4_predict_eve_aggregate.csv` | **2008-05-19** | 2025-12-17 | 21 (typical) |

### Column Schema

- **Targets**: `CA_QS1`, `CA_QS2`, `CA_QS3`, `CA_QS4` (values 0-9)
- **Aggregates**: `QS{pos}_{label}` counts (e.g., `QS1_0` through `QS1_9` for position 1)
- **Date**: Daily timestamps

---

## FIRST STEPS (Minimal Sequence)

### Step 0: Prerequisites

```bash
# Install Python dependencies
pip install -r scripts/requirements_c4_parts_forecast.txt
```

### Step 1: Mandatory Data Fix - Align Eve CSV

The eve CSV starts at 2008-05-19 while daily/mid start at 2008-06-09. **This must be fixed first.**

```bash
python scripts/trim_eve_to_2008_06_09.py \
  --in_csv data/raw/CA_4_predict_eve_aggregate.csv \
  --out_csv data/raw/CA_4_predict_eve_aggregate.csv
```

### Step 2: Run the Pipeline

```bash
python scripts/c4_parts_forecast_topk_pipeline.py \
  --daily_csv data/raw/CA_4_predict_daily_aggregate.csv \
  --mid_csv data/raw/CA_4_predict_mid_aggregate.csv \
  --eve_csv data/raw/CA_4_predict_eve_aggregate.csv \
  --out_dir artifacts
```

### Step 3: Review Outputs

Pipeline outputs in `artifacts/`:
- `invariant_row_sum_exceptions.csv` - Data quality exceptions
- `metrics_topk_by_position.csv` - Top-K accuracy curves
- `metrics_exclusion_by_position.csv` - Exclusion risk curves
- `next_day_predictions.json` - Model predictions
- `next_day_predictions_aggregate_ranker.json` - Baseline comparison
- `run_summary.json` - Split info and artifact paths

---

## BMAD Workflow Sequence (Full Process)

For systematic development using BMAD agents:

### Phase 1: Data Engineering
**Agent**: Data Engineer (`/bmad:bmm:agents:analyst` + capabilities from `bmad_agent_data_engineer_manufacturing_forecast.md`)

1. Validate dataset contract (schema, dates, ranges)
2. Enforce row-sum invariants (mid=16, eve=21, daily=37)
3. Build leakage-safe features (t -> t+1)
4. Produce walk-forward splits

**Outputs**: `artifacts/data_quality_report.md`, `artifacts/invariant_row_sum_exceptions.csv`

### Phase 2: Leakage Audit
**Agent**: Leakage Auditor (`bmad_agent_leakage_auditor_time_series.md`)

1. Verify supervised framing (features at t predict t+1)
2. Run leak probes (target shuffle, feature shift)
3. Certify pipeline is leakage-free

**Outputs**: `artifacts/leakage_audit_report.md`, `artifacts/leak_probe_results.json`

### Phase 3: ML Modeling
**Agent**: ML Scientist (`bmad_agent_ml_scientist_parts_forecast.md`)

1. Run baselines: majority, persistence, Markov, aggregate-ranker
2. Train pooled model (position as feature)
3. Compare pooled vs per-position models
4. Calibrate probabilities

**Outputs**: `artifacts/metrics_topk_by_position.csv`, `artifacts/metrics_exclusion_by_position.csv`

### Phase 4: Decision Policy
**Agent**: Ops Cost Modeler (`bmad_agent_ops_cost_modeler.md`)

1. Define cost parameters (shipping, downtime, expedite, salvage)
2. Evaluate policies: ship top-1, stage top-K, exclusion bottom-M
3. Recommend optimal K and M

**Outputs**: `artifacts/cost_model.yml`, `artifacts/recommended_policy.md`

### Phase 5: Operationalization
**Agent**: Dev (`/bmad:bmm:agents:dev`)

1. Build CLI: validate, train, evaluate, predict-next-day
2. Define stable daily output contract
3. Document retraining cadence

---

## Creating BMAD Agents

Use BMAD Builder to create the specialized agents:

```
/bmad:bmb:workflows:create-agent
```

Feed each source file from `c4_bmad_handoff/` to create:
1. **Data Engineer** - from `bmad_agent_data_engineer_manufacturing_forecast.md`
2. **ML Scientist** - from `bmad_agent_ml_scientist_parts_forecast.md`
3. **Leakage Auditor** - from `bmad_agent_leakage_auditor_time_series.md`
4. **Ops Cost Modeler** - from `bmad_agent_ops_cost_modeler.md`

Or update existing agents using `bmad_updates_existing_agents_c4.md`.

---

## Daily Operational Outputs

Per position (QS1-QS4), the system provides:
- **Top-1 part** - Most likely part to ship
- **Top-3 parts** - For staging/backup
- **Least-likely 7 parts** - Exclusion list
- **Probabilities** - Full distribution over labels 0-9
- **Confidence signals** - Entropy, top-1 margin

---

## Definition of Done (Phase 1)

- [ ] Data alignment fixed (eve trimmed to 2008-06-09)
- [ ] Pipeline runs end-to-end without errors
- [ ] Baselines + pooled model results saved
- [ ] Top-K curves + exclusion curves produced
- [ ] Policy recommendation delivered (K and M values)

---

## Key Documents

| Document | Location | Purpose |
|----------|----------|---------|
| PRD | `docs/prd.md` | Product requirements |
| Architecture | `docs/architecture.md` | System design |
| Handoff | `HANDOFF_TODO.md` | Full instructions |
| Agent Sources | `c4_bmad_handoff/` | New agent definitions |

---

## Technology Stack

- **Python 3.10+**
- **NumPy 1.26.4**
- **Pandas 2.2.2**
- **scikit-learn 1.5.1**
- **BMAD Framework 6.0.0-alpha.19**
