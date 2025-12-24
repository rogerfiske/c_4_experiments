# Start Here Tomorrow - December 23, 2025

## C4 Parts Forecasting Project - Day 2 Kickoff

---

## Quick Context Recap

**What this project does**: Predicts which of 10 parts (0-9) will be needed for each of 4 positions (QS1-QS4) tomorrow, enabling proactive staging to reduce downtime.

**Key insight discovered**: Parts are **interchangeable** across positions. A part "8" predicted for QS1 can fulfill a QS3 need. Therefore, **pooled coverage** is the correct evaluation metric.

**Current model performance** (100-day holdout):
| Top-K | Pooled Coverage | 100% Days | Parts Shipped |
|-------|-----------------|-----------|---------------|
| Top-5 | 91.2% | 69% | 20/day |
| Top-7 | 97.2% | 90% | 28/day |

**Pipeline status**: Leakage-free (certified), baselines complete, cost model defined.

---

## Phase 1 Complete - What's Done

- [x] Data alignment (eve trimmed to match daily/mid start dates)
- [x] End-to-end ML pipeline with HistGradientBoostingClassifier
- [x] Top-K accuracy curves and exclusion risk analysis
- [x] Leakage audit certified (3 probes passed)
- [x] Cost-aware policy evaluation (K=7 staging recommended)
- [x] Strict holdout test with pooled coverage metrics
- [x] BMAD agents created (4 new, 5 augmented)

---

## Phase 2 TODOs - Model Improvement

### Priority 1: Feature Engineering Experiments

The holdout test revealed which features dominate. Try these improvements:

1. **Expand lag windows**
   ```python
   # Current: lags = [1, 2, 7, 14, 28]
   # Try: lags = [1, 2, 3, 7, 14, 21, 28, 56, 90]
   ```

2. **Add rolling statistics**
   ```python
   # Rolling mean/std over 7, 14, 28 day windows
   for window in [7, 14, 28]:
       df[f'ca_roll_mean_{window}'] = df[ca_col].rolling(window).mean()
       df[f'ca_roll_std_{window}'] = df[ca_col].rolling(window).std()
   ```

3. **Add day-of-week/month features**
   ```python
   df['dow'] = df['date'].dt.dayofweek
   df['month'] = df['date'].dt.month
   df['is_monday'] = (df['dow'] == 0).astype(int)
   ```

4. **Cross-position features**
   ```python
   # Correlation between positions
   # Most common part across all positions yesterday
   ```

5. **Eve data integration**
   - The eve CSV contains per-part counts (e.g., `QS1_0` through `QS1_9`)
   - These could provide stronger signals than just `CA_QS*`

### Priority 2: Model Experiments

1. **Try different models**
   - XGBoost with custom objective
   - LightGBM
   - Neural network (LSTM for sequence)

2. **Calibration**
   - Current probabilities may not be well-calibrated
   - Try Platt scaling or isotonic regression

3. **Per-position vs pooled training**
   - Current: Train one model on all positions
   - Alternative: Train 4 separate models per position

### Priority 3: Evaluation Enhancements

1. **Walk-forward validation**
   - Implement expanding window training
   - Report stability of metrics over time

2. **Confidence thresholding**
   - Only ship if Top-1 confidence > X%
   - Could reduce shipping costs while maintaining coverage

3. **Economic optimization**
   - Directly optimize for expected cost, not accuracy
   - Adjust K dynamically based on prediction confidence

---

## Commands to Resume Work

### 1. Verify environment
```bash
cd C:\Users\Minis\CascadeProjects\c_4_experiments
python --version  # Should be 3.8+
pip list | grep -E "pandas|sklearn|numpy"
```

### 2. Re-run holdout test (baseline)
```bash
python scripts/holdout_test_fast.py
```

### 3. View current results
```bash
cat artifacts/holdout_test_summary.csv
cat artifacts/recommended_policy.md
```

---

## Key Files Reference

| Purpose | File |
|---------|------|
| Main pipeline | `scripts/c4_parts_forecast_topk_pipeline.py` |
| Holdout test | `scripts/holdout_test_fast.py` |
| Leakage audit | `scripts/leakage_audit.py` |
| Policy analysis | `scripts/policy_analysis.py` |
| Daily data | `data/raw/CA_4_predict_daily_aggregate.csv` |
| Holdout results | `artifacts/holdout_test_results.json` |
| Session summary | `docs/Session_Summary_2025-12-22.md` |

---

## Agent Reference

For BMAD workflows, use these specialized agents:

| Agent | Command | Purpose |
|-------|---------|---------|
| Data Engineer | `@c4/data-engineer` | Feature engineering, data validation |
| ML Scientist | `@c4/ml-scientist` | Model experiments, calibration |
| Leakage Auditor | `@c4/leakage-auditor` | QA probes, certification |
| Ops Cost Modeler | `@c4/ops-cost-modeler` | Policy optimization |

---

## Success Metrics for Phase 2

Target improvements over baseline:

| Metric | Current | Target |
|--------|---------|--------|
| Top-5 Coverage | 91.2% | 95%+ |
| Top-5 Perfect Days | 69% | 80%+ |
| Top-3 Coverage | 69.5% | 80%+ |

---

## Data Quality Notes

- **Row-sum invariant exceptions**: 836 rows where part counts don't sum to expected value
  - See `artifacts/invariant_row_sum_exceptions.csv`
  - Consider filtering or investigating these anomalies

- **Date ranges**:
  - Train: 2008-06-09 to 2024-06-19
  - Val: 2024-06-20 to 2024-12-16
  - Test/Holdout: 2024-12-17 to 2025-12-16

---

## Quick Start Tomorrow

1. Read this document
2. Run `python scripts/holdout_test_fast.py` to verify baseline
3. Pick one feature engineering experiment from Priority 1
4. Modify `holdout_test_fast.py` or create new script
5. Compare pooled coverage to baseline
6. If improved, integrate into main pipeline

Good luck!
