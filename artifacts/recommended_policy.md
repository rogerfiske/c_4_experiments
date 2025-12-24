# C4 Parts Forecast - Recommended Policy

**Analysis Date:** 2025-12-22T11:49:26.695559

---

## Cost Parameters

| Parameter | Value |
|-----------|-------|
| Ship Cost | $50.00 |
| Downtime Cost | $500.00/day |
| Expedite Multiplier | 3.0x |
| Salvage Rate | 80% |

## Policy Comparison

| Policy | Coverage | Expected Cost | Downtime Risk |
|--------|----------|---------------|---------------|
| Top-1 Ship | 9.5% | $601.92 | 90.5% |
| Top-3 Staging | 29.0% | $588.50 | 71.0% |
| Top-5 Staging | 48.8% | $504.99 | 51.2% |
| Top-7 Staging | 69.1% | $384.92 | 30.9% |

## Exclusion Safety

| Exclude | Risk | Status |
|---------|------|--------|
| Bottom-1 | 10.3% | RISKY |
| Bottom-2 | 21.6% | RISKY |
| Bottom-3 | 30.9% | RISKY |
| Bottom-4 | 40.8% | RISKY |
| Bottom-5 | 51.2% | RISKY |
| Bottom-6 | 61.0% | RISKY |
| Bottom-7 | 71.0% | RISKY |

---

## Final Recommendations

### Staging: Use Top-9

Stage top-9 parts to achieve target coverage.

### Exclusion: Safely Exclude Bottom-0

No parts can be safely excluded with <5% risk.

### Operational Implementation

Daily output for operations team:

| Position | Top-1 (Ship) | Top-3 (Stage) | Exclude |
|----------|--------------|---------------|----------|
| QS1 | Part 8 | Parts 8, 4, 0 | Parts  |
| QS2 | Part 5 | Parts 5, 8, 9 | Parts  |
| QS3 | Part 6 | Parts 6, 9, 1 | Parts  |
| QS4 | Part 7 | Parts 7, 4, 2 | Parts  |

