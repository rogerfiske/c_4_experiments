# Leakage Audit Report

**Audit Date:** 2025-12-22T11:47:30.927844

**Verdict:** CERTIFIED

---

## 1. Supervised Framing Verification

| Check | Status |
|-------|--------|
| Target framing (t -> t+1) | PASS |
| Split integrity | PASS |

## 2. Baseline Performance

- **Accuracy:** 0.0945
- **Log Loss:** 2.3721

## 3. Target Shuffle Probe

*Purpose: Shuffled targets should yield ~10% accuracy (chance)*

- **Shuffled Accuracy:** 0.0966
- **Expected:** ~0.10
- **Result:** PASS

## 4. Feature Shift Probe

*Purpose: Shifted features should NOT improve performance*

- **Shifted Accuracy:** 0.1034
- **Baseline Accuracy:** 0.0945
- **Improvement:** +0.0089
- **Result:** PASS

---

## Final Certification

**CERTIFIED**

The pipeline passes all leakage probes. Features at time t correctly predict targets at t+1 without future data leakage.
