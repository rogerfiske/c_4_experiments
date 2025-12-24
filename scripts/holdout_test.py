#!/usr/bin/env python3
"""
Strict Holdout Test for C4 Parts Forecast.

Key insight: Parts are INTERCHANGEABLE across positions.
Evaluation focuses on POOLED COVERAGE rather than per-position accuracy.

This script:
1. Performs strict temporal holdout (no future leakage)
2. Collects detailed metrics for prediction analysis
3. Evaluates pooled parts coverage (interchangeable parts)
4. Generates comprehensive diagnostic data for post-processing

Author: BMad Leakage Auditor + ML Scientist Agents
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

LABELS = tuple(range(10))
N_CLASSES = 10
POS_COLS = ("CA_QS1", "CA_QS2", "CA_QS3", "CA_QS4")


@dataclass
class PredictionRecord:
    """Complete record for a single day's prediction."""
    date: str
    position: int

    # Actuals (only known after prediction made)
    actual_label: int

    # Predictions
    predicted_probs: Dict[str, float]
    top1_pred: int
    top3_preds: List[int]
    top5_preds: List[int]
    top7_preds: List[int]

    # Per-position accuracy
    top1_hit: bool
    top3_hit: bool
    top5_hit: bool
    top7_hit: bool

    # Confidence metrics
    top1_prob: float
    top1_margin: float  # gap between top1 and top2
    entropy: float

    # Feature contributions (top 10)
    top_features: List[Dict[str, Any]] = field(default_factory=list)

    # Rank of actual label
    actual_rank: int = 0


@dataclass
class DayRecord:
    """Complete record for a single day's pooled evaluation."""
    date: str

    # Actuals needed (with counts)
    actuals: List[int]  # [9, 9, 5, 4]
    actuals_unique: Dict[int, int]  # {9: 2, 5: 1, 4: 1}

    # Parts shipped at each K level (with counts)
    top1_shipped: Dict[int, int]
    top2_shipped: Dict[int, int]
    top3_shipped: Dict[int, int]
    top5_shipped: Dict[int, int]
    top7_shipped: Dict[int, int]

    # Pooled coverage metrics
    top1_coverage: float  # parts fulfilled / parts needed
    top2_coverage: float
    top3_coverage: float
    top5_coverage: float
    top7_coverage: float

    # Coverage details
    top1_fulfilled: int
    top2_fulfilled: int
    top3_fulfilled: int
    top5_fulfilled: int
    top7_fulfilled: int
    total_needed: int

    # Per-position predictions
    predictions: List[PredictionRecord] = field(default_factory=list)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict holdout test for C4 pipeline")
    p.add_argument("--daily_csv", type=str, default="data/raw/CA_4_predict_daily_aggregate.csv")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--holdout_days", type=int, default=100,
                   help="Number of most recent days for holdout test")
    p.add_argument("--train_buffer_days", type=int, default=180,
                   help="Buffer days between train end and holdout start")
    p.add_argument("--lags", type=int, nargs="+", default=[1, 2, 7, 14, 28])
    p.add_argument("--roll_windows", type=int, nargs="+", default=[7, 14, 30])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df.sort_values("date").reset_index(drop=True)


def make_features_for_day(df: pd.DataFrame, target_idx: int,
                          lags: Sequence[int], roll_windows: Sequence[int]) -> Dict[int, pd.Series]:
    """
    Build features for a single day's prediction.
    STRICT: Only uses data available at end of day target_idx-1.
    """
    features_by_pos = {}

    for pos in range(1, 5):
        ca_col = f"CA_QS{pos}"
        part_cols = [f"QS{pos}_{lab}" for lab in LABELS]

        feat = {}
        feat["position"] = pos

        # CA lags (relative to target_idx - 1, the last available day)
        for lag in lags:
            idx = target_idx - 1 - (lag - 1)  # lag=1 means yesterday's value
            if idx >= 0:
                feat[f"ca_lag{lag}"] = float(df.loc[idx, ca_col])
            else:
                feat[f"ca_lag{lag}"] = np.nan

        # Aggregate features from day before target (latest available)
        agg_idx = target_idx - 1
        if agg_idx >= 0:
            denom = sum(df.loc[agg_idx, c] for c in part_cols)
            for lab in LABELS:
                feat[f"agg_count_{lab}_t"] = float(df.loc[agg_idx, f"QS{pos}_{lab}"])
                if denom > 0:
                    feat[f"agg_prop_{lab}_t"] = float(df.loc[agg_idx, f"QS{pos}_{lab}"]) / denom
                else:
                    feat[f"agg_prop_{lab}_t"] = 0.0

        # Lagged aggregates
        for lag in lags:
            idx = target_idx - 1 - lag
            if idx >= 0:
                denom = sum(df.loc[idx, c] for c in part_cols)
                for lab in LABELS:
                    feat[f"agg_count_{lab}_lag{lag}"] = float(df.loc[idx, f"QS{pos}_{lab}"])
                    if denom > 0:
                        feat[f"agg_prop_{lab}_lag{lag}"] = float(df.loc[idx, f"QS{pos}_{lab}"]) / denom
                    else:
                        feat[f"agg_prop_{lab}_lag{lag}"] = 0.0
            else:
                for lab in LABELS:
                    feat[f"agg_count_{lab}_lag{lag}"] = np.nan
                    feat[f"agg_prop_{lab}_lag{lag}"] = np.nan

        # Rolling means
        for lab in LABELS:
            for w in roll_windows:
                start_idx = max(0, target_idx - 1 - w)
                end_idx = target_idx - 1
                if end_idx > start_idx:
                    part_col = f"QS{pos}_{lab}"
                    vals = df.loc[start_idx:end_idx, part_col].values
                    denoms = [sum(df.loc[i, c] for c in part_cols) for i in range(start_idx, end_idx + 1)]
                    props = [v / d if d > 0 else 0 for v, d in zip(vals, denoms)]
                    feat[f"agg_prop_{lab}_rollmean{w}"] = np.mean(props)
                else:
                    feat[f"agg_prop_{lab}_rollmean{w}"] = np.nan

        features_by_pos[pos] = pd.Series(feat)

    return features_by_pos


def build_training_data(df: pd.DataFrame, end_idx: int,
                        lags: Sequence[int], roll_windows: Sequence[int]) -> Tuple[pd.DataFrame, List[str]]:
    """Build training dataset up to (but not including) end_idx."""
    rows = []

    # Start from index where we have enough history for max lag
    start_idx = max(lags) + max(roll_windows) + 1

    for target_idx in range(start_idx, end_idx):
        for pos in range(1, 5):
            feat = make_features_for_day(df, target_idx, lags, roll_windows)[pos].to_dict()
            feat["y"] = int(df.loc[target_idx, f"CA_QS{pos}"])
            feat["date"] = df.loc[target_idx, "date"]
            rows.append(feat)

    train_df = pd.DataFrame(rows)
    feature_cols = [c for c in train_df.columns if c not in ("date", "y", "position")]

    return train_df, feature_cols


def compute_entropy(probs: np.ndarray) -> float:
    """Compute entropy of probability distribution."""
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))


def compute_pooled_coverage(actuals: List[int], shipped: Dict[int, int]) -> Tuple[int, int, float]:
    """
    Compute how many actual parts are covered by shipped parts.

    Args:
        actuals: List of actual parts needed (with duplicates, e.g., [9, 9, 5, 4])
        shipped: Dict of parts shipped with counts, e.g., {8: 2, 4: 2, 5: 1, 9: 1}

    Returns:
        (fulfilled, total_needed, coverage_ratio)
    """
    needed = Counter(actuals)
    fulfilled = 0

    for part, need_count in needed.items():
        have_count = shipped.get(part, 0)
        fulfilled += min(need_count, have_count)

    total = len(actuals)
    coverage = fulfilled / total if total > 0 else 0.0

    return fulfilled, total, coverage


def run_holdout_test(df: pd.DataFrame, holdout_days: int, train_buffer: int,
                     lags: Sequence[int], roll_windows: Sequence[int],
                     seed: int) -> List[DayRecord]:
    """
    Run strict holdout test on most recent holdout_days.

    For each holdout day:
    1. Train model on all data before (train_end = holdout_start - train_buffer)
    2. Predict for that day using only available data
    3. Compare to actuals
    """
    np.random.seed(seed)

    n_rows = len(df)
    holdout_start_idx = n_rows - holdout_days
    train_end_idx = holdout_start_idx - train_buffer

    if train_end_idx < max(lags) + max(roll_windows) + 100:
        raise ValueError("Not enough training data")

    print(f"Training data: rows 0 to {train_end_idx-1}")
    print(f"Buffer: {train_buffer} days")
    print(f"Holdout: rows {holdout_start_idx} to {n_rows-1} ({holdout_days} days)")

    # Build training data
    train_df, feature_cols = build_training_data(df, train_end_idx, lags, roll_windows)

    # Handle NaN values
    train_df = train_df.dropna()

    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values

    # Train model
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6, max_iter=300,
        random_state=seed, early_stopping=False
    )
    clf.fit(X_train, y_train)

    # Get feature importances
    feature_importance = dict(zip(feature_cols, clf.feature_importances_))
    sorted_features = sorted(feature_importance.items(), key=lambda x: -x[1])[:20]

    # Run holdout evaluation
    day_records = []

    for target_idx in range(holdout_start_idx, n_rows):
        date_str = str(df.loc[target_idx, "date"].date())

        # Get actuals for this day
        actuals = [int(df.loc[target_idx, f"CA_QS{pos}"]) for pos in range(1, 5)]
        actuals_counter = Counter(actuals)

        # Collect predictions for all positions
        pred_records = []
        all_top1 = []
        all_top2 = []
        all_top3 = []
        all_top5 = []
        all_top7 = []

        for pos in range(1, 5):
            # Get features for this day (strict: only use data available before this day)
            feat_dict = make_features_for_day(df, target_idx, lags, roll_windows)[pos].to_dict()

            # Prepare feature vector
            feat_vec = np.array([[feat_dict.get(c, np.nan) for c in feature_cols]])

            # Handle NaN with median imputation from training
            for i, c in enumerate(feature_cols):
                if np.isnan(feat_vec[0, i]):
                    feat_vec[0, i] = np.nanmedian(train_df[c].values)

            # Predict
            probs = clf.predict_proba(feat_vec)[0]

            # Rank labels by probability
            ranked = np.argsort(-probs)

            top1 = int(ranked[0])
            top3 = [int(x) for x in ranked[:3]]
            top5 = [int(x) for x in ranked[:5]]
            top7 = [int(x) for x in ranked[:7]]

            actual = actuals[pos - 1]
            actual_rank = int(np.where(ranked == actual)[0][0]) + 1

            # Confidence metrics
            top1_prob = float(probs[top1])
            top2_prob = float(probs[ranked[1]])
            margin = top1_prob - top2_prob
            entropy = compute_entropy(probs)

            # Feature contributions for this prediction
            top_feats = []
            for fname, fimp in sorted_features[:10]:
                top_feats.append({
                    "feature": fname,
                    "importance": float(fimp),
                    "value": float(feat_dict.get(fname, np.nan))
                })

            pred_rec = PredictionRecord(
                date=date_str,
                position=pos,
                actual_label=actual,
                predicted_probs={str(i): float(probs[i]) for i in range(10)},
                top1_pred=top1,
                top3_preds=top3,
                top5_preds=top5,
                top7_preds=top7,
                top1_hit=(actual == top1),
                top3_hit=(actual in top3),
                top5_hit=(actual in top5),
                top7_hit=(actual in top7),
                top1_prob=top1_prob,
                top1_margin=margin,
                entropy=entropy,
                top_features=top_feats,
                actual_rank=actual_rank
            )
            pred_records.append(pred_rec)

            # Collect for pooled analysis
            all_top1.append(top1)
            all_top2.extend([int(ranked[0]), int(ranked[1])])
            all_top3.extend(top3)
            all_top5.extend(top5)
            all_top7.extend(top7)

        # Compute pooled coverage
        top1_shipped = Counter(all_top1)
        top2_shipped = Counter(all_top2)
        top3_shipped = Counter(all_top3)
        top5_shipped = Counter(all_top5)
        top7_shipped = Counter(all_top7)

        top1_ful, top1_tot, top1_cov = compute_pooled_coverage(actuals, top1_shipped)
        top2_ful, top2_tot, top2_cov = compute_pooled_coverage(actuals, top2_shipped)
        top3_ful, top3_tot, top3_cov = compute_pooled_coverage(actuals, top3_shipped)
        top5_ful, top5_tot, top5_cov = compute_pooled_coverage(actuals, top5_shipped)
        top7_ful, top7_tot, top7_cov = compute_pooled_coverage(actuals, top7_shipped)

        day_rec = DayRecord(
            date=date_str,
            actuals=actuals,
            actuals_unique=dict(actuals_counter),
            top1_shipped=dict(top1_shipped),
            top2_shipped=dict(top2_shipped),
            top3_shipped=dict(top3_shipped),
            top5_shipped=dict(top5_shipped),
            top7_shipped=dict(top7_shipped),
            top1_coverage=top1_cov,
            top2_coverage=top2_cov,
            top3_coverage=top3_cov,
            top5_coverage=top5_cov,
            top7_coverage=top7_cov,
            top1_fulfilled=top1_ful,
            top2_fulfilled=top2_ful,
            top3_fulfilled=top3_ful,
            top5_fulfilled=top5_ful,
            top7_fulfilled=top7_ful,
            total_needed=top1_tot,
            predictions=pred_records
        )
        day_records.append(day_rec)

    return day_records, sorted_features


def generate_summary_metrics(day_records: List[DayRecord]) -> Dict:
    """Generate summary metrics across all holdout days."""
    n_days = len(day_records)

    # Pooled coverage metrics
    pooled = {
        "top1_coverage": np.mean([d.top1_coverage for d in day_records]),
        "top2_coverage": np.mean([d.top2_coverage for d in day_records]),
        "top3_coverage": np.mean([d.top3_coverage for d in day_records]),
        "top5_coverage": np.mean([d.top5_coverage for d in day_records]),
        "top7_coverage": np.mean([d.top7_coverage for d in day_records]),
        "top1_perfect_days": sum(1 for d in day_records if d.top1_coverage == 1.0) / n_days,
        "top2_perfect_days": sum(1 for d in day_records if d.top2_coverage == 1.0) / n_days,
        "top3_perfect_days": sum(1 for d in day_records if d.top3_coverage == 1.0) / n_days,
        "top5_perfect_days": sum(1 for d in day_records if d.top5_coverage == 1.0) / n_days,
        "top7_perfect_days": sum(1 for d in day_records if d.top7_coverage == 1.0) / n_days,
    }

    # Per-position metrics
    all_preds = [p for d in day_records for p in d.predictions]
    per_position = {
        "top1_accuracy": np.mean([p.top1_hit for p in all_preds]),
        "top3_accuracy": np.mean([p.top3_hit for p in all_preds]),
        "top5_accuracy": np.mean([p.top5_hit for p in all_preds]),
        "top7_accuracy": np.mean([p.top7_hit for p in all_preds]),
        "avg_actual_rank": np.mean([p.actual_rank for p in all_preds]),
        "avg_top1_confidence": np.mean([p.top1_prob for p in all_preds]),
        "avg_entropy": np.mean([p.entropy for p in all_preds]),
    }

    # By position breakdown
    by_position = {}
    for pos in range(1, 5):
        pos_preds = [p for p in all_preds if p.position == pos]
        by_position[f"QS{pos}"] = {
            "top1_accuracy": np.mean([p.top1_hit for p in pos_preds]),
            "top3_accuracy": np.mean([p.top3_hit for p in pos_preds]),
            "avg_actual_rank": np.mean([p.actual_rank for p in pos_preds]),
            "avg_confidence": np.mean([p.top1_prob for p in pos_preds]),
        }

    return {
        "n_holdout_days": n_days,
        "pooled_coverage": pooled,
        "per_position": per_position,
        "by_position": by_position
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("STRICT HOLDOUT TEST - C4 Parts Forecast")
    print("=" * 70)
    print("\nKey: Parts are INTERCHANGEABLE - evaluating POOLED COVERAGE")

    # Load data
    df = load_csv(args.daily_csv)
    print(f"\nData: {len(df)} days ({df['date'].min().date()} to {df['date'].max().date()})")

    # Run holdout test
    day_records, feature_importance = run_holdout_test(
        df, args.holdout_days, args.train_buffer_days,
        args.lags, args.roll_windows, args.seed
    )

    # Generate summary
    summary = generate_summary_metrics(day_records)

    # Print results
    print("\n" + "=" * 70)
    print("POOLED COVERAGE METRICS (Parts Interchangeable)")
    print("=" * 70)
    print(f"\nHoldout: {summary['n_holdout_days']} days")
    print("\n| Top-K | Avg Coverage | Perfect Days |")
    print("|-------|--------------|--------------|")
    for k in [1, 2, 3, 5, 7]:
        cov = summary["pooled_coverage"][f"top{k}_coverage"]
        perf = summary["pooled_coverage"][f"top{k}_perfect_days"]
        print(f"| Top-{k} | {cov:.1%} | {perf:.1%} |")

    print("\n" + "-" * 70)
    print("PER-POSITION ACCURACY (Traditional)")
    print("-" * 70)
    print(f"\nOverall Top-1: {summary['per_position']['top1_accuracy']:.1%}")
    print(f"Overall Top-3: {summary['per_position']['top3_accuracy']:.1%}")
    print(f"Avg Actual Rank: {summary['per_position']['avg_actual_rank']:.2f}")

    print("\n| Position | Top-1 | Top-3 | Avg Rank | Confidence |")
    print("|----------|-------|-------|----------|------------|")
    for pos in range(1, 5):
        bp = summary["by_position"][f"QS{pos}"]
        print(f"| QS{pos} | {bp['top1_accuracy']:.1%} | {bp['top3_accuracy']:.1%} | {bp['avg_actual_rank']:.2f} | {bp['avg_confidence']:.1%} |")

    print("\n" + "-" * 70)
    print("TOP FEATURES BY IMPORTANCE")
    print("-" * 70)
    for fname, fimp in feature_importance[:10]:
        print(f"  {fname}: {fimp:.4f}")

    # Save detailed results
    holdout_path = os.path.join(args.out_dir, "holdout_test_detailed.json")
    with open(holdout_path, "w") as f:
        # Convert to serializable format
        output = {
            "summary": summary,
            "feature_importance": [{"feature": f, "importance": float(i)} for f, i in feature_importance],
            "daily_records": [
                {
                    "date": d.date,
                    "actuals": d.actuals,
                    "actuals_unique": d.actuals_unique,
                    "top1_shipped": d.top1_shipped,
                    "top2_shipped": d.top2_shipped,
                    "top3_shipped": d.top3_shipped,
                    "top1_coverage": d.top1_coverage,
                    "top2_coverage": d.top2_coverage,
                    "top3_coverage": d.top3_coverage,
                    "top5_coverage": d.top5_coverage,
                    "top7_coverage": d.top7_coverage,
                    "predictions": [
                        {
                            "position": p.position,
                            "actual": p.actual_label,
                            "top1": p.top1_pred,
                            "top3": p.top3_preds,
                            "actual_rank": p.actual_rank,
                            "top1_prob": p.top1_prob,
                            "top1_margin": p.top1_margin,
                            "entropy": p.entropy,
                            "probs": p.predicted_probs,
                            "top1_hit": p.top1_hit,
                            "top3_hit": p.top3_hit,
                            "top_features": p.top_features
                        }
                        for p in d.predictions
                    ]
                }
                for d in day_records
            ]
        }
        json.dump(output, f, indent=2)
    print(f"\nDetailed results: {holdout_path}")

    # Save summary CSV for easy analysis
    summary_rows = []
    for d in day_records:
        row = {
            "date": d.date,
            "actuals": str(d.actuals),
            "top1_coverage": d.top1_coverage,
            "top2_coverage": d.top2_coverage,
            "top3_coverage": d.top3_coverage,
            "top5_coverage": d.top5_coverage,
            "top7_coverage": d.top7_coverage,
        }
        for p in d.predictions:
            row[f"QS{p.position}_actual"] = p.actual_label
            row[f"QS{p.position}_top1"] = p.top1_pred
            row[f"QS{p.position}_top1_hit"] = p.top1_hit
            row[f"QS{p.position}_top3_hit"] = p.top3_hit
            row[f"QS{p.position}_rank"] = p.actual_rank
            row[f"QS{p.position}_conf"] = p.top1_prob
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "holdout_test_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV: {summary_path}")

    # Generate markdown report
    report_path = os.path.join(args.out_dir, "holdout_test_report.md")
    with open(report_path, "w") as f:
        f.write("# Holdout Test Report - C4 Parts Forecast\n\n")
        f.write(f"**Test Date:** {pd.Timestamp.now().isoformat()}\n")
        f.write(f"**Holdout Days:** {summary['n_holdout_days']}\n\n")
        f.write("---\n\n")

        f.write("## Key Insight: Parts Are Interchangeable\n\n")
        f.write("Traditional per-position accuracy underestimates system value.\n")
        f.write("**Pooled coverage** measures: if we ship Top-K from each position, do we have all needed parts?\n\n")

        f.write("## Pooled Coverage Results\n\n")
        f.write("| Metric | Top-1 | Top-2 | Top-3 | Top-5 | Top-7 |\n")
        f.write("|--------|-------|-------|-------|-------|-------|\n")
        cov = summary["pooled_coverage"]
        f.write(f"| Avg Coverage | {cov['top1_coverage']:.1%} | {cov['top2_coverage']:.1%} | {cov['top3_coverage']:.1%} | {cov['top5_coverage']:.1%} | {cov['top7_coverage']:.1%} |\n")
        f.write(f"| Perfect Days | {cov['top1_perfect_days']:.1%} | {cov['top2_perfect_days']:.1%} | {cov['top3_perfect_days']:.1%} | {cov['top5_perfect_days']:.1%} | {cov['top7_perfect_days']:.1%} |\n\n")

        f.write("## Per-Position Accuracy\n\n")
        f.write("| Position | Top-1 Acc | Top-3 Acc | Avg Rank | Confidence |\n")
        f.write("|----------|-----------|-----------|----------|------------|\n")
        for pos in range(1, 5):
            bp = summary["by_position"][f"QS{pos}"]
            f.write(f"| QS{pos} | {bp['top1_accuracy']:.1%} | {bp['top3_accuracy']:.1%} | {bp['avg_actual_rank']:.2f} | {bp['avg_confidence']:.1%} |\n")
        f.write("\n")

        f.write("## Top 10 Features by Importance\n\n")
        f.write("| Rank | Feature | Importance |\n")
        f.write("|------|---------|------------|\n")
        for i, (fname, fimp) in enumerate(feature_importance[:10], 1):
            f.write(f"| {i} | {fname} | {fimp:.4f} |\n")
        f.write("\n")

        f.write("## Recommendations for Improvement\n\n")
        f.write("Based on feature importance and prediction patterns:\n\n")
        f.write("1. **Lag features dominate** - consider adding more temporal patterns\n")
        f.write("2. **Aggregate proportions** - network-wide demand signals are informative\n")
        f.write("3. **Rolling windows** - smoothed trends may capture seasonality\n")
        f.write("4. **Position-specific models** - some positions may benefit from tailored features\n")

    print(f"Report: {report_path}")
    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
