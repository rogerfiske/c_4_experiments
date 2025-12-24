#!/usr/bin/env python3
"""
Temporal Stability Analysis for C4 Parts Forecast.

Analyzes whether part/position difficulty patterns are stable over time,
or if they fluctuate (in which case targeted optimization would be futile).

Key questions:
1. Is Part 2 consistently hard, or does difficulty rotate?
2. Are position difficulties stable across time windows?
3. What is the correlation of difficulty rankings between time periods?
4. Should we invest in part-specific models or is it chasing noise?
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy import stats

LABELS = tuple(range(10))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal stability analysis")
    p.add_argument("--daily_csv", type=str, default="data/raw/CA_4_predict_daily_aggregate.csv")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--window_size", type=int, default=60, help="Rolling window size in days")
    p.add_argument("--n_windows", type=int, default=6, help="Number of windows to analyze")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df.sort_values("date").reset_index(drop=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    lags = [1, 2, 3, 7, 14, 21, 28, 56]
    rolling_windows = [7, 14, 28]

    rows = []
    for pos in range(1, 5):
        ca_col = f"CA_QS{pos}"
        tmp = pd.DataFrame({"date": df["date"].values, "date_idx": df.index.values})
        tmp["position"] = pos
        tmp["y"] = df[ca_col].shift(-1).astype("float")

        for lag in lags:
            tmp[f"ca_lag{lag}"] = df[ca_col].shift(lag).astype("float")

        for window in rolling_windows:
            tmp[f"ca_roll_mean_{window}"] = df[ca_col].shift(1).rolling(window).mean().astype("float")

        part_cols = [f"QS{pos}_{lab}" for lab in LABELS]
        denom = df[part_cols].sum(axis=1).replace(0, np.nan)
        for lab in LABELS:
            tmp[f"agg_count_{lab}_t"] = df[f"QS{pos}_{lab}"].astype("float")
            tmp[f"agg_prop_{lab}_t"] = (df[f"QS{pos}_{lab}"] / denom).astype("float")

        rows.append(tmp)

    out = pd.concat(rows).sort_values(["date", "position"]).reset_index(drop=True)
    out = out.dropna(subset=["y"])
    out["y"] = out["y"].astype(int)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 70)
    print("TEMPORAL STABILITY ANALYSIS")
    print("=" * 70)
    print(f"\nWindow size: {args.window_size} days")
    print(f"Number of windows: {args.n_windows}")
    print(f"Total evaluation period: {args.window_size * args.n_windows} days")

    df = load_csv(args.daily_csv)
    sup = make_features(df)
    feature_cols = [c for c in sup.columns if c not in ("date", "date_idx", "y", "position")]

    unique_dates = sup["date"].unique()
    n_dates = len(unique_dates)

    # Define windows (non-overlapping, going backwards from most recent)
    total_test_days = args.window_size * args.n_windows
    test_start = n_dates - total_test_days

    windows = []
    for i in range(args.n_windows):
        start_idx = test_start + i * args.window_size
        end_idx = start_idx + args.window_size
        window_dates = unique_dates[start_idx:end_idx]
        windows.append({
            "idx": i + 1,
            "start": window_dates[0],
            "end": window_dates[-1],
            "dates": window_dates
        })

    print(f"\nWindows defined:")
    for w in windows:
        print(f"  Window {w['idx']}: {w['start'].date()} to {w['end'].date()}")

    # Collect predictions for each window
    window_results = {}

    for w in windows:
        print(f"\nProcessing Window {w['idx']}...")

        # Train on data before this window
        train_end_idx = test_start + (w['idx'] - 1) * args.window_size - 30
        if train_end_idx < 100:
            train_end_idx = test_start - 30

        train_dates = unique_dates[:train_end_idx]
        train_df = sup[sup["date"].isin(train_dates)].dropna()

        X_train = train_df[feature_cols].values
        y_train = train_df["y"].values

        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=6, max_iter=300,
            random_state=args.seed, early_stopping=False
        )
        clf.fit(X_train, y_train)

        # Evaluate on window
        predictions = []
        for test_date in w["dates"]:
            day_data = sup[sup["date"] == test_date].sort_values("position")
            if len(day_data) != 4:
                continue

            X_day = day_data[feature_cols].values
            probs = clf.predict_proba(X_day)

            for j, pos in enumerate(range(1, 5)):
                actual = int(day_data.iloc[j]["y"])
                ranked = np.argsort(-probs[j])
                predictions.append({
                    "position": pos,
                    "actual": actual,
                    "top5_hit": actual in ranked[:5],
                })

        window_results[w['idx']] = pd.DataFrame(predictions)

    # ===== PART STABILITY ANALYSIS =====
    print("\n" + "=" * 70)
    print("PART DIFFICULTY OVER TIME")
    print("=" * 70)

    part_by_window = defaultdict(dict)
    for window_idx, preds in window_results.items():
        for part in LABELS:
            part_preds = preds[preds["actual"] == part]
            if len(part_preds) >= 5:
                part_by_window[part][window_idx] = part_preds["top5_hit"].mean()

    print("\n| Part |", end="")
    for i in range(1, args.n_windows + 1):
        print(f" W{i} |", end="")
    print(" Mean | Std | CV | Stable? |")

    print("|------|", end="")
    for _ in range(args.n_windows):
        print("------|", end="")
    print("------|------|------|---------|")

    part_stability = {}
    for part in LABELS:
        values = [part_by_window[part].get(i, np.nan) for i in range(1, args.n_windows + 1)]
        valid_values = [v for v in values if not np.isnan(v)]

        if len(valid_values) >= 3:
            mean_acc = np.mean(valid_values)
            std_acc = np.std(valid_values)
            cv = std_acc / mean_acc if mean_acc > 0 else 0
            stable = cv < 0.25  # CV < 25% considered stable

            part_stability[part] = {
                "mean": mean_acc,
                "std": std_acc,
                "cv": cv,
                "stable": stable,
                "values": valid_values
            }

            row = f"| {part} |"
            for v in values:
                if np.isnan(v):
                    row += "  -  |"
                else:
                    row += f" {v:.0%} |"
            row += f" {mean_acc:.0%} | {std_acc:.0%} | {cv:.0%} | {'YES' if stable else 'NO'} |"
            print(row)

    # ===== POSITION STABILITY ANALYSIS =====
    print("\n" + "=" * 70)
    print("POSITION DIFFICULTY OVER TIME")
    print("=" * 70)

    pos_by_window = defaultdict(dict)
    for window_idx, preds in window_results.items():
        for pos in range(1, 5):
            pos_preds = preds[preds["position"] == pos]
            if len(pos_preds) >= 10:
                pos_by_window[pos][window_idx] = pos_preds["top5_hit"].mean()

    print("\n| Pos |", end="")
    for i in range(1, args.n_windows + 1):
        print(f"  W{i}  |", end="")
    print(" Mean | Std | CV | Stable? |")

    print("|-----|", end="")
    for _ in range(args.n_windows):
        print("-------|", end="")
    print("------|------|------|---------|")

    pos_stability = {}
    for pos in range(1, 5):
        values = [pos_by_window[pos].get(i, np.nan) for i in range(1, args.n_windows + 1)]
        valid_values = [v for v in values if not np.isnan(v)]

        if len(valid_values) >= 3:
            mean_acc = np.mean(valid_values)
            std_acc = np.std(valid_values)
            cv = std_acc / mean_acc if mean_acc > 0 else 0
            stable = cv < 0.15

            pos_stability[pos] = {
                "mean": mean_acc,
                "std": std_acc,
                "cv": cv,
                "stable": stable,
                "values": valid_values
            }

            row = f"| QS{pos} |"
            for v in values:
                if np.isnan(v):
                    row += "   -   |"
                else:
                    row += f" {v:.1%} |"
            row += f" {mean_acc:.0%} | {std_acc:.0%} | {cv:.0%} | {'YES' if stable else 'NO'} |"
            print(row)

    # ===== RANKING STABILITY (Spearman correlation) =====
    print("\n" + "=" * 70)
    print("RANKING STABILITY (Are hard/easy parts consistent?)")
    print("=" * 70)

    # Build ranking matrix for parts
    part_rankings = {}
    for window_idx in range(1, args.n_windows + 1):
        window_acc = {p: part_by_window[p].get(window_idx, 0) for p in LABELS}
        # Rank parts by accuracy (1 = best)
        sorted_parts = sorted(window_acc.keys(), key=lambda p: window_acc[p], reverse=True)
        part_rankings[window_idx] = {p: rank + 1 for rank, p in enumerate(sorted_parts)}

    print("\nPart rankings by window (1=easiest, 10=hardest):")
    print("| Part |", end="")
    for i in range(1, args.n_windows + 1):
        print(f" W{i} |", end="")
    print(" Avg Rank | Rank Std |")

    print("|------|", end="")
    for _ in range(args.n_windows):
        print("----|", end="")
    print("----------|----------|")

    for part in LABELS:
        ranks = [part_rankings[w].get(part, 5.5) for w in range(1, args.n_windows + 1)]
        row = f"| {part} |"
        for r in ranks:
            row += f" {r:2.0f} |"
        row += f" {np.mean(ranks):5.1f} | {np.std(ranks):5.1f} |"
        print(row)

    # Spearman correlation between consecutive windows
    print("\nSpearman rank correlation between consecutive windows:")
    correlations = []
    for i in range(1, args.n_windows):
        ranks_i = [part_rankings[i][p] for p in LABELS]
        ranks_j = [part_rankings[i + 1][p] for p in LABELS]
        corr, pval = stats.spearmanr(ranks_i, ranks_j)
        correlations.append(corr)
        sig = "**" if pval < 0.05 else ""
        print(f"  W{i} vs W{i+1}: rho = {corr:.3f} (p={pval:.3f}){sig}")

    avg_corr = np.mean(correlations)
    print(f"\nAverage rank correlation: {avg_corr:.3f}")
    if avg_corr > 0.6:
        print("INTERPRETATION: Rankings are STABLE - part difficulty is consistent")
    elif avg_corr > 0.3:
        print("INTERPRETATION: Rankings are MODERATELY stable - some patterns exist")
    else:
        print("INTERPRETATION: Rankings are UNSTABLE - difficulty rotates randomly")

    # ===== CONSISTENTLY HARD/EASY PARTS =====
    print("\n" + "=" * 70)
    print("CONSISTENTLY HARD vs EASY PARTS")
    print("=" * 70)

    consistently_hard = []
    consistently_easy = []
    volatile = []

    for part, stability in part_stability.items():
        if stability["stable"]:
            if stability["mean"] < 0.45:
                consistently_hard.append((part, stability["mean"]))
            elif stability["mean"] > 0.55:
                consistently_easy.append((part, stability["mean"]))
        else:
            volatile.append((part, stability["cv"]))

    print(f"\nConsistently HARD parts (stable + low accuracy):")
    for p, acc in sorted(consistently_hard, key=lambda x: x[1]):
        print(f"  Part {p}: {acc:.1%} avg (WORTH TARGETING)")

    print(f"\nConsistently EASY parts (stable + high accuracy):")
    for p, acc in sorted(consistently_easy, key=lambda x: -x[1]):
        print(f"  Part {p}: {acc:.1%} avg")

    print(f"\nVOLATILE parts (high variance - DON'T target):")
    for p, cv in sorted(volatile, key=lambda x: -x[1]):
        print(f"  Part {p}: CV={cv:.1%} (unstable)")

    # ===== RECOMMENDATIONS =====
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    print("\n1. TEMPORAL STABILITY VERDICT:")
    if avg_corr > 0.5 and len(consistently_hard) > 0:
        print("   Part difficulty is STABLE - targeted improvements ARE worthwhile")
        print(f"   Focus on: Parts {[p for p, _ in consistently_hard]}")
    elif avg_corr > 0.3:
        print("   Part difficulty is MODERATELY stable - cautious optimization OK")
    else:
        print("   Part difficulty is UNSTABLE - avoid part-specific models")
        print("   Instead focus on: overall model improvements, feature engineering")

    print("\n2. POSITION VERDICT:")
    pos_stable_count = sum(1 for s in pos_stability.values() if s["stable"])
    if pos_stable_count >= 3:
        print("   Positions are stable - position-specific tuning may help")
    else:
        print("   Positions are variable - use pooled model")

    print("\n3. METRICS TO TRACK (Data Collection Plan):")
    print("   - Per-part Top-5 accuracy (rolling 30-day window)")
    print("   - Per-position Top-5 accuracy (rolling 30-day window)")
    print("   - Rank correlation with previous period")
    print("   - Coefficient of variation for each part/position")
    print("   - Feature importance drift")

    # Save results
    output = {
        "window_size_days": args.window_size,
        "n_windows": args.n_windows,
        "part_stability": {str(k): {kk: float(vv) if isinstance(vv, (np.floating, float)) else (bool(vv) if isinstance(vv, np.bool_) else vv)
                                     for kk, vv in v.items() if kk != 'values'}
                          for k, v in part_stability.items()},
        "position_stability": {f"QS{k}": {kk: float(vv) if isinstance(vv, (np.floating, float)) else (bool(vv) if isinstance(vv, np.bool_) else vv)
                                           for kk, vv in v.items() if kk != 'values'}
                               for k, v in pos_stability.items()},
        "avg_rank_correlation": float(avg_corr),
        "consistently_hard_parts": [int(p) for p, _ in consistently_hard],
        "volatile_parts": [int(p) for p, _ in volatile],
        "recommendations": {
            "target_parts": [int(p) for p, _ in consistently_hard] if avg_corr > 0.5 else [],
            "stability_verdict": "stable" if avg_corr > 0.5 else "moderate" if avg_corr > 0.3 else "unstable"
        }
    }

    out_path = os.path.join(args.out_dir, "temporal_stability_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
