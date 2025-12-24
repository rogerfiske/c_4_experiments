#!/usr/bin/env python3
"""
Part Difficulty Analysis - Full Dataset Version.

Uses walk-forward cross-validation on the entire dataset for robust estimates.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

LABELS = tuple(range(10))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part difficulty analysis - full dataset")
    p.add_argument("--daily_csv", type=str, default="data/raw/CA_4_predict_daily_aggregate.csv")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--test_days", type=int, default=365, help="Days to evaluate (walk-forward)")
    p.add_argument("--retrain_every", type=int, default=30, help="Retrain model every N days")
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
    print("PART DIFFICULTY ANALYSIS - FULL DATASET (Walk-Forward)")
    print("=" * 70)

    df = load_csv(args.daily_csv)
    sup = make_features(df)
    feature_cols = [c for c in sup.columns if c not in ("date", "date_idx", "y", "position")]

    unique_dates = sup["date"].unique()
    n_dates = len(unique_dates)

    # Walk-forward evaluation over last N days
    test_start = n_dates - args.test_days
    test_dates = unique_dates[test_start:]

    print(f"\nDataset: {n_dates} total days")
    print(f"Evaluation period: {args.test_days} days ({test_dates[0].date()} to {test_dates[-1].date()})")
    print(f"Total predictions to evaluate: {args.test_days * 4}")
    print(f"Retraining every {args.retrain_every} days")

    predictions = []
    clf = None
    last_train_idx = -1

    for i, test_date in enumerate(test_dates):
        # Retrain periodically
        if i % args.retrain_every == 0:
            train_end_idx = test_start + i - 30  # 30-day gap
            if train_end_idx > 100:  # Need at least 100 days for training
                train_dates = unique_dates[:train_end_idx]
                train_df = sup[sup["date"].isin(train_dates)].dropna()
                X_train = train_df[feature_cols].values
                y_train = train_df["y"].values

                clf = HistGradientBoostingClassifier(
                    learning_rate=0.05, max_depth=6, max_iter=300,
                    random_state=args.seed, early_stopping=False
                )
                clf.fit(X_train, y_train)
                last_train_idx = train_end_idx

        if clf is None:
            continue

        # Evaluate on test_date
        day_data = sup[sup["date"] == test_date].sort_values("position")
        if len(day_data) != 4:
            continue

        X_day = day_data[feature_cols].values
        probs = clf.predict_proba(X_day)

        for j, pos in enumerate(range(1, 5)):
            actual = int(day_data.iloc[j]["y"])
            ranked = np.argsort(-probs[j])
            pred_top1 = int(ranked[0])
            actual_rank = int(np.where(ranked == actual)[0][0]) + 1

            predictions.append({
                "date": str(test_date.date()),
                "position": pos,
                "actual": actual,
                "pred_top1": pred_top1,
                "actual_rank": actual_rank,
                "top1_hit": actual == pred_top1,
                "top3_hit": actual in ranked[:3],
                "top5_hit": actual in ranked[:5],
            })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{args.test_days} days...")

    pred_df = pd.DataFrame(predictions)
    n_preds = len(pred_df)

    print(f"\n" + "=" * 70)
    print(f"RESULTS (N = {n_preds} predictions)")
    print("=" * 70)

    # ===== Per-Part Analysis =====
    print("\n" + "-" * 70)
    print("PER-PART DIFFICULTY")
    print("-" * 70)

    print("\n| Part | N | Top-1 | Top-3 | Top-5 | Avg Rank | 95% CI (Top-5) |")
    print("|------|-----|-------|-------|-------|----------|----------------|")

    part_stats = {}
    for part in LABELS:
        part_preds = pred_df[pred_df["actual"] == part]
        n = len(part_preds)
        if n < 10:
            continue

        top1 = part_preds["top1_hit"].mean()
        top3 = part_preds["top3_hit"].mean()
        top5 = part_preds["top5_hit"].mean()
        avg_rank = part_preds["actual_rank"].mean()

        # 95% CI for Top-5 (Wilson score interval approximation)
        se = np.sqrt(top5 * (1 - top5) / n)
        ci_low = max(0, top5 - 1.96 * se)
        ci_high = min(1, top5 + 1.96 * se)

        part_stats[part] = {
            "n": n, "top1": top1, "top3": top3, "top5": top5,
            "avg_rank": avg_rank, "ci_low": ci_low, "ci_high": ci_high
        }

        print(f"| {part} | {n:4d} | {top1:.1%} | {top3:.1%} | {top5:.1%} | {avg_rank:.2f} | [{ci_low:.1%}-{ci_high:.1%}] |")

    # ===== Per-Position Analysis =====
    print("\n" + "-" * 70)
    print("PER-POSITION DIFFICULTY")
    print("-" * 70)

    print("\n| Pos | N | Top-1 | Top-3 | Top-5 | Avg Rank | 95% CI (Top-5) |")
    print("|-----|-----|-------|-------|-------|----------|----------------|")

    pos_stats = {}
    for pos in range(1, 5):
        pos_preds = pred_df[pred_df["position"] == pos]
        n = len(pos_preds)

        top1 = pos_preds["top1_hit"].mean()
        top3 = pos_preds["top3_hit"].mean()
        top5 = pos_preds["top5_hit"].mean()
        avg_rank = pos_preds["actual_rank"].mean()

        se = np.sqrt(top5 * (1 - top5) / n)
        ci_low = max(0, top5 - 1.96 * se)
        ci_high = min(1, top5 + 1.96 * se)

        pos_stats[pos] = {
            "n": n, "top1": top1, "top3": top3, "top5": top5,
            "avg_rank": avg_rank, "ci_low": ci_low, "ci_high": ci_high
        }

        print(f"| QS{pos} | {n:4d} | {top1:.1%} | {top3:.1%} | {top5:.1%} | {avg_rank:.2f} | [{ci_low:.1%}-{ci_high:.1%}] |")

    # ===== Part x Position Interaction =====
    print("\n" + "-" * 70)
    print("PART x POSITION INTERACTION (Top-5 Accuracy, N in parentheses)")
    print("-" * 70)

    print("\n           Part:")
    print("Pos   | " + " | ".join(f"  {p}  " for p in LABELS) + " |")
    print("------|" + "|".join(["------"] * 10) + "|")

    interaction = {}
    for pos in range(1, 5):
        row = f"QS{pos}  |"
        interaction[pos] = {}
        for part in LABELS:
            subset = pred_df[(pred_df["position"] == pos) & (pred_df["actual"] == part)]
            n = len(subset)
            if n >= 10:
                acc = subset["top5_hit"].mean()
                interaction[pos][part] = {"n": n, "top5": acc}
                row += f"{acc:4.0%}({n:2d})|"
            else:
                row += f"  -({n:2d})|"
        print(row)

    # ===== Statistical Significance =====
    print("\n" + "-" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("-" * 70)

    if part_stats:
        best_part = max(part_stats.keys(), key=lambda p: part_stats[p]["top5"])
        worst_part = min(part_stats.keys(), key=lambda p: part_stats[p]["top5"])

        best = part_stats[best_part]
        worst = part_stats[worst_part]

        # Check if CIs overlap
        overlap = best["ci_low"] <= worst["ci_high"]

        print(f"\nBest part: {best_part} (Top-5: {best['top5']:.1%}, CI: [{best['ci_low']:.1%}-{best['ci_high']:.1%}])")
        print(f"Worst part: {worst_part} (Top-5: {worst['top5']:.1%}, CI: [{worst['ci_low']:.1%}-{worst['ci_high']:.1%}])")
        print(f"CIs overlap: {'YES - difference may not be significant' if overlap else 'NO - difference is statistically significant'}")

    best_pos = max(pos_stats.keys(), key=lambda p: pos_stats[p]["top5"])
    worst_pos = min(pos_stats.keys(), key=lambda p: pos_stats[p]["top5"])

    best_p = pos_stats[best_pos]
    worst_p = pos_stats[worst_pos]

    overlap_pos = best_p["ci_low"] <= worst_p["ci_high"]

    print(f"\nBest position: QS{best_pos} (Top-5: {best_p['top5']:.1%}, CI: [{best_p['ci_low']:.1%}-{best_p['ci_high']:.1%}])")
    print(f"Worst position: QS{worst_pos} (Top-5: {worst_p['top5']:.1%}, CI: [{worst_p['ci_low']:.1%}-{worst_p['ci_high']:.1%}])")
    print(f"CIs overlap: {'YES - difference may not be significant' if overlap_pos else 'NO - difference is statistically significant'}")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nTotal predictions analyzed: {n_preds}")
    print(f"Evaluation period: {args.test_days} days")

    part_ranking = sorted(part_stats.keys(), key=lambda p: part_stats[p]["top5"], reverse=True)
    print(f"\nPart ranking (easiest to hardest): {' > '.join(str(p) for p in part_ranking)}")

    pos_ranking = sorted(pos_stats.keys(), key=lambda p: pos_stats[p]["top5"], reverse=True)
    print(f"Position ranking: QS{' > QS'.join(str(p) for p in pos_ranking)}")

    # Save results
    output = {
        "n_predictions": n_preds,
        "test_days": args.test_days,
        "part_stats": {str(k): v for k, v in part_stats.items()},
        "position_stats": {f"QS{k}": v for k, v in pos_stats.items()},
        "interaction": {f"QS{pos}": {str(part): v for part, v in parts.items()}
                        for pos, parts in interaction.items()},
        "part_ranking": part_ranking,
        "position_ranking": [f"QS{p}" for p in pos_ranking]
    }

    out_path = os.path.join(args.out_dir, "part_difficulty_analysis_full.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
