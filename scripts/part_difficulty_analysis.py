#!/usr/bin/env python3
"""
Part Difficulty Analysis for C4 Parts Forecast.

Analyzes:
1. Which parts (0-9) are hardest to predict overall
2. Which positions (QS1-QS4) are hardest to predict
3. Part difficulty by position (interaction effects)
4. Class distribution and prediction bias
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
    p = argparse.ArgumentParser(description="Part difficulty analysis")
    p.add_argument("--daily_csv", type=str, default="data/raw/CA_4_predict_daily_aggregate.csv")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--holdout_days", type=int, default=100)
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
    print("PART DIFFICULTY ANALYSIS")
    print("=" * 70)

    df = load_csv(args.daily_csv)
    sup = make_features(df)
    feature_cols = [c for c in sup.columns if c not in ("date", "date_idx", "y", "position")]

    unique_dates = sup["date"].unique()
    n_dates = len(unique_dates)
    holdout_start = n_dates - args.holdout_days
    holdout_dates = unique_dates[holdout_start:]
    train_dates = unique_dates[:holdout_start - 30]

    train_df = sup[sup["date"].isin(train_dates)].dropna()
    holdout_df = sup[sup["date"].isin(holdout_dates)].dropna()

    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values

    print(f"\nTraining model...")
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6, max_iter=300,
        random_state=args.seed, early_stopping=False
    )
    clf.fit(X_train, y_train)

    # Collect predictions
    predictions = []
    for date in holdout_dates:
        day_data = holdout_df[holdout_df["date"] == date].sort_values("position")
        if len(day_data) != 4:
            continue

        X_day = day_data[feature_cols].values
        probs = clf.predict_proba(X_day)

        for i, pos in enumerate(range(1, 5)):
            actual = int(day_data.iloc[i]["y"])
            ranked = np.argsort(-probs[i])
            pred_top1 = int(ranked[0])
            pred_top3 = [int(ranked[j]) for j in range(3)]
            pred_top5 = [int(ranked[j]) for j in range(5)]
            actual_rank = int(np.where(ranked == actual)[0][0]) + 1
            actual_prob = float(probs[i, actual])
            top1_prob = float(probs[i, pred_top1])

            predictions.append({
                "date": str(date.date()) if hasattr(date, 'date') else str(date)[:10],
                "position": pos,
                "actual": actual,
                "pred_top1": pred_top1,
                "pred_top3": pred_top3,
                "pred_top5": pred_top5,
                "actual_rank": actual_rank,
                "actual_prob": actual_prob,
                "top1_prob": top1_prob,
                "top1_hit": actual == pred_top1,
                "top3_hit": actual in pred_top3,
                "top5_hit": actual in pred_top5,
            })

    pred_df = pd.DataFrame(predictions)

    # ===== ANALYSIS 1: Class Distribution =====
    print("\n" + "=" * 70)
    print("1. CLASS DISTRIBUTION (Holdout Set)")
    print("=" * 70)

    print("\n| Part | Count | Freq | Train Freq |")
    print("|------|-------|------|------------|")

    holdout_dist = pred_df["actual"].value_counts().sort_index()
    train_dist = pd.Series(y_train).value_counts().sort_index()
    train_total = len(y_train)
    holdout_total = len(pred_df)

    for part in LABELS:
        h_count = holdout_dist.get(part, 0)
        h_freq = h_count / holdout_total
        t_freq = train_dist.get(part, 0) / train_total
        print(f"| {part} | {h_count:4d} | {h_freq:.1%} | {t_freq:.1%} |")

    # ===== ANALYSIS 2: Per-Part Difficulty =====
    print("\n" + "=" * 70)
    print("2. PER-PART PREDICTION DIFFICULTY")
    print("=" * 70)

    print("\n| Part | Count | Top-1 Acc | Top-3 Acc | Top-5 Acc | Avg Rank | Avg Prob |")
    print("|------|-------|-----------|-----------|-----------|----------|----------|")

    part_stats = {}
    for part in LABELS:
        part_preds = pred_df[pred_df["actual"] == part]
        if len(part_preds) == 0:
            continue

        count = len(part_preds)
        top1_acc = part_preds["top1_hit"].mean()
        top3_acc = part_preds["top3_hit"].mean()
        top5_acc = part_preds["top5_hit"].mean()
        avg_rank = part_preds["actual_rank"].mean()
        avg_prob = part_preds["actual_prob"].mean()

        part_stats[part] = {
            "count": count,
            "top1_acc": top1_acc,
            "top3_acc": top3_acc,
            "top5_acc": top5_acc,
            "avg_rank": avg_rank,
            "avg_prob": avg_prob
        }

        print(f"| {part} | {count:4d} | {top1_acc:.1%} | {top3_acc:.1%} | {top5_acc:.1%} | {avg_rank:.2f} | {avg_prob:.1%} |")

    # Find easiest and hardest parts
    if part_stats:
        easiest = max(part_stats.keys(), key=lambda p: part_stats[p]["top5_acc"])
        hardest = min(part_stats.keys(), key=lambda p: part_stats[p]["top5_acc"])
        print(f"\nEASIEST part to predict: {easiest} (Top-5: {part_stats[easiest]['top5_acc']:.1%})")
        print(f"HARDEST part to predict: {hardest} (Top-5: {part_stats[hardest]['top5_acc']:.1%})")

    # ===== ANALYSIS 3: Per-Position Difficulty =====
    print("\n" + "=" * 70)
    print("3. PER-POSITION PREDICTION DIFFICULTY (QS1-QS4)")
    print("=" * 70)

    print("\n| Pos | Top-1 Acc | Top-3 Acc | Top-5 Acc | Avg Rank | Avg Conf |")
    print("|-----|-----------|-----------|-----------|----------|----------|")

    pos_stats = {}
    for pos in range(1, 5):
        pos_preds = pred_df[pred_df["position"] == pos]

        top1_acc = pos_preds["top1_hit"].mean()
        top3_acc = pos_preds["top3_hit"].mean()
        top5_acc = pos_preds["top5_hit"].mean()
        avg_rank = pos_preds["actual_rank"].mean()
        avg_conf = pos_preds["top1_prob"].mean()

        pos_stats[pos] = {
            "top1_acc": top1_acc,
            "top3_acc": top3_acc,
            "top5_acc": top5_acc,
            "avg_rank": avg_rank,
            "avg_conf": avg_conf
        }

        print(f"| QS{pos} | {top1_acc:.1%} | {top3_acc:.1%} | {top5_acc:.1%} | {avg_rank:.2f} | {avg_conf:.1%} |")

    easiest_pos = max(pos_stats.keys(), key=lambda p: pos_stats[p]["top5_acc"])
    hardest_pos = min(pos_stats.keys(), key=lambda p: pos_stats[p]["top5_acc"])
    print(f"\nEASIEST position: QS{easiest_pos} (Top-5: {pos_stats[easiest_pos]['top5_acc']:.1%})")
    print(f"HARDEST position: QS{hardest_pos} (Top-5: {pos_stats[hardest_pos]['top5_acc']:.1%})")

    # ===== ANALYSIS 4: Part x Position Interaction =====
    print("\n" + "=" * 70)
    print("4. PART x POSITION INTERACTION (Top-5 Accuracy)")
    print("=" * 70)

    print("\n           Part:")
    print("Pos   | " + " | ".join(f" {p} " for p in LABELS) + " |")
    print("------|" + "|".join(["-----"] * 10) + "|")

    interaction = defaultdict(dict)
    for pos in range(1, 5):
        row = f"QS{pos}  |"
        for part in LABELS:
            subset = pred_df[(pred_df["position"] == pos) & (pred_df["actual"] == part)]
            if len(subset) >= 3:  # Need at least 3 samples
                acc = subset["top5_hit"].mean()
                interaction[pos][part] = acc
                row += f" {acc:.0%} |"
            else:
                row += "  -  |"
        print(row)

    # ===== ANALYSIS 5: Prediction Bias =====
    print("\n" + "=" * 70)
    print("5. PREDICTION BIAS ANALYSIS")
    print("=" * 70)

    print("\n| Part | Actual % | Predicted % | Bias |")
    print("|------|----------|-------------|------|")

    pred_dist = pred_df["pred_top1"].value_counts().sort_index()
    for part in LABELS:
        actual_pct = holdout_dist.get(part, 0) / holdout_total
        pred_pct = pred_dist.get(part, 0) / holdout_total
        bias = pred_pct - actual_pct
        bias_str = f"+{bias:.1%}" if bias > 0 else f"{bias:.1%}"
        print(f"| {part} | {actual_pct:.1%} | {pred_pct:.1%} | {bias_str} |")

    # ===== ANALYSIS 6: Confusion Pattern =====
    print("\n" + "=" * 70)
    print("6. COMMON CONFUSION PATTERNS (When Top-1 is wrong)")
    print("=" * 70)

    wrong_preds = pred_df[~pred_df["top1_hit"]]
    confusion_pairs = Counter(zip(wrong_preds["actual"], wrong_preds["pred_top1"]))

    print("\n| Actual | Predicted | Count | % of Errors |")
    print("|--------|-----------|-------|-------------|")

    total_errors = len(wrong_preds)
    for (actual, pred), count in confusion_pairs.most_common(15):
        pct = count / total_errors
        print(f"| {actual} | {pred} | {count:4d} | {pct:.1%} |")

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Part difficulty ranking
    if part_stats:
        part_ranking = sorted(part_stats.keys(), key=lambda p: part_stats[p]["top5_acc"], reverse=True)
        print(f"\nPart difficulty ranking (easiest to hardest by Top-5):")
        print(f"  {' > '.join(str(p) for p in part_ranking)}")

    # Position difficulty ranking
    pos_ranking = sorted(pos_stats.keys(), key=lambda p: pos_stats[p]["top5_acc"], reverse=True)
    print(f"\nPosition difficulty ranking (easiest to hardest by Top-5):")
    print(f"  QS{' > QS'.join(str(p) for p in pos_ranking)}")

    # Save results
    output = {
        "class_distribution": {
            "holdout": {str(k): int(v) for k, v in holdout_dist.items()},
            "train": {str(k): int(v) for k, v in train_dist.items()}
        },
        "part_stats": {str(k): v for k, v in part_stats.items()},
        "position_stats": {f"QS{k}": v for k, v in pos_stats.items()},
        "part_ranking": part_ranking if part_stats else [],
        "position_ranking": [f"QS{p}" for p in pos_ranking],
        "confusion_top15": [{"actual": a, "predicted": p, "count": c}
                           for (a, p), c in confusion_pairs.most_common(15)]
    }

    out_path = os.path.join(args.out_dir, "part_difficulty_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
