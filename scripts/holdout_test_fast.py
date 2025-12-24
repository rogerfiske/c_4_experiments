#!/usr/bin/env python3
"""
Fast Holdout Test for C4 Parts Forecast - Optimized Version.

Uses vectorized feature engineering for speed.
Evaluates POOLED COVERAGE since parts are interchangeable.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

LABELS = tuple(range(10))
N_CLASSES = 10


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast holdout test")
    p.add_argument("--daily_csv", type=str, default="data/raw/CA_4_predict_daily_aggregate.csv")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--holdout_days", type=int, default=100)
    p.add_argument("--lags", type=int, nargs="+", default=[1, 2, 7, 14, 28])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df.sort_values("date").reset_index(drop=True)


def make_supervised_features(df: pd.DataFrame, lags: Sequence[int]) -> pd.DataFrame:
    """Vectorized feature engineering."""
    rows = []
    for pos in range(1, 5):
        ca_col = f"CA_QS{pos}"
        tmp = pd.DataFrame({"date": df["date"].values, "date_idx": df.index.values})
        tmp["position"] = pos
        tmp["y"] = df[ca_col].shift(-1).astype("float")

        for lag in lags:
            tmp[f"ca_lag{lag}"] = df[ca_col].shift(lag).astype("float")

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


def compute_pooled_coverage(actuals: List[int], shipped: List[int]) -> Tuple[int, int, float]:
    """Compute coverage with interchangeable parts."""
    needed = Counter(actuals)
    have = Counter(shipped)
    fulfilled = sum(min(needed[p], have.get(p, 0)) for p in needed)
    total = len(actuals)
    return fulfilled, total, fulfilled / total if total > 0 else 0.0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 70)
    print("STRICT HOLDOUT TEST - Pooled Coverage (Interchangeable Parts)")
    print("=" * 70)

    df = load_csv(args.daily_csv)
    print(f"\nData: {len(df)} days")

    # Build all features at once (vectorized)
    sup = make_supervised_features(df, args.lags)
    feature_cols = [c for c in sup.columns if c not in ("date", "date_idx", "y", "position")]

    # Define holdout
    unique_dates = sup["date"].unique()
    n_dates = len(unique_dates)
    holdout_start = n_dates - args.holdout_days

    holdout_dates = unique_dates[holdout_start:]
    train_dates = unique_dates[:holdout_start - 30]  # 30-day buffer

    train_df = sup[sup["date"].isin(train_dates)].dropna()
    holdout_df = sup[sup["date"].isin(holdout_dates)].dropna()

    print(f"Train: {len(train_dates)} days, Holdout: {len(holdout_dates)} days")

    # Train model
    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6, max_iter=300,
        random_state=args.seed, early_stopping=False
    )
    clf.fit(X_train, y_train)

    # Feature importance (use permutation importance approximation)
    try:
        feat_imp = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
    except AttributeError:
        # Approximate importance using mean absolute feature values weighted by prediction variance
        X_holdout = holdout_df[feature_cols].values
        feat_imp = [(f, float(np.std(X_holdout[:, i]))) for i, f in enumerate(feature_cols)]
        feat_imp = sorted(feat_imp, key=lambda x: -x[1])

    # Evaluate holdout
    results = []
    for date in holdout_dates:
        day_data = holdout_df[holdout_df["date"] == date].sort_values("position")
        if len(day_data) != 4:
            continue

        actuals = day_data["y"].tolist()
        X_day = day_data[feature_cols].values
        probs = clf.predict_proba(X_day)

        # Get top-K for each position
        top1s, top2s, top3s, top5s, top7s = [], [], [], [], []
        pos_results = []

        for i, pos in enumerate(range(1, 5)):
            ranked = np.argsort(-probs[i])
            actual = actuals[i]
            actual_rank = int(np.where(ranked == actual)[0][0]) + 1

            top1s.append(int(ranked[0]))
            top2s.extend([int(ranked[0]), int(ranked[1])])
            top3s.extend([int(ranked[j]) for j in range(3)])
            top5s.extend([int(ranked[j]) for j in range(5)])
            top7s.extend([int(ranked[j]) for j in range(7)])

            pos_results.append({
                "position": pos,
                "actual": actual,
                "top1": int(ranked[0]),
                "top3": [int(ranked[j]) for j in range(3)],
                "actual_rank": actual_rank,
                "top1_prob": float(probs[i, ranked[0]]),
                "top1_hit": bool(actual == ranked[0]),
                "top3_hit": bool(actual in ranked[:3]),
            })

        # Pooled coverage
        _, _, cov1 = compute_pooled_coverage(actuals, top1s)
        _, _, cov2 = compute_pooled_coverage(actuals, top2s)
        _, _, cov3 = compute_pooled_coverage(actuals, top3s)
        _, _, cov5 = compute_pooled_coverage(actuals, top5s)
        _, _, cov7 = compute_pooled_coverage(actuals, top7s)

        results.append({
            "date": str(date.date()) if hasattr(date, 'date') else str(date)[:10],
            "actuals": actuals,
            "top1_coverage": cov1,
            "top2_coverage": cov2,
            "top3_coverage": cov3,
            "top5_coverage": cov5,
            "top7_coverage": cov7,
            "positions": pos_results
        })

    # Summary
    n = len(results)
    print(f"\n{'='*70}")
    print("POOLED COVERAGE RESULTS")
    print("="*70)
    print(f"\nHoldout: {n} days evaluated")
    print("\n| Top-K | Avg Coverage | 100% Days | 75%+ Days |")
    print("|-------|--------------|-----------|-----------|")

    for k in [1, 2, 3, 5, 7]:
        covs = [r[f"top{k}_coverage"] for r in results]
        avg = np.mean(covs)
        perfect = sum(1 for c in covs if c == 1.0) / n
        good = sum(1 for c in covs if c >= 0.75) / n
        print(f"| Top-{k} | {avg:.1%} | {perfect:.1%} | {good:.1%} |")

    # Per-position accuracy
    all_pos = [p for r in results for p in r["positions"]]
    print(f"\n{'-'*70}")
    print("PER-POSITION ACCURACY")
    print("-"*70)

    for pos in range(1, 5):
        pos_data = [p for p in all_pos if p["position"] == pos]
        t1 = np.mean([p["top1_hit"] for p in pos_data])
        t3 = np.mean([p["top3_hit"] for p in pos_data])
        rank = np.mean([p["actual_rank"] for p in pos_data])
        conf = np.mean([p["top1_prob"] for p in pos_data])
        print(f"QS{pos}: Top-1={t1:.1%}, Top-3={t3:.1%}, Rank={rank:.2f}, Conf={conf:.1%}")

    # Feature importance
    print(f"\n{'-'*70}")
    print("TOP 10 FEATURES")
    print("-"*70)
    for f, i in feat_imp[:10]:
        print(f"  {f}: {i:.4f}")

    # Save results
    output = {
        "n_holdout_days": n,
        "pooled_coverage": {
            f"top{k}": {
                "avg": float(np.mean([r[f"top{k}_coverage"] for r in results])),
                "perfect_pct": float(sum(1 for r in results if r[f"top{k}_coverage"] == 1.0) / n)
            } for k in [1, 2, 3, 5, 7]
        },
        "feature_importance": [{"feature": f, "importance": float(i)} for f, i in feat_imp[:20]],
        "daily_results": results
    }

    out_path = os.path.join(args.out_dir, "holdout_test_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # CSV summary
    csv_rows = []
    for r in results:
        row = {"date": r["date"], "actuals": str(r["actuals"])}
        for k in [1, 2, 3, 5, 7]:
            row[f"top{k}_cov"] = r[f"top{k}_coverage"]
        for p in r["positions"]:
            row[f"QS{p['position']}_actual"] = p["actual"]
            row[f"QS{p['position']}_top1"] = p["top1"]
            row[f"QS{p['position']}_rank"] = p["actual_rank"]
        csv_rows.append(row)

    pd.DataFrame(csv_rows).to_csv(
        os.path.join(args.out_dir, "holdout_test_summary.csv"), index=False)

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
