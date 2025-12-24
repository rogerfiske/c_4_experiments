#!/usr/bin/env python3
"""
Enhanced Holdout Test for C4 Parts Forecast - Priority 1 Feature Engineering.

Adds:
1. Extended lags: [1, 2, 3, 7, 14, 21, 28, 56, 90]
2. Rolling statistics: 7/14/28-day mean and std
3. Temporal features: day-of-week, month, is_monday
4. Eve data integration: per-part proportions from eve aggregate

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
    p = argparse.ArgumentParser(description="Enhanced holdout test with new features")
    p.add_argument("--daily_csv", type=str, default="data/raw/CA_4_predict_daily_aggregate.csv")
    p.add_argument("--eve_csv", type=str, default="data/raw/CA_4_predict_eve_aggregate.csv")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--holdout_days", type=int, default=100)
    p.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3, 7, 14, 21, 28, 56, 90])
    p.add_argument("--rolling_windows", type=int, nargs="+", default=[7, 14, 28])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df.sort_values("date").reset_index(drop=True)


def make_enhanced_features(
    daily_df: pd.DataFrame,
    eve_df: pd.DataFrame,
    lags: Sequence[int],
    rolling_windows: Sequence[int]
) -> pd.DataFrame:
    """
    Enhanced feature engineering with:
    - Extended lag features
    - Rolling statistics (mean, std)
    - Temporal features (dow, month, is_monday)
    - Eve aggregate proportions
    """
    # Merge eve data with daily on date
    eve_cols = ["date"] + [c for c in eve_df.columns if c.startswith("QS")]
    eve_subset = eve_df[eve_cols].copy()

    # Rename eve columns to avoid collision
    for c in eve_subset.columns:
        if c.startswith("QS"):
            eve_subset.rename(columns={c: f"eve_{c}"}, inplace=True)

    merged = daily_df.merge(eve_subset, on="date", how="left")

    rows = []
    for pos in range(1, 5):
        ca_col = f"CA_QS{pos}"
        tmp = pd.DataFrame({
            "date": merged["date"].values,
            "date_idx": merged.index.values
        })
        tmp["position"] = pos

        # Target: next day's part (shift -1)
        tmp["y"] = merged[ca_col].shift(-1).astype("float")

        # --- LAG FEATURES ---
        for lag in lags:
            tmp[f"ca_lag{lag}"] = merged[ca_col].shift(lag).astype("float")

        # --- ROLLING STATISTICS ---
        for window in rolling_windows:
            # Rolling mean and std on CA column
            tmp[f"ca_roll_mean_{window}"] = merged[ca_col].shift(1).rolling(window).mean().astype("float")
            tmp[f"ca_roll_std_{window}"] = merged[ca_col].shift(1).rolling(window).std().astype("float")

        # --- TEMPORAL FEATURES ---
        tmp["dow"] = merged["date"].dt.dayofweek.astype("float")
        tmp["month"] = merged["date"].dt.month.astype("float")
        tmp["is_monday"] = (merged["date"].dt.dayofweek == 0).astype("float")
        tmp["day_of_month"] = merged["date"].dt.day.astype("float")

        # --- DAILY AGGREGATE FEATURES (from daily CSV) ---
        part_cols = [f"QS{pos}_{lab}" for lab in LABELS]
        denom = merged[part_cols].sum(axis=1).replace(0, np.nan)
        for lab in LABELS:
            tmp[f"agg_count_{lab}_t"] = merged[f"QS{pos}_{lab}"].astype("float")
            tmp[f"agg_prop_{lab}_t"] = (merged[f"QS{pos}_{lab}"] / denom).astype("float")

        # --- EVE AGGREGATE FEATURES ---
        eve_part_cols = [f"eve_QS{pos}_{lab}" for lab in LABELS]
        # Check if eve columns exist
        existing_eve_cols = [c for c in eve_part_cols if c in merged.columns]
        if existing_eve_cols:
            eve_denom = merged[existing_eve_cols].sum(axis=1).replace(0, np.nan)
            for lab in LABELS:
                eve_col = f"eve_QS{pos}_{lab}"
                if eve_col in merged.columns:
                    tmp[f"eve_prop_{lab}_t"] = (merged[eve_col] / eve_denom).astype("float")
                    tmp[f"eve_count_{lab}_t"] = merged[eve_col].astype("float")

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
    print("ENHANCED HOLDOUT TEST - Priority 1 Feature Engineering")
    print("=" * 70)
    print(f"\nNew features:")
    print(f"  - Extended lags: {args.lags}")
    print(f"  - Rolling windows: {args.rolling_windows}")
    print(f"  - Temporal: dow, month, is_monday, day_of_month")
    print(f"  - Eve data: per-part proportions and counts")

    # Load data
    daily_df = load_csv(args.daily_csv)
    eve_df = load_csv(args.eve_csv)
    print(f"\nDaily data: {len(daily_df)} days")
    print(f"Eve data: {len(eve_df)} days")

    # Build enhanced features
    sup = make_enhanced_features(daily_df, eve_df, args.lags, args.rolling_windows)
    feature_cols = [c for c in sup.columns if c not in ("date", "date_idx", "y", "position")]
    print(f"Total features: {len(feature_cols)}")

    # Define holdout
    unique_dates = sup["date"].unique()
    n_dates = len(unique_dates)
    holdout_start = n_dates - args.holdout_days

    holdout_dates = unique_dates[holdout_start:]
    train_dates = unique_dates[:holdout_start - 30]  # 30-day buffer

    train_df = sup[sup["date"].isin(train_dates)].dropna()
    holdout_df = sup[sup["date"].isin(holdout_dates)].dropna()

    print(f"Train: {len(train_dates)} days ({len(train_df)} samples)")
    print(f"Holdout: {len(holdout_dates)} days")

    # Train model
    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values

    print("\nTraining HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6, max_iter=300,
        random_state=args.seed, early_stopping=False
    )
    clf.fit(X_train, y_train)

    # Feature importance
    try:
        feat_imp = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
    except AttributeError:
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
    print("POOLED COVERAGE RESULTS (ENHANCED FEATURES)")
    print("="*70)
    print(f"\nHoldout: {n} days evaluated")
    print("\n| Top-K | Avg Coverage | 100% Days | 75%+ Days | BASELINE |")
    print("|-------|--------------|-----------|-----------|----------|")

    baselines = {1: 0.282, 2: 0.535, 3: 0.695, 5: 0.912, 7: 0.972}
    for k in [1, 2, 3, 5, 7]:
        covs = [r[f"top{k}_coverage"] for r in results]
        avg = np.mean(covs)
        perfect = sum(1 for c in covs if c == 1.0) / n
        good = sum(1 for c in covs if c >= 0.75) / n
        baseline = baselines.get(k, 0)
        delta = avg - baseline
        delta_str = f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}"
        print(f"| Top-{k} | {avg:.1%} | {perfect:.1%} | {good:.1%} | {baseline:.1%} ({delta_str}) |")

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
    print("TOP 15 FEATURES (Enhanced)")
    print("-"*70)
    for f, imp in feat_imp[:15]:
        print(f"  {f}: {imp:.4f}")

    # New features analysis
    print(f"\n{'-'*70}")
    print("NEW FEATURE IMPACT")
    print("-"*70)

    new_feature_types = {
        "Extended lags (3,21,56,90)": [f for f, _ in feat_imp if any(f"lag{l}" in f for l in [3, 21, 56, 90])],
        "Rolling stats": [f for f, _ in feat_imp if "roll_" in f],
        "Temporal": [f for f, _ in feat_imp if f in ["dow", "month", "is_monday", "day_of_month"]],
        "Eve data": [f for f, _ in feat_imp if f.startswith("eve_")]
    }

    for feat_type, feats in new_feature_types.items():
        if feats:
            total_imp = sum(imp for f, imp in feat_imp if f in feats)
            top_feat = max([(f, imp) for f, imp in feat_imp if f in feats], key=lambda x: x[1], default=(None, 0))
            print(f"  {feat_type}: {len(feats)} features, total importance={total_imp:.4f}")
            if top_feat[0]:
                print(f"    Best: {top_feat[0]} ({top_feat[1]:.4f})")

    # Save results
    output = {
        "experiment": "enhanced_features_priority1",
        "features_added": {
            "extended_lags": [3, 21, 56, 90],
            "rolling_windows": list(args.rolling_windows),
            "temporal": ["dow", "month", "is_monday", "day_of_month"],
            "eve_integration": True
        },
        "n_features": len(feature_cols),
        "n_holdout_days": n,
        "pooled_coverage": {
            f"top{k}": {
                "avg": float(np.mean([r[f"top{k}_coverage"] for r in results])),
                "perfect_pct": float(sum(1 for r in results if r[f"top{k}_coverage"] == 1.0) / n),
                "baseline": baselines.get(k, 0),
                "delta": float(np.mean([r[f"top{k}_coverage"] for r in results]) - baselines.get(k, 0))
            } for k in [1, 2, 3, 5, 7]
        },
        "feature_importance": [{"feature": f, "importance": float(i)} for f, i in feat_imp[:30]],
        "daily_results": results
    }

    out_path = os.path.join(args.out_dir, "holdout_enhanced_results.json")
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
        os.path.join(args.out_dir, "holdout_enhanced_summary.csv"), index=False)

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
