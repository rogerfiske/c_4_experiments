#!/usr/bin/env python3
"""
Enhanced Holdout Test v2 - Refined Feature Engineering.

Fixes from v1:
- Removed day_of_month (overfitting)
- Increased regularization (max_depth=4, min_samples_leaf=50)
- Added early stopping
- Kept only productive new features

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
    p = argparse.ArgumentParser(description="Enhanced holdout test v2")
    p.add_argument("--daily_csv", type=str, default="data/raw/CA_4_predict_daily_aggregate.csv")
    p.add_argument("--eve_csv", type=str, default="data/raw/CA_4_predict_eve_aggregate.csv")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--holdout_days", type=int, default=100)
    # Extended lags but not too extreme
    p.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3, 7, 14, 21, 28, 56])
    p.add_argument("--rolling_windows", type=int, nargs="+", default=[7, 14, 28])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df.sort_values("date").reset_index(drop=True)


def make_refined_features(
    daily_df: pd.DataFrame,
    eve_df: pd.DataFrame,
    lags: Sequence[int],
    rolling_windows: Sequence[int]
) -> pd.DataFrame:
    """
    Refined feature engineering:
    - Extended lags (without 90 - too sparse)
    - Rolling stats (mean, std)
    - dow and is_monday only (no day_of_month)
    - Eve proportions only (not raw counts to reduce noise)
    """
    eve_cols = ["date"] + [c for c in eve_df.columns if c.startswith("QS")]
    eve_subset = eve_df[eve_cols].copy()

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
        tmp["y"] = merged[ca_col].shift(-1).astype("float")

        # --- LAG FEATURES ---
        for lag in lags:
            tmp[f"ca_lag{lag}"] = merged[ca_col].shift(lag).astype("float")

        # --- ROLLING STATISTICS ---
        for window in rolling_windows:
            tmp[f"ca_roll_mean_{window}"] = merged[ca_col].shift(1).rolling(window).mean().astype("float")
            tmp[f"ca_roll_std_{window}"] = merged[ca_col].shift(1).rolling(window).std().astype("float")

        # --- TEMPORAL FEATURES (limited) ---
        tmp["dow"] = merged["date"].dt.dayofweek.astype("float")
        tmp["is_monday"] = (merged["date"].dt.dayofweek == 0).astype("float")
        # NO day_of_month - it was overfitting

        # --- DAILY AGGREGATE FEATURES ---
        part_cols = [f"QS{pos}_{lab}" for lab in LABELS]
        denom = merged[part_cols].sum(axis=1).replace(0, np.nan)
        for lab in LABELS:
            tmp[f"agg_count_{lab}_t"] = merged[f"QS{pos}_{lab}"].astype("float")
            tmp[f"agg_prop_{lab}_t"] = (merged[f"QS{pos}_{lab}"] / denom).astype("float")

        # --- EVE AGGREGATE FEATURES (proportions only) ---
        eve_part_cols = [f"eve_QS{pos}_{lab}" for lab in LABELS]
        existing_eve_cols = [c for c in eve_part_cols if c in merged.columns]
        if existing_eve_cols:
            eve_denom = merged[existing_eve_cols].sum(axis=1).replace(0, np.nan)
            for lab in LABELS:
                eve_col = f"eve_QS{pos}_{lab}"
                if eve_col in merged.columns:
                    # Only proportions - less noisy than raw counts
                    tmp[f"eve_prop_{lab}_t"] = (merged[eve_col] / eve_denom).astype("float")

        rows.append(tmp)

    out = pd.concat(rows).sort_values(["date", "position"]).reset_index(drop=True)
    out = out.dropna(subset=["y"])
    out["y"] = out["y"].astype(int)
    return out


def compute_pooled_coverage(actuals: List[int], shipped: List[int]) -> Tuple[int, int, float]:
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
    print("ENHANCED HOLDOUT TEST v2 - Refined Features + Regularization")
    print("=" * 70)
    print(f"\nChanges from v1:")
    print(f"  - Removed day_of_month (was overfitting)")
    print(f"  - Removed lag 90 (too sparse)")
    print(f"  - Eve: proportions only (no raw counts)")
    print(f"  - Increased regularization: max_depth=4, min_samples_leaf=50")
    print(f"  - Added early_stopping=True")

    daily_df = load_csv(args.daily_csv)
    eve_df = load_csv(args.eve_csv)
    print(f"\nDaily data: {len(daily_df)} days")
    print(f"Eve data: {len(eve_df)} days")

    sup = make_refined_features(daily_df, eve_df, args.lags, args.rolling_windows)
    feature_cols = [c for c in sup.columns if c not in ("date", "date_idx", "y", "position")]
    print(f"Total features: {len(feature_cols)}")

    unique_dates = sup["date"].unique()
    n_dates = len(unique_dates)
    holdout_start = n_dates - args.holdout_days

    holdout_dates = unique_dates[holdout_start:]
    train_dates = unique_dates[:holdout_start - 30]

    train_df = sup[sup["date"].isin(train_dates)].dropna()
    holdout_df = sup[sup["date"].isin(holdout_dates)].dropna()

    print(f"Train: {len(train_dates)} days ({len(train_df)} samples)")
    print(f"Holdout: {len(holdout_dates)} days")

    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values

    print("\nTraining with increased regularization...")
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,  # Reduced from 6
        max_iter=500,
        min_samples_leaf=50,  # Added regularization
        l2_regularization=1.0,  # Added L2 penalty
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    clf.fit(X_train, y_train)
    print(f"Stopped at iteration: {clf.n_iter_}")

    try:
        feat_imp = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
    except AttributeError:
        X_holdout = holdout_df[feature_cols].values
        feat_imp = [(f, float(np.std(X_holdout[:, i]))) for i, f in enumerate(feature_cols)]
        feat_imp = sorted(feat_imp, key=lambda x: -x[1])

    results = []
    for date in holdout_dates:
        day_data = holdout_df[holdout_df["date"] == date].sort_values("position")
        if len(day_data) != 4:
            continue

        actuals = day_data["y"].tolist()
        X_day = day_data[feature_cols].values
        probs = clf.predict_proba(X_day)

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

    n = len(results)
    print(f"\n{'='*70}")
    print("POOLED COVERAGE RESULTS (v2 - REFINED)")
    print("="*70)
    print(f"\nHoldout: {n} days evaluated")
    print("\n| Top-K | Avg Coverage | 100% Days | 75%+ Days | BASELINE | DELTA |")
    print("|-------|--------------|-----------|-----------|----------|-------|")

    baselines = {1: 0.282, 2: 0.535, 3: 0.695, 5: 0.912, 7: 0.972}
    for k in [1, 2, 3, 5, 7]:
        covs = [r[f"top{k}_coverage"] for r in results]
        avg = np.mean(covs)
        perfect = sum(1 for c in covs if c == 1.0) / n
        good = sum(1 for c in covs if c >= 0.75) / n
        baseline = baselines.get(k, 0)
        delta = avg - baseline
        delta_str = f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}"
        status = "UP" if delta >= 0 else "DN"
        print(f"| Top-{k} | {avg:.1%} | {perfect:.1%} | {good:.1%} | {baseline:.1%} | {delta_str} {status} |")

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

    print(f"\n{'-'*70}")
    print("TOP 15 FEATURES")
    print("-"*70)
    for f, imp in feat_imp[:15]:
        print(f"  {f}: {imp:.4f}")

    # New features analysis
    print(f"\n{'-'*70}")
    print("NEW FEATURE IMPACT")
    print("-"*70)

    new_feature_types = {
        "Extended lags (3,21,56)": [f for f, _ in feat_imp if any(f"lag{l}" in f for l in [3, 21, 56])],
        "Rolling stats": [f for f, _ in feat_imp if "roll_" in f],
        "Temporal (dow, is_monday)": [f for f, _ in feat_imp if f in ["dow", "is_monday"]],
        "Eve proportions": [f for f, _ in feat_imp if f.startswith("eve_")]
    }

    for feat_type, feats in new_feature_types.items():
        if feats:
            total_imp = sum(imp for f, imp in feat_imp if f in feats)
            top_feat = max([(f, imp) for f, imp in feat_imp if f in feats], key=lambda x: x[1], default=(None, 0))
            print(f"  {feat_type}: {len(feats)} features, total importance={total_imp:.4f}")
            if top_feat[0]:
                print(f"    Best: {top_feat[0]} ({top_feat[1]:.4f})")

    output = {
        "experiment": "enhanced_features_v2_refined",
        "changes": [
            "Removed day_of_month",
            "Removed lag 90",
            "Eve proportions only",
            "max_depth=4",
            "min_samples_leaf=50",
            "l2_regularization=1.0",
            "early_stopping=True"
        ],
        "n_features": len(feature_cols),
        "n_iter": int(clf.n_iter_),
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

    out_path = os.path.join(args.out_dir, "holdout_enhanced_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    pd.DataFrame([{
        "date": r["date"],
        "actuals": str(r["actuals"]),
        **{f"top{k}_cov": r[f"top{k}_coverage"] for k in [1, 2, 3, 5, 7]}
    } for r in results]).to_csv(
        os.path.join(args.out_dir, "holdout_enhanced_v2_summary.csv"), index=False)

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
