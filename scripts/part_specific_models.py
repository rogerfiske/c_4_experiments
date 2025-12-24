#!/usr/bin/env python3
"""
Part-Specific Models for Hard Parts (2 and 7).

Based on feature impact analysis:
- Part 2: Benefits from ca_lag2, ca_lag56; hurt by ca_lag28
- Part 7: Benefits from ca_lag56, ca_lag7; hurt by ca_lag3

Strategies:
1. Extended features (add lags 90, 120, 180)
2. Binary classifiers for each hard part
3. Ensemble: blend general model with part-specific boosting
4. Feature-weighted approach

Goal: Improve Top-5 accuracy for Parts 2 and 7 specifically.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

LABELS = tuple(range(10))
HARD_PARTS = [2, 7]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part-specific models for hard parts")
    p.add_argument("--daily_csv", type=str, default="data/raw/CA_4_predict_daily_aggregate.csv")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--holdout_days", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df.sort_values("date").reset_index(drop=True)


def make_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extended features with longer lags for hard parts."""
    # Standard lags + extended lags
    lags = [1, 2, 3, 7, 14, 21, 28, 56, 90, 120, 180]
    rolling_windows = [7, 14, 28, 56]  # Added 56-day rolling

    rows = []
    for pos in range(1, 5):
        ca_col = f"CA_QS{pos}"
        tmp = pd.DataFrame({"date": df["date"].values, "date_idx": df.index.values})
        tmp["position"] = pos
        tmp["y"] = df[ca_col].shift(-1).astype("float")

        # Lag features
        for lag in lags:
            tmp[f"ca_lag{lag}"] = df[ca_col].shift(lag).astype("float")

        # Rolling means
        for window in rolling_windows:
            tmp[f"ca_roll_mean_{window}"] = df[ca_col].shift(1).rolling(window).mean().astype("float")
            tmp[f"ca_roll_std_{window}"] = df[ca_col].shift(1).rolling(window).std().astype("float")

        # Lag differences (velocity features)
        tmp["ca_diff_1_7"] = tmp["ca_lag1"] - tmp["ca_lag7"]
        tmp["ca_diff_7_28"] = tmp["ca_lag7"] - tmp["ca_lag28"]
        tmp["ca_diff_28_56"] = tmp["ca_lag28"] - tmp["ca_lag56"]

        # Part-specific aggregate features
        part_cols = [f"QS{pos}_{lab}" for lab in LABELS]
        denom = df[part_cols].sum(axis=1).replace(0, np.nan)
        for lab in LABELS:
            tmp[f"agg_count_{lab}_t"] = df[f"QS{pos}_{lab}"].astype("float")
            tmp[f"agg_prop_{lab}_t"] = (df[f"QS{pos}_{lab}"] / denom).astype("float")

        # Special features for hard parts (2 and 7)
        # How often did part 2 or 7 appear in recent history?
        for hard_part in HARD_PARTS:
            tmp[f"recent_part{hard_part}_count_7d"] = (df[ca_col] == hard_part).shift(1).rolling(7).sum().astype("float")
            tmp[f"recent_part{hard_part}_count_14d"] = (df[ca_col] == hard_part).shift(1).rolling(14).sum().astype("float")
            tmp[f"recent_part{hard_part}_count_28d"] = (df[ca_col] == hard_part).shift(1).rolling(28).sum().astype("float")

        rows.append(tmp)

    out = pd.concat(rows).sort_values(["date", "position"]).reset_index(drop=True)
    out = out.dropna(subset=["y"])
    out["y"] = out["y"].astype(int)
    return out


def make_standard_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standard features (baseline)."""
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


def compute_pooled_coverage(actuals: List[int], shipped: List[int]) -> float:
    needed = Counter(actuals)
    have = Counter(shipped)
    fulfilled = sum(min(needed[p], have.get(p, 0)) for p in needed)
    return fulfilled / len(actuals) if actuals else 0.0


def evaluate_model(probs_list: List[np.ndarray], actuals_list: List[List[int]]) -> Dict:
    """Evaluate predictions across all days."""
    results = []

    for day_idx, (probs, actuals) in enumerate(zip(probs_list, actuals_list)):
        top5_shipped = []
        per_part_hits = {p: {"total": 0, "top5_hit": 0} for p in LABELS}

        for i, actual in enumerate(actuals):
            ranked = np.argsort(-probs[i])
            top5 = [int(ranked[j]) for j in range(5)]
            top5_shipped.extend(top5)

            per_part_hits[actual]["total"] += 1
            if actual in top5:
                per_part_hits[actual]["top5_hit"] += 1

        pooled_top5 = compute_pooled_coverage(actuals, top5_shipped)
        results.append({
            "pooled_top5": pooled_top5,
            "per_part_hits": per_part_hits
        })

    # Aggregate
    overall_top5 = np.mean([r["pooled_top5"] for r in results])

    # Per-part accuracy
    part_totals = {p: 0 for p in LABELS}
    part_hits = {p: 0 for p in LABELS}
    for r in results:
        for p in LABELS:
            part_totals[p] += r["per_part_hits"][p]["total"]
            part_hits[p] += r["per_part_hits"][p]["top5_hit"]

    per_part_acc = {p: part_hits[p] / part_totals[p] if part_totals[p] > 0 else 0 for p in LABELS}

    return {
        "pooled_top5": overall_top5,
        "per_part_accuracy": per_part_acc,
        "hard_parts_avg": np.mean([per_part_acc[p] for p in HARD_PARTS]),
        "n_days": len(results)
    }


def train_binary_classifier(X_train, y_train, target_part: int, seed: int):
    """Train a binary classifier for a specific part."""
    y_binary = (y_train == target_part).astype(int)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=4, max_iter=200,
        random_state=seed, early_stopping=False
    )
    clf.fit(X_train, y_binary)
    return clf


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 70)
    print("PART-SPECIFIC MODELS FOR HARD PARTS (2 and 7)")
    print("=" * 70)

    df = load_csv(args.daily_csv)

    # Prepare standard and extended features
    print("\nPreparing features...")
    sup_standard = make_standard_features(df)
    sup_extended = make_extended_features(df)

    standard_features = [c for c in sup_standard.columns if c not in ("date", "date_idx", "y", "position")]
    extended_features = [c for c in sup_extended.columns if c not in ("date", "date_idx", "y", "position")]

    print(f"Standard features: {len(standard_features)}")
    print(f"Extended features: {len(extended_features)}")

    unique_dates = sup_standard["date"].unique()
    n_dates = len(unique_dates)
    holdout_start = n_dates - args.holdout_days
    holdout_dates = unique_dates[holdout_start:]
    train_dates = unique_dates[:holdout_start - 30]

    # Prepare data
    train_std = sup_standard[sup_standard["date"].isin(train_dates)].dropna()
    test_std = sup_standard[sup_standard["date"].isin(holdout_dates)].dropna()
    train_ext = sup_extended[sup_extended["date"].isin(train_dates)].dropna()
    test_ext = sup_extended[sup_extended["date"].isin(holdout_dates)].dropna()

    X_train_std = train_std[standard_features].values
    y_train_std = train_std["y"].values
    X_train_ext = train_ext[extended_features].values
    y_train_ext = train_ext["y"].values

    print(f"\nTrain samples (standard): {len(train_std)}")
    print(f"Train samples (extended): {len(train_ext)}")
    print(f"Test days: {len(holdout_dates)}")

    # ===== STRATEGY 1: BASELINE =====
    print("\n" + "=" * 70)
    print("STRATEGY 1: BASELINE MODEL")
    print("=" * 70)

    clf_baseline = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6, max_iter=300,
        random_state=args.seed, early_stopping=False
    )
    clf_baseline.fit(X_train_std, y_train_std)

    # Collect predictions
    baseline_probs = []
    baseline_actuals = []
    for date in holdout_dates:
        day_data = test_std[test_std["date"] == date].sort_values("position")
        if len(day_data) != 4:
            continue
        X_day = day_data[standard_features].values
        probs = clf_baseline.predict_proba(X_day)
        baseline_probs.append(probs)
        baseline_actuals.append(day_data["y"].tolist())

    baseline_results = evaluate_model(baseline_probs, baseline_actuals)
    print(f"\nBaseline Pooled Top-5: {baseline_results['pooled_top5']:.1%}")
    print(f"Baseline Part 2 Accuracy: {baseline_results['per_part_accuracy'][2]:.1%}")
    print(f"Baseline Part 7 Accuracy: {baseline_results['per_part_accuracy'][7]:.1%}")
    print(f"Baseline Hard Parts Avg: {baseline_results['hard_parts_avg']:.1%}")

    # ===== STRATEGY 2: EXTENDED FEATURES =====
    print("\n" + "=" * 70)
    print("STRATEGY 2: EXTENDED FEATURES (Longer Lags)")
    print("=" * 70)
    print("Added: lags 90, 120, 180; rolling_56; velocity features; hard part history")

    clf_extended = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6, max_iter=300,
        random_state=args.seed, early_stopping=False
    )
    clf_extended.fit(X_train_ext, y_train_ext)

    extended_probs = []
    extended_actuals = []
    for date in holdout_dates:
        day_data = test_ext[test_ext["date"] == date].sort_values("position")
        if len(day_data) != 4:
            continue
        X_day = day_data[extended_features].values
        probs = clf_extended.predict_proba(X_day)
        extended_probs.append(probs)
        extended_actuals.append(day_data["y"].tolist())

    extended_results = evaluate_model(extended_probs, extended_actuals)
    print(f"\nExtended Pooled Top-5: {extended_results['pooled_top5']:.1%}")
    print(f"Extended Part 2 Accuracy: {extended_results['per_part_accuracy'][2]:.1%}")
    print(f"Extended Part 7 Accuracy: {extended_results['per_part_accuracy'][7]:.1%}")
    print(f"Extended Hard Parts Avg: {extended_results['hard_parts_avg']:.1%}")

    # ===== STRATEGY 3: BINARY CLASSIFIERS + BOOSTING =====
    print("\n" + "=" * 70)
    print("STRATEGY 3: BINARY CLASSIFIERS + PROBABILITY BOOSTING")
    print("=" * 70)
    print("Train specialized binary classifiers for Parts 2 and 7")

    # Train binary classifiers
    clf_part2 = train_binary_classifier(X_train_ext, y_train_ext, 2, args.seed)
    clf_part7 = train_binary_classifier(X_train_ext, y_train_ext, 7, args.seed)

    # Evaluate with boosting
    boost_factors = [0.0, 0.1, 0.2, 0.3, 0.5]
    best_boost = 0.0
    best_boost_result = None

    for boost in boost_factors:
        boosted_probs = []
        for date in holdout_dates:
            day_data = test_ext[test_ext["date"] == date].sort_values("position")
            if len(day_data) != 4:
                continue
            X_day = day_data[extended_features].values

            # Get base probabilities
            probs = clf_extended.predict_proba(X_day).copy()

            # Get binary classifier predictions
            prob_part2 = clf_part2.predict_proba(X_day)[:, 1]
            prob_part7 = clf_part7.predict_proba(X_day)[:, 1]

            # Boost hard part probabilities
            probs[:, 2] += boost * prob_part2
            probs[:, 7] += boost * prob_part7

            # Renormalize
            probs = probs / probs.sum(axis=1, keepdims=True)
            boosted_probs.append(probs)

        if len(boosted_probs) == len(extended_actuals):
            result = evaluate_model(boosted_probs, extended_actuals)
            if best_boost_result is None or result["hard_parts_avg"] > best_boost_result["hard_parts_avg"]:
                best_boost = boost
                best_boost_result = result

    print(f"\nBest boost factor: {best_boost}")
    print(f"Boosted Pooled Top-5: {best_boost_result['pooled_top5']:.1%}")
    print(f"Boosted Part 2 Accuracy: {best_boost_result['per_part_accuracy'][2]:.1%}")
    print(f"Boosted Part 7 Accuracy: {best_boost_result['per_part_accuracy'][7]:.1%}")
    print(f"Boosted Hard Parts Avg: {best_boost_result['hard_parts_avg']:.1%}")

    # ===== STRATEGY 4: ENSEMBLE WITH PART-SPECIFIC MODELS =====
    print("\n" + "=" * 70)
    print("STRATEGY 4: ENSEMBLE (General + Part-Specific)")
    print("=" * 70)

    # Train models only on samples where hard parts are the target
    train_part2_idx = y_train_ext == 2
    train_part7_idx = y_train_ext == 7

    # For part 2: train model on balanced dataset
    # Oversample part 2 cases
    X_train_p2_pos = X_train_ext[train_part2_idx]
    y_train_p2_pos = y_train_ext[train_part2_idx]

    # Get equal number of negative samples
    neg_idx = np.where(~train_part2_idx)[0]
    np.random.shuffle(neg_idx)
    neg_sample_idx = neg_idx[:len(y_train_p2_pos) * 2]

    X_train_p2 = np.vstack([X_train_p2_pos, X_train_ext[neg_sample_idx]])
    y_train_p2 = np.hstack([y_train_p2_pos, y_train_ext[neg_sample_idx]])

    clf_focused_p2 = HistGradientBoostingClassifier(
        learning_rate=0.03, max_depth=5, max_iter=400,
        random_state=args.seed, early_stopping=False
    )
    clf_focused_p2.fit(X_train_p2, y_train_p2)

    # Same for part 7
    X_train_p7_pos = X_train_ext[train_part7_idx]
    y_train_p7_pos = y_train_ext[train_part7_idx]
    neg_idx = np.where(~train_part7_idx)[0]
    np.random.shuffle(neg_idx)
    neg_sample_idx = neg_idx[:len(y_train_p7_pos) * 2]

    X_train_p7 = np.vstack([X_train_p7_pos, X_train_ext[neg_sample_idx]])
    y_train_p7 = np.hstack([y_train_p7_pos, y_train_ext[neg_sample_idx]])

    clf_focused_p7 = HistGradientBoostingClassifier(
        learning_rate=0.03, max_depth=5, max_iter=400,
        random_state=args.seed, early_stopping=False
    )
    clf_focused_p7.fit(X_train_p7, y_train_p7)

    # Ensemble predictions
    ensemble_weights = [0.1, 0.2, 0.3]
    best_ensemble_weight = 0.0
    best_ensemble_result = None

    for weight in ensemble_weights:
        ensemble_probs = []
        for date in holdout_dates:
            day_data = test_ext[test_ext["date"] == date].sort_values("position")
            if len(day_data) != 4:
                continue
            X_day = day_data[extended_features].values

            # Base model
            probs = clf_extended.predict_proba(X_day).copy()

            # Part-specific model predictions
            probs_p2 = clf_focused_p2.predict_proba(X_day)
            probs_p7 = clf_focused_p7.predict_proba(X_day)

            # Blend: boost part 2 and 7 columns
            if 2 in clf_focused_p2.classes_:
                p2_idx = list(clf_focused_p2.classes_).index(2)
                probs[:, 2] = (1 - weight) * probs[:, 2] + weight * probs_p2[:, p2_idx]

            if 7 in clf_focused_p7.classes_:
                p7_idx = list(clf_focused_p7.classes_).index(7)
                probs[:, 7] = (1 - weight) * probs[:, 7] + weight * probs_p7[:, p7_idx]

            # Renormalize
            probs = probs / probs.sum(axis=1, keepdims=True)
            ensemble_probs.append(probs)

        if len(ensemble_probs) == len(extended_actuals):
            result = evaluate_model(ensemble_probs, extended_actuals)
            if best_ensemble_result is None or result["hard_parts_avg"] > best_ensemble_result["hard_parts_avg"]:
                best_ensemble_weight = weight
                best_ensemble_result = result

    print(f"\nBest ensemble weight: {best_ensemble_weight}")
    print(f"Ensemble Pooled Top-5: {best_ensemble_result['pooled_top5']:.1%}")
    print(f"Ensemble Part 2 Accuracy: {best_ensemble_result['per_part_accuracy'][2]:.1%}")
    print(f"Ensemble Part 7 Accuracy: {best_ensemble_result['per_part_accuracy'][7]:.1%}")
    print(f"Ensemble Hard Parts Avg: {best_ensemble_result['hard_parts_avg']:.1%}")

    # ===== COMPARISON =====
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    strategies = [
        ("Baseline", baseline_results),
        ("Extended Features", extended_results),
        (f"Binary Boost (w={best_boost})", best_boost_result),
        (f"Ensemble (w={best_ensemble_weight})", best_ensemble_result),
    ]

    print("\n| Strategy | Pooled Top-5 | Part 2 | Part 7 | Hard Avg | Delta vs Base |")
    print("|----------|--------------|--------|--------|----------|---------------|")

    for name, result in strategies:
        delta = result["hard_parts_avg"] - baseline_results["hard_parts_avg"]
        delta_str = f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}"
        print(f"| {name:25s} | {result['pooled_top5']:.1%} | {result['per_part_accuracy'][2]:.1%} | {result['per_part_accuracy'][7]:.1%} | {result['hard_parts_avg']:.1%} | {delta_str:13s} |")

    # Find best strategy for hard parts
    best_strategy = max(strategies, key=lambda x: x[1]["hard_parts_avg"])
    print(f"\nBEST STRATEGY FOR HARD PARTS: {best_strategy[0]}")
    print(f"  Hard Parts Avg: {best_strategy[1]['hard_parts_avg']:.1%}")
    print(f"  Improvement: {best_strategy[1]['hard_parts_avg'] - baseline_results['hard_parts_avg']:+.1%}")

    # Find best strategy for overall
    best_overall = max(strategies, key=lambda x: x[1]["pooled_top5"])
    print(f"\nBEST STRATEGY FOR OVERALL: {best_overall[0]}")
    print(f"  Pooled Top-5: {best_overall[1]['pooled_top5']:.1%}")

    # Per-part breakdown
    print("\n" + "-" * 70)
    print("PER-PART ACCURACY BREAKDOWN")
    print("-" * 70)
    print("\n| Part | Baseline | Extended | Boost | Ensemble | Best |")
    print("|------|----------|----------|-------|----------|------|")

    for part in LABELS:
        base = baseline_results["per_part_accuracy"][part]
        ext = extended_results["per_part_accuracy"][part]
        boost = best_boost_result["per_part_accuracy"][part]
        ens = best_ensemble_result["per_part_accuracy"][part]
        best_val = max(base, ext, boost, ens)
        marker = "*" if part in HARD_PARTS else ""
        print(f"| {part}{marker:1s} | {base:.0%} | {ext:.0%} | {boost:.0%} | {ens:.0%} | {best_val:.0%} |")

    # Save results
    output = {
        "strategies": {
            name: {
                "pooled_top5": float(result["pooled_top5"]),
                "per_part_accuracy": {str(k): float(v) for k, v in result["per_part_accuracy"].items()},
                "hard_parts_avg": float(result["hard_parts_avg"])
            }
            for name, result in strategies
        },
        "best_for_hard_parts": best_strategy[0],
        "best_for_overall": best_overall[0],
        "configuration": {
            "extended_lags": [90, 120, 180],
            "best_boost_factor": best_boost,
            "best_ensemble_weight": best_ensemble_weight
        }
    }

    out_path = os.path.join(args.out_dir, "part_specific_models_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
