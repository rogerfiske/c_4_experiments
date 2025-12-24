#!/usr/bin/env python3
"""
Feature Impact Tracking System for C4 Parts Forecast.

Comprehensive analysis of how each feature affects predictions:
1. Per-Part Feature Impact - Which features help predict each part (0-9)
2. Per-Position Feature Impact - Which features help each position (QS1-QS4)
3. Feature Ablation Study - Impact of removing feature groups
4. Feature-Accuracy Correlation - How features relate to prediction success
5. Temporal Feature Importance - How importance changes over time windows

This enables targeted feature engineering for hard-to-predict parts.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

LABELS = tuple(range(10))

# Feature categories for ablation
FEATURE_CATEGORIES = {
    "lag_original": ["ca_lag1", "ca_lag2", "ca_lag7", "ca_lag14", "ca_lag28"],
    "lag_extended": ["ca_lag3", "ca_lag21", "ca_lag56"],
    "rolling": ["ca_roll_mean_7", "ca_roll_mean_14", "ca_roll_mean_28"],
    "agg_counts": [f"agg_count_{i}_t" for i in range(10)],
    "agg_props": [f"agg_prop_{i}_t" for i in range(10)],
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature impact tracking")
    p.add_argument("--daily_csv", type=str, default="data/raw/CA_4_predict_daily_aggregate.csv")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--holdout_days", type=int, default=100)
    p.add_argument("--n_windows", type=int, default=4, help="Windows for temporal analysis")
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


def get_feature_importance(clf, feature_cols: List[str]) -> List[Tuple[str, float]]:
    """Extract feature importance from model."""
    try:
        return sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
    except AttributeError:
        return [(f, 0.0) for f in feature_cols]


def evaluate_subset(X_train, y_train, X_test, y_test, seed: int) -> Dict:
    """Train and evaluate model on a feature subset."""
    if X_train.shape[1] == 0:
        return {"top5_accuracy": 0.0, "top1_accuracy": 0.0}

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6, max_iter=300,
        random_state=seed, early_stopping=False
    )
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)

    top1_hits = 0
    top5_hits = 0
    for i, actual in enumerate(y_test):
        ranked = np.argsort(-probs[i])
        if actual == ranked[0]:
            top1_hits += 1
        if actual in ranked[:5]:
            top5_hits += 1

    return {
        "top5_accuracy": top5_hits / len(y_test),
        "top1_accuracy": top1_hits / len(y_test)
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 70)
    print("FEATURE IMPACT TRACKING SYSTEM")
    print("=" * 70)

    df = load_csv(args.daily_csv)
    sup = make_features(df)
    feature_cols = [c for c in sup.columns if c not in ("date", "date_idx", "y", "position")]

    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Feature categories:")
    for cat, feats in FEATURE_CATEGORIES.items():
        present = [f for f in feats if f in feature_cols]
        print(f"  {cat}: {len(present)} features")

    unique_dates = sup["date"].unique()
    n_dates = len(unique_dates)
    holdout_start = n_dates - args.holdout_days
    holdout_dates = unique_dates[holdout_start:]
    train_dates = unique_dates[:holdout_start - 30]

    train_df = sup[sup["date"].isin(train_dates)].dropna()
    holdout_df = sup[sup["date"].isin(holdout_dates)].dropna()

    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values
    X_test = holdout_df[feature_cols].values
    y_test = holdout_df["y"].values

    print(f"\nTrain: {len(train_df)} samples")
    print(f"Test: {len(holdout_df)} samples")

    # Train baseline model
    print("\n" + "-" * 70)
    print("1. BASELINE MODEL")
    print("-" * 70)

    clf_baseline = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6, max_iter=300,
        random_state=args.seed, early_stopping=False
    )
    clf_baseline.fit(X_train, y_train)
    baseline_imp = get_feature_importance(clf_baseline, feature_cols)
    baseline_metrics = evaluate_subset(X_train, y_train, X_test, y_test, args.seed)

    print(f"Baseline Top-5 Accuracy: {baseline_metrics['top5_accuracy']:.1%}")
    print(f"Baseline Top-1 Accuracy: {baseline_metrics['top1_accuracy']:.1%}")

    # ===== SECTION 2: PER-PART FEATURE IMPACT =====
    print("\n" + "=" * 70)
    print("2. PER-PART FEATURE IMPACT")
    print("=" * 70)
    print("\nAnalyzing which features are most important for predicting each part...")

    per_part_impact = {}

    for part in LABELS:
        # Create binary classification: is this part or not
        y_train_binary = (train_df["y"] == part).astype(int).values
        y_test_binary = (holdout_df["y"] == part).astype(int).values

        # Skip if too few samples
        if y_train_binary.sum() < 50:
            print(f"Part {part}: Insufficient samples ({y_train_binary.sum()}), skipping")
            continue

        # Train binary classifier
        clf_part = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=4, max_iter=200,
            random_state=args.seed, early_stopping=False
        )
        clf_part.fit(X_train, y_train_binary)

        # Get importance for this part
        part_imp = get_feature_importance(clf_part, feature_cols)

        per_part_impact[part] = {
            "top_5_features": [(f, float(i)) for f, i in part_imp[:5]],
            "feature_category_importance": {}
        }

        # Aggregate by category
        for cat, cat_feats in FEATURE_CATEGORIES.items():
            cat_imp = sum(i for f, i in part_imp if f in cat_feats)
            per_part_impact[part]["feature_category_importance"][cat] = float(cat_imp)

    print("\nTop features by part (for binary prediction):")
    print("\n| Part | #1 Feature | #2 Feature | #3 Feature |")
    print("|------|------------|------------|------------|")
    for part in LABELS:
        if part in per_part_impact:
            top3 = per_part_impact[part]["top_5_features"][:3]
            row = f"| {part} |"
            for f, i in top3:
                row += f" {f[:12]:12s} |"
            print(row)

    # ===== SECTION 3: PER-POSITION FEATURE IMPACT =====
    print("\n" + "=" * 70)
    print("3. PER-POSITION FEATURE IMPACT")
    print("=" * 70)

    per_position_impact = {}

    for pos in range(1, 5):
        pos_train = train_df[train_df["position"] == pos]
        pos_test = holdout_df[holdout_df["position"] == pos]

        X_pos_train = pos_train[feature_cols].values
        y_pos_train = pos_train["y"].values
        X_pos_test = pos_test[feature_cols].values
        y_pos_test = pos_test["y"].values

        # Train position-specific model
        clf_pos = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=6, max_iter=300,
            random_state=args.seed, early_stopping=False
        )
        clf_pos.fit(X_pos_train, y_pos_train)

        pos_imp = get_feature_importance(clf_pos, feature_cols)
        pos_metrics = evaluate_subset(X_pos_train, y_pos_train, X_pos_test, y_pos_test, args.seed)

        per_position_impact[f"QS{pos}"] = {
            "top5_accuracy": float(pos_metrics["top5_accuracy"]),
            "top_5_features": [(f, float(i)) for f, i in pos_imp[:5]],
            "feature_category_importance": {
                cat: float(sum(i for f, i in pos_imp if f in cat_feats))
                for cat, cat_feats in FEATURE_CATEGORIES.items()
            }
        }

    print("\nPosition-specific model performance and top features:")
    for pos in range(1, 5):
        p = per_position_impact[f"QS{pos}"]
        print(f"\nQS{pos} (Top-5: {p['top5_accuracy']:.1%}):")
        for f, i in p["top_5_features"][:3]:
            print(f"  - {f}: {i:.4f}")

    # ===== SECTION 4: FEATURE ABLATION STUDY =====
    print("\n" + "=" * 70)
    print("4. FEATURE ABLATION STUDY")
    print("=" * 70)
    print("\nRemoving feature categories to measure their impact...")

    ablation_results = {"baseline": baseline_metrics}

    for cat_to_remove in FEATURE_CATEGORIES.keys():
        # Get features to keep (exclude this category)
        feats_to_remove = FEATURE_CATEGORIES[cat_to_remove]
        feats_to_keep = [f for f in feature_cols if f not in feats_to_remove]
        keep_idx = [feature_cols.index(f) for f in feats_to_keep]

        if len(keep_idx) == 0:
            continue

        X_train_ablated = X_train[:, keep_idx]
        X_test_ablated = X_test[:, keep_idx]

        metrics = evaluate_subset(X_train_ablated, y_train, X_test_ablated, y_test, args.seed)
        ablation_results[f"without_{cat_to_remove}"] = metrics

    print("\n| Configuration | Top-5 Acc | Top-1 Acc | Delta Top-5 |")
    print("|---------------|-----------|-----------|-------------|")

    baseline_t5 = ablation_results["baseline"]["top5_accuracy"]
    for config, metrics in ablation_results.items():
        delta = metrics["top5_accuracy"] - baseline_t5
        delta_str = f"{delta:+.1%}" if config != "baseline" else "-"
        print(f"| {config:20s} | {metrics['top5_accuracy']:.1%} | {metrics['top1_accuracy']:.1%} | {delta_str:11s} |")

    # Identify most impactful categories
    category_impact = {}
    for cat in FEATURE_CATEGORIES.keys():
        key = f"without_{cat}"
        if key in ablation_results:
            impact = baseline_t5 - ablation_results[key]["top5_accuracy"]
            category_impact[cat] = impact

    sorted_impact = sorted(category_impact.items(), key=lambda x: -x[1])
    print("\nFeature category impact ranking (most to least important):")
    for cat, impact in sorted_impact:
        direction = "hurts" if impact > 0 else "helps"
        print(f"  {cat}: removing {direction} by {abs(impact):.1%}")

    # ===== SECTION 5: FEATURE-ACCURACY CORRELATION =====
    print("\n" + "=" * 70)
    print("5. FEATURE-ACCURACY CORRELATION")
    print("=" * 70)
    print("\nCorrelating feature values with prediction success...")

    # Get predictions with features
    probs = clf_baseline.predict_proba(X_test)
    predictions = []
    for i in range(len(y_test)):
        ranked = np.argsort(-probs[i])
        predictions.append({
            "actual": y_test[i],
            "top5_hit": int(y_test[i] in ranked[:5]),
            "actual_rank": int(np.where(ranked == y_test[i])[0][0]) + 1,
            **{f: float(X_test[i, j]) for j, f in enumerate(feature_cols)}
        })

    pred_df = pd.DataFrame(predictions)

    # Correlate features with success
    feature_correlations = {}
    for f in feature_cols[:20]:  # Top 20 by importance
        if f in pred_df.columns:
            corr, pval = stats.pointbiserialr(pred_df["top5_hit"], pred_df[f])
            if not np.isnan(corr):
                feature_correlations[f] = {"correlation": float(corr), "pvalue": float(pval)}

    print("\nFeatures correlated with Top-5 success:")
    sorted_corr = sorted(feature_correlations.items(), key=lambda x: -abs(x[1]["correlation"]))
    print("\n| Feature | Correlation | Significant? |")
    print("|---------|-------------|--------------|")
    for f, c in sorted_corr[:10]:
        sig = "**" if c["pvalue"] < 0.05 else ""
        print(f"| {f:20s} | {c['correlation']:+.3f} | {sig:12s} |")

    # ===== SECTION 6: HARD PART FEATURE ANALYSIS =====
    print("\n" + "=" * 70)
    print("6. HARD PART FEATURE ANALYSIS (Parts 2 & 7)")
    print("=" * 70)

    hard_parts = [2, 7]
    hard_part_analysis = {}

    for part in hard_parts:
        part_mask = holdout_df["y"] == part
        part_pred_df = pred_df[pred_df["actual"] == part]

        if len(part_pred_df) < 10:
            continue

        # Features when we correctly predict this part vs miss it
        correct = part_pred_df[part_pred_df["top5_hit"] == 1]
        incorrect = part_pred_df[part_pred_df["top5_hit"] == 0]

        if len(correct) < 3 or len(incorrect) < 3:
            continue

        feature_diffs = {}
        for f in feature_cols[:15]:
            if f in correct.columns:
                correct_mean = correct[f].mean()
                incorrect_mean = incorrect[f].mean()
                diff = correct_mean - incorrect_mean
                # Normalize by std
                std = part_pred_df[f].std()
                if std > 0:
                    diff_normalized = diff / std
                    feature_diffs[f] = {
                        "correct_mean": float(correct_mean),
                        "incorrect_mean": float(incorrect_mean),
                        "diff_normalized": float(diff_normalized)
                    }

        hard_part_analysis[part] = {
            "n_samples": len(part_pred_df),
            "n_correct": len(correct),
            "n_incorrect": len(incorrect),
            "accuracy": len(correct) / len(part_pred_df),
            "distinguishing_features": sorted(
                feature_diffs.items(),
                key=lambda x: -abs(x[1]["diff_normalized"])
            )[:5]
        }

        print(f"\nPart {part} Analysis (n={len(part_pred_df)}, accuracy={len(correct)/len(part_pred_df):.1%}):")
        print("Features that distinguish correct vs incorrect predictions:")
        for f, d in hard_part_analysis[part]["distinguishing_features"]:
            direction = "higher" if d["diff_normalized"] > 0 else "lower"
            print(f"  {f}: {direction} when correct (diff={d['diff_normalized']:.2f} std)")

    # ===== SECTION 7: TEMPORAL FEATURE IMPORTANCE =====
    print("\n" + "=" * 70)
    print("7. TEMPORAL FEATURE IMPORTANCE DRIFT")
    print("=" * 70)

    window_size = args.holdout_days // args.n_windows
    temporal_importance = defaultdict(list)

    for w in range(args.n_windows):
        start_idx = holdout_start + w * window_size
        end_idx = start_idx + window_size
        window_dates = unique_dates[start_idx:end_idx]

        # Train on data before window
        train_end = start_idx - 30
        w_train_dates = unique_dates[:train_end]
        w_train_df = sup[sup["date"].isin(w_train_dates)].dropna()

        if len(w_train_df) < 1000:
            continue

        X_w_train = w_train_df[feature_cols].values
        y_w_train = w_train_df["y"].values

        clf_w = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=6, max_iter=300,
            random_state=args.seed, early_stopping=False
        )
        clf_w.fit(X_w_train, y_w_train)

        w_imp = get_feature_importance(clf_w, feature_cols)
        for rank, (f, i) in enumerate(w_imp):
            temporal_importance[f].append({"window": w + 1, "importance": i, "rank": rank + 1})

    # Check stability of top features
    print("\nTop feature importance stability across windows:")
    print("\n| Feature | W1 Rank | W2 Rank | W3 Rank | W4 Rank | Rank Std |")
    print("|---------|---------|---------|---------|---------|----------|")

    top_features = [f for f, i in baseline_imp[:10]]
    feature_rank_stability = {}

    for f in top_features:
        if f in temporal_importance:
            ranks = [t["rank"] for t in temporal_importance[f]]
            if len(ranks) == args.n_windows:
                rank_std = np.std(ranks)
                feature_rank_stability[f] = rank_std
                row = f"| {f:15s} |"
                for r in ranks:
                    row += f" {r:7d} |"
                row += f" {rank_std:8.1f} |"
                print(row)

    # ===== SUMMARY & RECOMMENDATIONS =====
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print("\n1. MOST IMPACTFUL FEATURE CATEGORIES:")
    for cat, impact in sorted_impact[:3]:
        print(f"   - {cat}: {impact:+.1%} impact when removed")

    print("\n2. FEATURES FOR HARD PARTS (2, 7):")
    for part in hard_parts:
        if part in hard_part_analysis:
            print(f"   Part {part}:")
            for f, d in hard_part_analysis[part]["distinguishing_features"][:2]:
                print(f"     - {f} ({d['diff_normalized']:+.2f} std when correct)")

    print("\n3. MOST STABLE FEATURES (for production):")
    stable_features = sorted(feature_rank_stability.items(), key=lambda x: x[1])[:5]
    for f, std in stable_features:
        print(f"   - {f} (rank std: {std:.1f})")

    print("\n4. FEATURE ENGINEERING RECOMMENDATIONS:")
    # Based on ablation
    if category_impact.get("lag_extended", 0) > 0.01:
        print("   - KEEP extended lags (3, 21, 56) - they help")
    if category_impact.get("rolling", 0) > 0.01:
        print("   - KEEP rolling means - they help")
    if category_impact.get("agg_props", 0) < -0.01:
        print("   - Consider REMOVING agg_props - they may hurt")

    # Save results
    output = {
        "baseline_metrics": baseline_metrics,
        "per_part_impact": {str(k): v for k, v in per_part_impact.items()},
        "per_position_impact": per_position_impact,
        "ablation_results": ablation_results,
        "category_impact_ranking": sorted_impact,
        "feature_correlations": feature_correlations,
        "hard_part_analysis": {str(k): {
            "n_samples": v["n_samples"],
            "accuracy": v["accuracy"],
            "distinguishing_features": [(f, d) for f, d in v["distinguishing_features"]]
        } for k, v in hard_part_analysis.items()},
        "feature_rank_stability": feature_rank_stability
    }

    out_path = os.path.join(args.out_dir, "feature_impact_tracking.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
