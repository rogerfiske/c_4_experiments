#!/usr/bin/env python3
"""
Priority 2: Model Tuning - Push for higher coverage.

Tests various hyperparameter configurations for LightGBM and HistGB.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')

LABELS = tuple(range(10))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model tuning")
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


def compute_pooled_coverage(actuals: List[int], shipped: List[int]) -> float:
    needed = Counter(actuals)
    have = Counter(shipped)
    fulfilled = sum(min(needed[p], have.get(p, 0)) for p in needed)
    return fulfilled / len(actuals) if actuals else 0.0


def evaluate_model(clf, holdout_df, feature_cols, holdout_dates) -> Dict:
    results = []
    for date in holdout_dates:
        day_data = holdout_df[holdout_df["date"] == date].sort_values("position")
        if len(day_data) != 4:
            continue

        actuals = day_data["y"].tolist()
        X_day = day_data[feature_cols].values
        probs = clf.predict_proba(X_day)

        top1s, top3s, top5s, top7s = [], [], [], []
        for i in range(4):
            ranked = np.argsort(-probs[i])
            top1s.append(int(ranked[0]))
            top3s.extend([int(ranked[j]) for j in range(3)])
            top5s.extend([int(ranked[j]) for j in range(5)])
            top7s.extend([int(ranked[j]) for j in range(7)])

        results.append({
            "top1": compute_pooled_coverage(actuals, top1s),
            "top3": compute_pooled_coverage(actuals, top3s),
            "top5": compute_pooled_coverage(actuals, top5s),
            "top7": compute_pooled_coverage(actuals, top7s),
        })

    n = len(results)
    return {
        "top1": np.mean([r["top1"] for r in results]),
        "top3": np.mean([r["top3"] for r in results]),
        "top5": np.mean([r["top5"] for r in results]),
        "top7": np.mean([r["top7"] for r in results]),
        "top5_perfect": sum(1 for r in results if r["top5"] == 1.0) / n,
        "top7_perfect": sum(1 for r in results if r["top7"] == 1.0) / n,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 70)
    print("MODEL TUNING: Push for Higher Coverage")
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

    print(f"\nTrain: {len(train_df)} samples, Holdout: {len(holdout_dates)} days")

    results = {}

    # Baseline config
    configs = [
        ("HistGB-default", HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=6, max_iter=300,
            random_state=args.seed, early_stopping=False
        )),
        ("HistGB-deep", HistGradientBoostingClassifier(
            learning_rate=0.03, max_depth=8, max_iter=500,
            random_state=args.seed, early_stopping=False
        )),
        ("HistGB-shallow", HistGradientBoostingClassifier(
            learning_rate=0.1, max_depth=4, max_iter=200,
            random_state=args.seed, early_stopping=False
        )),
        ("LGB-default", lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=args.seed, verbose=-1
        )),
        ("LGB-more-trees", lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            random_state=args.seed, verbose=-1
        )),
        ("LGB-deep", lgb.LGBMClassifier(
            n_estimators=400, max_depth=10, learning_rate=0.05,
            num_leaves=63, random_state=args.seed, verbose=-1
        )),
        ("LGB-regularized", lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=args.seed, verbose=-1
        )),
    ]

    print("\n| Config | Top-1 | Top-3 | Top-5 | Top-7 | T5-Perf | T7-Perf |")
    print("|--------|-------|-------|-------|-------|---------|---------|")

    for name, clf in configs:
        clf.fit(X_train, y_train)
        r = evaluate_model(clf, holdout_df, feature_cols, holdout_dates)
        results[name] = r
        print(f"| {name:14s} | {r['top1']:.1%} | {r['top3']:.1%} | {r['top5']:.1%} | {r['top7']:.1%} | {r['top5_perfect']:.0%} | {r['top7_perfect']:.0%} |")

    # Find best
    best_name = max(results.keys(), key=lambda k: (results[k]["top5"], results[k]["top7"]))
    best = results[best_name]

    print(f"\n" + "-" * 70)
    print(f"BEST CONFIG: {best_name}")
    print(f"  Top-5: {best['top5']:.1%}, Top-7: {best['top7']:.1%}")
    print(f"  Top-5 Perfect: {best['top5_perfect']:.0%}, Top-7 Perfect: {best['top7_perfect']:.0%}")

    # Compare to baseline
    baseline = {"top5": 0.912, "top7": 0.972}
    print(f"\nIMPROVEMENT vs Original Baseline:")
    print(f"  Top-5: {best['top5']:.1%} vs 91.2% ({best['top5'] - 0.912:+.1%})")
    print(f"  Top-7: {best['top7']:.1%} vs 97.2% ({best['top7'] - 0.972:+.1%})")

    # Save results
    output = {
        "experiment": "model_tuning",
        "configs": {name: {k: float(v) for k, v in r.items()} for name, r in results.items()},
        "best_config": best_name
    }

    out_path = os.path.join(args.out_dir, "model_tuning_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
