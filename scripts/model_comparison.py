#!/usr/bin/env python3
"""
Priority 2: Model Comparison for C4 Parts Forecast.

Compares:
1. HistGradientBoosting (baseline)
2. XGBoost
3. LightGBM
4. Per-position models (4 separate)
5. Calibrated models (temperature scaling)

Uses winning feature set: Lags + Rolling Means.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# Optional imports
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings('ignore')

LABELS = tuple(range(10))
N_CLASSES = 10


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model comparison for C4")
    p.add_argument("--daily_csv", type=str, default="data/raw/CA_4_predict_daily_aggregate.csv")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--holdout_days", type=int, default=100)
    p.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3, 7, 14, 21, 28, 56])
    p.add_argument("--rolling_windows", type=int, nargs="+", default=[7, 14, 28])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df.sort_values("date").reset_index(drop=True)


def make_features(df: pd.DataFrame, lags: Sequence[int], rolling_windows: Sequence[int]) -> pd.DataFrame:
    """Winning feature set: lags + rolling means."""
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


def evaluate_model(clf, holdout_df: pd.DataFrame, feature_cols: List[str],
                   holdout_dates, per_position: bool = False) -> Dict:
    """Evaluate a model on holdout set."""
    results = []

    for date in holdout_dates:
        day_data = holdout_df[holdout_df["date"] == date].sort_values("position")
        if len(day_data) != 4:
            continue

        actuals = day_data["y"].tolist()

        if per_position:
            # clf is a dict of 4 models
            probs_list = []
            for i, pos in enumerate(range(1, 5)):
                pos_data = day_data[day_data["position"] == pos]
                X_pos = pos_data[feature_cols].values
                probs_list.append(clf[pos].predict_proba(X_pos)[0])
            probs = np.array(probs_list)
        else:
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
            "top1_cov": compute_pooled_coverage(actuals, top1s),
            "top3_cov": compute_pooled_coverage(actuals, top3s),
            "top5_cov": compute_pooled_coverage(actuals, top5s),
            "top7_cov": compute_pooled_coverage(actuals, top7s),
        })

    n = len(results)
    return {
        "n_days": n,
        "top1": np.mean([r["top1_cov"] for r in results]),
        "top3": np.mean([r["top3_cov"] for r in results]),
        "top5": np.mean([r["top5_cov"] for r in results]),
        "top7": np.mean([r["top7_cov"] for r in results]),
        "top5_perfect": sum(1 for r in results if r["top5_cov"] == 1.0) / n,
        "top7_perfect": sum(1 for r in results if r["top7_cov"] == 1.0) / n,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 70)
    print("PRIORITY 2: MODEL COMPARISON")
    print("=" * 70)
    print(f"\nModels to test:")
    print(f"  1. HistGradientBoosting (current best)")
    if HAS_XGB:
        print(f"  2. XGBoost")
    if HAS_LGB:
        print(f"  3. LightGBM")
    print(f"  4. Per-position models (4 separate HistGB)")
    print(f"  5. Calibrated HistGB (isotonic)")

    df = load_csv(args.daily_csv)
    sup = make_features(df, args.lags, args.rolling_windows)
    feature_cols = [c for c in sup.columns if c not in ("date", "date_idx", "y", "position")]

    print(f"\nData: {len(df)} days, {len(feature_cols)} features")

    unique_dates = sup["date"].unique()
    n_dates = len(unique_dates)
    holdout_start = n_dates - args.holdout_days

    holdout_dates = unique_dates[holdout_start:]
    val_start = holdout_start - 60
    train_dates = unique_dates[:val_start]
    val_dates = unique_dates[val_start:holdout_start - 30]

    train_df = sup[sup["date"].isin(train_dates)].dropna()
    val_df = sup[sup["date"].isin(val_dates)].dropna()
    holdout_df = sup[sup["date"].isin(holdout_dates)].dropna()

    print(f"Train: {len(train_dates)} days")
    print(f"Val: {len(val_dates)} days (for calibration)")
    print(f"Holdout: {len(holdout_dates)} days")

    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["y"].values

    results = {}

    # 1. HistGradientBoosting (baseline)
    print("\n" + "-" * 70)
    print("Training: HistGradientBoosting...")
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6, max_iter=300,
        random_state=args.seed, early_stopping=False
    )
    hgb.fit(X_train, y_train)
    results["HistGB"] = evaluate_model(hgb, holdout_df, feature_cols, holdout_dates)
    print(f"  Top-5: {results['HistGB']['top5']:.1%}, Top-7: {results['HistGB']['top7']:.1%}")

    # 2. XGBoost
    if HAS_XGB:
        print("\nTraining: XGBoost...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            objective='multi:softprob',
            num_class=N_CLASSES,
            random_state=args.seed,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0
        )
        xgb_clf.fit(X_train, y_train)
        results["XGBoost"] = evaluate_model(xgb_clf, holdout_df, feature_cols, holdout_dates)
        print(f"  Top-5: {results['XGBoost']['top5']:.1%}, Top-7: {results['XGBoost']['top7']:.1%}")

    # 3. LightGBM
    if HAS_LGB:
        print("\nTraining: LightGBM...")
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            objective='multiclass',
            num_class=N_CLASSES,
            random_state=args.seed,
            verbose=-1
        )
        lgb_clf.fit(X_train, y_train)
        results["LightGBM"] = evaluate_model(lgb_clf, holdout_df, feature_cols, holdout_dates)
        print(f"  Top-5: {results['LightGBM']['top5']:.1%}, Top-7: {results['LightGBM']['top7']:.1%}")

    # 4. Per-position models
    print("\nTraining: Per-position models (4 x HistGB)...")
    pos_models = {}
    for pos in range(1, 5):
        pos_train = train_df[train_df["position"] == pos]
        X_pos = pos_train[feature_cols].values
        y_pos = pos_train["y"].values

        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=6, max_iter=300,
            random_state=args.seed, early_stopping=False
        )
        clf.fit(X_pos, y_pos)
        pos_models[pos] = clf

    results["Per-Position"] = evaluate_model(pos_models, holdout_df, feature_cols, holdout_dates, per_position=True)
    print(f"  Top-5: {results['Per-Position']['top5']:.1%}, Top-7: {results['Per-Position']['top7']:.1%}")

    # 5. Calibrated model
    print("\nTraining: Calibrated HistGB (isotonic)...")
    # Combine train + val for calibration
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train, y_val])

    cal_clf = CalibratedClassifierCV(
        estimator=HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=6, max_iter=300,
            random_state=args.seed, early_stopping=False
        ),
        method='isotonic',
        cv=3
    )
    cal_clf.fit(X_train_val, y_train_val)
    results["Calibrated"] = evaluate_model(cal_clf, holdout_df, feature_cols, holdout_dates)
    print(f"  Top-5: {results['Calibrated']['top5']:.1%}, Top-7: {results['Calibrated']['top7']:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    print("\n| Model | Top-1 | Top-3 | Top-5 | Top-7 | Top-5 Perfect | Top-7 Perfect |")
    print("|-------|-------|-------|-------|-------|---------------|---------------|")

    baseline = {"top1": 0.282, "top3": 0.695, "top5": 0.912, "top7": 0.972}

    for name, r in results.items():
        t5_delta = r["top5"] - baseline["top5"]
        t5_str = f"+{t5_delta:.1%}" if t5_delta >= 0 else f"{t5_delta:.1%}"
        print(f"| {name:12s} | {r['top1']:.1%} | {r['top3']:.1%} | {r['top5']:.1%} | {r['top7']:.1%} | {r['top5_perfect']:.0%} | {r['top7_perfect']:.0%} |")

    # Find best model
    best_name = max(results.keys(), key=lambda k: results[k]["top5"])
    best = results[best_name]

    print(f"\n" + "-" * 70)
    print(f"BEST MODEL: {best_name}")
    print(f"  Top-5 Coverage: {best['top5']:.1%} (baseline: 91.2%)")
    print(f"  Top-7 Coverage: {best['top7']:.1%} (baseline: 97.2%)")
    print(f"  Top-5 Perfect Days: {best['top5_perfect']:.0%}")
    print(f"  Top-7 Perfect Days: {best['top7_perfect']:.0%}")

    # Save results
    output = {
        "experiment": "model_comparison_priority2",
        "feature_set": "lags_rolling_means",
        "n_features": len(feature_cols),
        "models": {name: {k: float(v) if isinstance(v, (np.floating, float)) else v
                         for k, v in r.items()}
                   for name, r in results.items()},
        "best_model": best_name,
        "baseline": baseline
    }

    out_path = os.path.join(args.out_dir, "model_comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
