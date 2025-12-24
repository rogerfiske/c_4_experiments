#!/usr/bin/env python3
"""
Metrics Collection Script for C4 Parts Forecast.

Collects and logs metrics for model monitoring:
- Accuracy metrics (per-part, per-position, overall)
- Stability metrics (CV, rank correlation)
- Feature health (importance tracking)
- Alerts based on thresholds

Usage:
  python scripts/collect_metrics.py --window_days 30
  python scripts/collect_metrics.py --window_days 30 --save_predictions
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

LABELS = tuple(range(10))

# Alert thresholds
THRESHOLDS = {
    "accuracy_critical": 0.85,      # Top-5 < 85% = critical
    "accuracy_warning": 0.88,       # Top-5 < 88% = warning
    "part_degradation": 0.15,       # Part accuracy drop > 15%
    "feature_drift": 0.70,          # Feature importance correlation < 0.7
    "rank_correlation_warning": 0.30,
    "cv_warning": 0.25,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect metrics for C4 forecast")
    p.add_argument("--daily_csv", type=str, default="data/raw/CA_4_predict_daily_aggregate.csv")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--window_days", type=int, default=30, help="Rolling window for metrics")
    p.add_argument("--save_predictions", action="store_true", help="Save daily predictions log")
    p.add_argument("--baseline_file", type=str, default=None, help="Previous metrics for comparison")
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


def collect_predictions(clf, holdout_df, feature_cols, holdout_dates) -> pd.DataFrame:
    """Collect all predictions for the holdout period."""
    predictions = []

    for date in holdout_dates:
        day_data = holdout_df[holdout_df["date"] == date].sort_values("position")
        if len(day_data) != 4:
            continue

        X_day = day_data[feature_cols].values
        probs = clf.predict_proba(X_day)

        day_preds = []
        for i, pos in enumerate(range(1, 5)):
            actual = int(day_data.iloc[i]["y"])
            ranked = np.argsort(-probs[i])

            day_preds.append({
                "date": str(date.date()) if hasattr(date, 'date') else str(date)[:10],
                "position": pos,
                "actual": actual,
                "pred_top1": int(ranked[0]),
                "top3": [int(ranked[j]) for j in range(3)],
                "top5": [int(ranked[j]) for j in range(5)],
                "actual_rank": int(np.where(ranked == actual)[0][0]) + 1,
                "top1_confidence": float(probs[i, ranked[0]]),
                "actual_prob": float(probs[i, actual]),
                "top1_hit": actual == ranked[0],
                "top3_hit": actual in ranked[:3],
                "top5_hit": actual in ranked[:5],
            })

        # Add pooled coverage
        actuals = [p["actual"] for p in day_preds]
        top5_shipped = []
        for p in day_preds:
            top5_shipped.extend(p["top5"])

        pooled_top5 = compute_pooled_coverage(actuals, top5_shipped)

        for p in day_preds:
            p["day_pooled_top5"] = pooled_top5

        predictions.extend(day_preds)

    return pd.DataFrame(predictions)


def compute_metrics(pred_df: pd.DataFrame, feature_importance: List) -> Dict:
    """Compute all metrics from predictions."""
    metrics = {
        "collection_timestamp": datetime.now().isoformat(),
        "n_predictions": len(pred_df),
        "n_days": pred_df["date"].nunique(),
        "date_range": {
            "start": pred_df["date"].min(),
            "end": pred_df["date"].max()
        }
    }

    # 1. Overall Accuracy
    unique_days = pred_df.groupby("date").first()
    metrics["accuracy"] = {
        "pooled_top5_mean": float(unique_days["day_pooled_top5"].mean()),
        "pooled_top5_std": float(unique_days["day_pooled_top5"].std()),
        "pooled_top5_cv": float(unique_days["day_pooled_top5"].std() / unique_days["day_pooled_top5"].mean())
            if unique_days["day_pooled_top5"].mean() > 0 else 0,
        "perfect_day_rate_top5": float((unique_days["day_pooled_top5"] == 1.0).mean()),
        "per_position_top5_mean": float(pred_df["top5_hit"].mean()),
        "per_position_top1_mean": float(pred_df["top1_hit"].mean()),
    }

    # 2. Per-Part Accuracy
    part_metrics = {}
    for part in LABELS:
        part_preds = pred_df[pred_df["actual"] == part]
        if len(part_preds) >= 5:
            part_metrics[str(part)] = {
                "n": int(len(part_preds)),
                "top5_accuracy": float(part_preds["top5_hit"].mean()),
                "top3_accuracy": float(part_preds["top3_hit"].mean()),
                "top1_accuracy": float(part_preds["top1_hit"].mean()),
                "avg_rank": float(part_preds["actual_rank"].mean()),
                "avg_prob": float(part_preds["actual_prob"].mean()),
            }
    metrics["per_part"] = part_metrics

    # 3. Per-Position Accuracy
    pos_metrics = {}
    for pos in range(1, 5):
        pos_preds = pred_df[pred_df["position"] == pos]
        pos_metrics[f"QS{pos}"] = {
            "n": int(len(pos_preds)),
            "top5_accuracy": float(pos_preds["top5_hit"].mean()),
            "top3_accuracy": float(pos_preds["top3_hit"].mean()),
            "top1_accuracy": float(pos_preds["top1_hit"].mean()),
            "avg_rank": float(pos_preds["actual_rank"].mean()),
            "avg_confidence": float(pos_preds["top1_confidence"].mean()),
        }
    metrics["per_position"] = pos_metrics

    # 4. Stability Metrics
    part_accuracies = [part_metrics[str(p)]["top5_accuracy"] for p in LABELS if str(p) in part_metrics]
    pos_accuracies = [pos_metrics[f"QS{p}"]["top5_accuracy"] for p in range(1, 5)]

    metrics["stability"] = {
        "part_accuracy_cv": float(np.std(part_accuracies) / np.mean(part_accuracies))
            if np.mean(part_accuracies) > 0 else 0,
        "pos_accuracy_cv": float(np.std(pos_accuracies) / np.mean(pos_accuracies))
            if np.mean(pos_accuracies) > 0 else 0,
        "part_accuracy_range": float(max(part_accuracies) - min(part_accuracies)) if part_accuracies else 0,
        "pos_accuracy_range": float(max(pos_accuracies) - min(pos_accuracies)) if pos_accuracies else 0,
    }

    # 5. Calibration Metrics
    metrics["calibration"] = {
        "avg_top1_confidence": float(pred_df["top1_confidence"].mean()),
        "confidence_std": float(pred_df["top1_confidence"].std()),
        "avg_actual_prob": float(pred_df["actual_prob"].mean()),
    }

    # 6. Prediction Bias
    pred_dist = pred_df["pred_top1"].value_counts(normalize=True)
    actual_dist = pred_df["actual"].value_counts(normalize=True)

    bias = {}
    for part in LABELS:
        pred_pct = pred_dist.get(part, 0)
        actual_pct = actual_dist.get(part, 0)
        bias[str(part)] = float(pred_pct - actual_pct)

    metrics["prediction_bias"] = bias
    metrics["prediction_bias_max"] = float(max(abs(b) for b in bias.values()))

    # 7. Feature Importance
    if feature_importance:
        metrics["feature_importance"] = {
            "top_10": [{"feature": f, "importance": float(i)} for f, i in feature_importance[:10]],
            "lag_features_total": float(sum(i for f, i in feature_importance if "lag" in f)),
            "rolling_features_total": float(sum(i for f, i in feature_importance if "roll" in f)),
            "aggregate_features_total": float(sum(i for f, i in feature_importance if "agg" in f)),
        }

    # 8. Identify Hard/Easy Parts
    if part_metrics:
        sorted_parts = sorted(part_metrics.keys(), key=lambda p: part_metrics[p]["top5_accuracy"])
        metrics["part_ranking"] = {
            "hardest": sorted_parts[:3],
            "easiest": sorted_parts[-3:][::-1],
            "hard_parts_below_45pct": [p for p in sorted_parts if part_metrics[p]["top5_accuracy"] < 0.45]
        }

    return metrics


def check_alerts(metrics: Dict, baseline: Optional[Dict] = None) -> List[Dict]:
    """Check metrics against thresholds and generate alerts."""
    alerts = []

    # Accuracy alerts
    top5 = metrics["accuracy"]["pooled_top5_mean"]
    if top5 < THRESHOLDS["accuracy_critical"]:
        alerts.append({
            "level": "CRITICAL",
            "type": "ACCURACY_CRASH",
            "message": f"Top-5 pooled coverage at {top5:.1%} (threshold: {THRESHOLDS['accuracy_critical']:.0%})",
            "metric": "pooled_top5_mean",
            "value": top5
        })
    elif top5 < THRESHOLDS["accuracy_warning"]:
        alerts.append({
            "level": "WARNING",
            "type": "ACCURACY_LOW",
            "message": f"Top-5 pooled coverage at {top5:.1%} (threshold: {THRESHOLDS['accuracy_warning']:.0%})",
            "metric": "pooled_top5_mean",
            "value": top5
        })

    # CV alerts
    cv = metrics["stability"]["part_accuracy_cv"]
    if cv > THRESHOLDS["cv_warning"]:
        alerts.append({
            "level": "WARNING",
            "type": "HIGH_VARIANCE",
            "message": f"Part accuracy CV at {cv:.1%} (threshold: {THRESHOLDS['cv_warning']:.0%})",
            "metric": "part_accuracy_cv",
            "value": cv
        })

    # Bias alerts
    bias_max = metrics["prediction_bias_max"]
    if bias_max > 0.05:
        alerts.append({
            "level": "INFO",
            "type": "PREDICTION_BIAS",
            "message": f"Max prediction bias at {bias_max:.1%}",
            "metric": "prediction_bias_max",
            "value": bias_max
        })

    # Drift alerts (if baseline provided)
    if baseline and "accuracy" in baseline:
        prev_top5 = baseline["accuracy"].get("pooled_top5_mean", top5)
        drift = top5 - prev_top5
        if drift < -0.05:
            alerts.append({
                "level": "WARNING",
                "type": "ACCURACY_DRIFT",
                "message": f"Top-5 dropped {abs(drift):.1%} from previous period",
                "metric": "accuracy_drift",
                "value": drift
            })

    return alerts


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 70)
    print("METRICS COLLECTION")
    print("=" * 70)
    print(f"Window: {args.window_days} days")

    # Load data
    df = load_csv(args.daily_csv)
    sup = make_features(df)
    feature_cols = [c for c in sup.columns if c not in ("date", "date_idx", "y", "position")]

    unique_dates = sup["date"].unique()
    n_dates = len(unique_dates)

    # Define windows
    holdout_start = n_dates - args.window_days
    holdout_dates = unique_dates[holdout_start:]
    train_dates = unique_dates[:holdout_start - 30]

    train_df = sup[sup["date"].isin(train_dates)].dropna()
    holdout_df = sup[sup["date"].isin(holdout_dates)].dropna()

    print(f"\nEvaluation period: {holdout_dates[0].date()} to {holdout_dates[-1].date()}")
    print(f"Training on {len(train_dates)} days")

    # Train model
    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6, max_iter=300,
        random_state=args.seed, early_stopping=False
    )
    clf.fit(X_train, y_train)

    # Get feature importance
    try:
        feat_imp = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
    except AttributeError:
        feat_imp = []

    # Collect predictions
    print("\nCollecting predictions...")
    pred_df = collect_predictions(clf, holdout_df, feature_cols, holdout_dates)

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(pred_df, feat_imp)

    # Load baseline for comparison
    baseline = None
    if args.baseline_file and os.path.exists(args.baseline_file):
        with open(args.baseline_file) as f:
            baseline = json.load(f)

    # Check alerts
    alerts = check_alerts(metrics, baseline)
    metrics["alerts"] = alerts

    # Print summary
    print(f"\n{'=' * 70}")
    print("METRICS SUMMARY")
    print("=" * 70)

    acc = metrics["accuracy"]
    print(f"\nOVERALL ACCURACY:")
    print(f"  Pooled Top-5: {acc['pooled_top5_mean']:.1%} (std: {acc['pooled_top5_std']:.1%})")
    print(f"  Perfect Day Rate: {acc['perfect_day_rate_top5']:.0%}")
    print(f"  Per-Position Top-5: {acc['per_position_top5_mean']:.1%}")

    print(f"\nPER-PART TOP-5 ACCURACY:")
    for part in LABELS:
        if str(part) in metrics["per_part"]:
            p = metrics["per_part"][str(part)]
            status = "HARD" if p["top5_accuracy"] < 0.45 else "EASY" if p["top5_accuracy"] > 0.55 else ""
            print(f"  Part {part}: {p['top5_accuracy']:.1%} (n={p['n']}) {status}")

    print(f"\nPER-POSITION TOP-5 ACCURACY:")
    for pos in range(1, 5):
        p = metrics["per_position"][f"QS{pos}"]
        print(f"  QS{pos}: {p['top5_accuracy']:.1%} (conf: {p['avg_confidence']:.1%})")

    print(f"\nSTABILITY:")
    stab = metrics["stability"]
    print(f"  Part accuracy CV: {stab['part_accuracy_cv']:.1%}")
    print(f"  Position accuracy CV: {stab['pos_accuracy_cv']:.1%}")

    if alerts:
        print(f"\nALERTS ({len(alerts)}):")
        for a in alerts:
            print(f"  [{a['level']}] {a['type']}: {a['message']}")
    else:
        print(f"\nALERTS: None")

    # Save metrics
    out_path = os.path.join(args.out_dir, "metrics_current.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nMetrics saved: {out_path}")

    # Save predictions log if requested
    if args.save_predictions:
        pred_path = os.path.join(args.out_dir, "predictions_log.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions saved: {pred_path}")

    # Append to historical metrics
    hist_path = os.path.join(args.out_dir, "metrics_history.jsonl")
    with open(hist_path, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "window_days": args.window_days,
            "pooled_top5": acc["pooled_top5_mean"],
            "pooled_top5_cv": acc["pooled_top5_cv"],
            "perfect_day_rate": acc["perfect_day_rate_top5"],
            "hard_parts": metrics["part_ranking"]["hard_parts_below_45pct"],
            "alert_count": len(alerts)
        }, default=str) + "\n")
    print(f"Appended to history: {hist_path}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
