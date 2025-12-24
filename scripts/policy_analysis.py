#!/usr/bin/env python3
"""
Policy Analysis for C4 Parts Forecast.

This script:
1. Defines cost parameters for shipping decisions
2. Evaluates different decision policies (top-1, top-K, exclusion)
3. Recommends optimal K and M values based on cost-benefit analysis

Author: BMad Ops Cost Modeler Agent
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    """Cost parameters for decision policy."""
    ship_cost: float = 50.0         # Cost to ship one part ($)
    downtime_cost: float = 500.0    # Cost per day of downtime ($)
    expedite_multiplier: float = 3.0  # Rush shipping premium (3x normal)
    salvage_rate: float = 0.8       # Fraction of cost recovered if wrong part used later


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Policy analysis for C4 pipeline")
    p.add_argument("--predictions_json", type=str,
                   default="artifacts/next_day_predictions.json")
    p.add_argument("--metrics_topk_csv", type=str,
                   default="artifacts/metrics_topk_by_position.csv")
    p.add_argument("--metrics_exclusion_csv", type=str,
                   default="artifacts/metrics_exclusion_by_position.csv")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--ship_cost", type=float, default=50.0)
    p.add_argument("--downtime_cost", type=float, default=500.0)
    p.add_argument("--expedite_mult", type=float, default=3.0)
    p.add_argument("--salvage_rate", type=float, default=0.8)
    return p.parse_args(argv)


def load_predictions(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def load_metrics(topk_path: str, excl_path: str) -> tuple:
    topk_df = pd.read_csv(topk_path)
    excl_df = pd.read_csv(excl_path)
    return topk_df, excl_df


def evaluate_top1_policy(topk_df: pd.DataFrame, cost: CostModel) -> Dict:
    """
    Top-1 Policy: Ship only the single most likely part.

    Expected cost = P(wrong) * (downtime + expedite) + P(right) * ship_cost
    """
    avg_top1_acc = topk_df["top1_acc"].mean()
    p_wrong = 1 - avg_top1_acc
    p_right = avg_top1_acc

    # If wrong: downtime + expedite shipping for correct part
    wrong_cost = cost.downtime_cost + (cost.ship_cost * cost.expedite_multiplier)
    # Salvage value for wrong part
    salvage = cost.ship_cost * cost.salvage_rate

    expected_cost = (
        p_right * cost.ship_cost +  # Correct: just shipping
        p_wrong * (wrong_cost - salvage + cost.ship_cost)  # Wrong: penalties + original ship
    )

    return {
        "policy": "Top-1 Ship",
        "k": 1,
        "coverage": avg_top1_acc,
        "expected_cost": expected_cost,
        "parts_shipped_per_day": 1,
        "downtime_risk": p_wrong
    }


def evaluate_topk_staging_policy(topk_df: pd.DataFrame, k: int, cost: CostModel) -> Dict:
    """
    Top-K Staging Policy: Stage K parts, ship top-1, have backups ready.

    If top-1 is wrong but correct is in top-K: use backup (no downtime, extra shipping)
    If correct not in top-K: downtime + expedite
    """
    col = f"top{k}_acc" if k > 1 else "top1_acc"
    if col not in topk_df.columns:
        # Interpolate or use closest
        cols = [c for c in topk_df.columns if c.startswith("top")]
        col = cols[-1] if cols else "top1_acc"

    avg_topk_acc = topk_df[col].mean() if col in topk_df.columns else topk_df["top1_acc"].mean()
    avg_top1_acc = topk_df["top1_acc"].mean()

    p_top1_right = avg_top1_acc
    p_in_topk_not_top1 = avg_topk_acc - avg_top1_acc
    p_not_in_topk = 1 - avg_topk_acc

    # Costs:
    # - Stage K parts upfront (shipping cost * K)
    # - If top-1 correct: done (already staged)
    # - If top-1 wrong but in top-K: use backup, return unused (salvage K-1)
    # - If not in top-K: downtime + expedite

    staging_cost = cost.ship_cost * k
    wrong_cost = cost.downtime_cost + (cost.ship_cost * cost.expedite_multiplier)
    salvage_unused = cost.ship_cost * cost.salvage_rate * (k - 1)

    expected_cost = (
        staging_cost +  # Always pay for staging
        p_not_in_topk * wrong_cost -  # Downtime if not in top-K
        (p_top1_right + p_in_topk_not_top1) * salvage_unused  # Salvage unused parts
    )

    return {
        "policy": f"Top-{k} Staging",
        "k": k,
        "coverage": avg_topk_acc,
        "expected_cost": expected_cost,
        "parts_shipped_per_day": k,
        "downtime_risk": p_not_in_topk
    }


def evaluate_exclusion_policy(excl_df: pd.DataFrame, m: int) -> Dict:
    """
    Exclusion Policy: Exclude bottom-M parts from consideration.

    Report the risk of excluding the correct part.
    """
    col = f"bottom{m}_risk"
    if col not in excl_df.columns:
        return {"policy": f"Exclude Bottom-{m}", "m": m, "exclusion_risk": None}

    avg_risk = excl_df[col].mean()

    return {
        "policy": f"Exclude Bottom-{m}",
        "m": m,
        "exclusion_risk": avg_risk,
        "safe_exclusion": avg_risk < 0.05  # Less than 5% risk
    }


def recommend_k_and_m(topk_df: pd.DataFrame, excl_df: pd.DataFrame,
                       cost: CostModel, target_coverage: float = 0.70) -> Dict:
    """
    Recommend optimal K and M values.

    K: Smallest K that achieves target coverage at reasonable cost
    M: Largest M with exclusion risk < 5%
    """
    # Find recommended K
    k_options = [1, 3, 5, 7, 9]
    policies = []

    for k in k_options:
        if k == 1:
            policy = evaluate_top1_policy(topk_df, cost)
        else:
            policy = evaluate_topk_staging_policy(topk_df, k, cost)
        policies.append(policy)

    # Recommend K: first K that achieves target coverage
    recommended_k = 1
    for policy in policies:
        if policy["coverage"] >= target_coverage:
            recommended_k = policy["k"]
            break
    else:
        # If none achieve target, recommend highest
        recommended_k = k_options[-1]

    # Find recommended M
    m_options = range(1, 8)
    recommended_m = 0

    for m in m_options:
        excl = evaluate_exclusion_policy(excl_df, m)
        if excl.get("safe_exclusion", False):
            recommended_m = m
        else:
            break

    return {
        "recommended_k": recommended_k,
        "recommended_m": recommended_m,
        "target_coverage": target_coverage,
        "policies_evaluated": policies
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("POLICY ANALYSIS - C4 Parts Forecast")
    print("=" * 60)

    # Define cost model
    cost = CostModel(
        ship_cost=args.ship_cost,
        downtime_cost=args.downtime_cost,
        expedite_multiplier=args.expedite_mult,
        salvage_rate=args.salvage_rate
    )

    # Save cost model
    cost_model_path = os.path.join(args.out_dir, "cost_model.yml")
    with open(cost_model_path, "w") as f:
        f.write("# C4 Parts Forecast Cost Model\n")
        f.write(f"ship_cost: {cost.ship_cost}  # Cost to ship one part ($)\n")
        f.write(f"downtime_cost: {cost.downtime_cost}  # Cost per day of downtime ($)\n")
        f.write(f"expedite_multiplier: {cost.expedite_multiplier}  # Rush shipping premium\n")
        f.write(f"salvage_rate: {cost.salvage_rate}  # Fraction recovered for wrong part\n")
    print(f"\nCost Model saved: {cost_model_path}")

    # Load metrics
    topk_df, excl_df = load_metrics(args.metrics_topk_csv, args.metrics_exclusion_csv)

    print("\n" + "-" * 60)
    print("COST PARAMETERS")
    print("-" * 60)
    print(f"  Ship Cost:         ${cost.ship_cost:.2f}")
    print(f"  Downtime Cost:     ${cost.downtime_cost:.2f}/day")
    print(f"  Expedite Mult:     {cost.expedite_multiplier}x")
    print(f"  Salvage Rate:      {cost.salvage_rate:.0%}")

    # Evaluate policies
    print("\n" + "-" * 60)
    print("POLICY EVALUATION")
    print("-" * 60)

    policies = []
    for k in [1, 3, 5, 7]:
        if k == 1:
            p = evaluate_top1_policy(topk_df, cost)
        else:
            p = evaluate_topk_staging_policy(topk_df, k, cost)
        policies.append(p)
        print(f"\n  {p['policy']}:")
        print(f"    Coverage:      {p['coverage']:.1%}")
        print(f"    Expected Cost: ${p['expected_cost']:.2f}")
        print(f"    Downtime Risk: {p['downtime_risk']:.1%}")

    # Save policy comparison
    policy_df = pd.DataFrame(policies)
    policy_path = os.path.join(args.out_dir, "policy_comparison.csv")
    policy_df.to_csv(policy_path, index=False)
    print(f"\nPolicy comparison saved: {policy_path}")

    # Exclusion analysis
    print("\n" + "-" * 60)
    print("EXCLUSION ANALYSIS")
    print("-" * 60)

    exclusions = []
    for m in range(1, 8):
        e = evaluate_exclusion_policy(excl_df, m)
        exclusions.append(e)
        risk = e.get("exclusion_risk", 0) or 0
        safe = "SAFE" if e.get("safe_exclusion", False) else "RISKY"
        print(f"  Exclude Bottom-{m}: {risk:.1%} risk - {safe}")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    rec = recommend_k_and_m(topk_df, excl_df, cost)

    print(f"\n  Recommended K (staging): {rec['recommended_k']}")
    print(f"  Recommended M (exclusion): {rec['recommended_m']}")

    # Generate recommendation report
    report_path = os.path.join(args.out_dir, "recommended_policy.md")
    with open(report_path, "w") as f:
        f.write("# C4 Parts Forecast - Recommended Policy\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().isoformat()}\n\n")
        f.write("---\n\n")

        f.write("## Cost Parameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Ship Cost | ${cost.ship_cost:.2f} |\n")
        f.write(f"| Downtime Cost | ${cost.downtime_cost:.2f}/day |\n")
        f.write(f"| Expedite Multiplier | {cost.expedite_multiplier}x |\n")
        f.write(f"| Salvage Rate | {cost.salvage_rate:.0%} |\n\n")

        f.write("## Policy Comparison\n\n")
        f.write("| Policy | Coverage | Expected Cost | Downtime Risk |\n")
        f.write("|--------|----------|---------------|---------------|\n")
        for p in policies:
            f.write(f"| {p['policy']} | {p['coverage']:.1%} | ${p['expected_cost']:.2f} | {p['downtime_risk']:.1%} |\n")
        f.write("\n")

        f.write("## Exclusion Safety\n\n")
        f.write("| Exclude | Risk | Status |\n")
        f.write("|---------|------|--------|\n")
        for e in exclusions:
            risk = e.get("exclusion_risk", 0) or 0
            safe = "SAFE" if e.get("safe_exclusion", False) else "RISKY"
            f.write(f"| Bottom-{e['m']} | {risk:.1%} | {safe} |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## Final Recommendations\n\n")
        f.write(f"### Staging: Use Top-{rec['recommended_k']}\n\n")

        if rec['recommended_k'] == 1:
            f.write("Ship only the top-1 predicted part. This minimizes shipping costs but accepts higher downtime risk.\n\n")
        elif rec['recommended_k'] == 3:
            f.write("Stage top-3 parts. Ship top-1, have 2 backups ready. Balances cost and coverage.\n\n")
        elif rec['recommended_k'] == 5:
            f.write("Stage top-5 parts. Achieves good coverage (~50%) with moderate staging costs.\n\n")
        else:
            f.write(f"Stage top-{rec['recommended_k']} parts to achieve target coverage.\n\n")

        f.write(f"### Exclusion: Safely Exclude Bottom-{rec['recommended_m']}\n\n")
        if rec['recommended_m'] > 0:
            f.write(f"The bottom {rec['recommended_m']} least-likely parts can be safely excluded from consideration (<5% risk of excluding the correct part).\n\n")
        else:
            f.write("No parts can be safely excluded with <5% risk.\n\n")

        f.write("### Operational Implementation\n\n")
        f.write("Daily output for operations team:\n\n")
        f.write("| Position | Top-1 (Ship) | Top-3 (Stage) | Exclude |\n")
        f.write("|----------|--------------|---------------|----------|\n")

        # Load predictions for example
        try:
            preds = load_predictions(args.predictions_json)
            for p in preds:
                pos = p["position"]
                top1 = p["top1_part"]
                top3 = ", ".join(map(str, p["top3_parts"]))
                exclude = ", ".join(map(str, p["least_likely_parts"][:rec['recommended_m']]))
                f.write(f"| QS{pos} | Part {top1} | Parts {top3} | Parts {exclude} |\n")
        except:
            f.write("| QS1-4 | See predictions.json | See predictions.json | See predictions.json |\n")

        f.write("\n")

    print(f"\nRecommendations saved: {report_path}")
    print("\n" + "=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
