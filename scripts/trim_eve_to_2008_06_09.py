#!/usr/bin/env python3
"""Trim CA_4_predict_eve_aggregate.csv to a given start date.

Usage:
    python scripts/trim_eve_to_2008_06_09.py \
      --in_csv CA_4_predict_eve_aggregate.csv \
      --out_csv CA_4_predict_eve_aggregate.csv

This script overwrites output if --out_csv points to the same path.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import pandas as pd


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description="Trim eve aggregate CSV to start date.")
    p.add_argument("--in_csv", type=str, required=True, help="Input eve CSV path.")
    p.add_argument("--out_csv", type=str, required=True, help="Output CSV path.")
    p.add_argument("--start_date", type=str, default="2008-06-09", help="Start date YYYY-MM-DD.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entrypoint.

    Args:
        argv: Optional argv list.

    Returns:
        Exit code.
    """
    args = parse_args(argv)
    df = pd.read_csv(args.in_csv)
    if "date" not in df.columns:
        raise ValueError("Missing 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    start = pd.to_datetime(args.start_date)
    out = df[df["date"] >= start].sort_values("date").reset_index(drop=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out)} rows to {args.out_csv} starting at {args.start_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# requirements.txt (copy/paste)
# pandas==2.2.2
