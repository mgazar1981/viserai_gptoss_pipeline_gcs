"""
scripts/update_universe.py

Purpose
-------
Fetch and persist the latest S&P 500 universe to a CSV file.

This script is intentionally thin: it provides a CLI wrapper around the
core business logic in `viserai.universe.save_sp500_csv(...)`.

Typical usage
-------------
Run from the repo root so imports resolve cleanly:

    python -m scripts.update_universe
    python -m scripts.update_universe --out universe/sp500.csv

Outputs
-------
- Writes a CSV of S&P 500 constituents to `--out`.
- Ensures the parent directory of `--out` exists (creates it if needed).
"""

from __future__ import annotations

import argparse
import os

from viserai.universe import save_sp500_csv


def main() -> None:
    # CLI definition:
    # - Keep arguments minimal here; most logic should live in viserai.universe.
    ap = argparse.ArgumentParser(
        description="Download the latest S&P 500 constituents and save as a CSV."
    )
    ap.add_argument(
        "--out",
        default="universe/sp500.csv",
        help="Output CSV path (directories will be created if needed).",
    )
    args = ap.parse_args()

    # Create the output directory (if the user provided one).
    # Note: os.path.dirname("sp500.csv") -> "" (no directory), so guard it.
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Delegate the actual fetching/parsing/writing to the library code.
    save_sp500_csv(args.out)

    # Simple confirmation for humans + logs.
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
