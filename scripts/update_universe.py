"""
scripts/update_universe.py

Fetch and persist the latest S&P 500 universe to a JSONL file.

This CLI wrapper delegates to `viserai.universe.save_sp500_jsonl(...)`, which writes
a *minimal* JSONL file containing ONLY:

  - symbol (ticker)
  - name   (company name)

Typical usage (run from repo root):
    python -m scripts.update_universe
    python -m scripts.update_universe --out universe/sp500.jsonl
"""

from __future__ import annotations

import argparse
import os

from viserai.universe import save_sp500_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download S&P 500 constituents and save a minimal JSONL (symbol, name)."
    )
    ap.add_argument(
        "--out",
        default="universe/sp500.jsonl",
        help="Output JSONL path (directories will be created if needed).",
    )
    args = ap.parse_args()

    # Ensure output directory exists (if user provided a directory component).
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_sp500_jsonl(args.out)
    print(f"saved minimal universe (symbol,name) as JSONL: {args.out}")


if __name__ == "__main__":
    main()
