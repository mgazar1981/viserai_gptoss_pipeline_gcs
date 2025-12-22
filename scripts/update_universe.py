from __future__ import annotations
import argparse, os
from viserai.universe import save_sp500_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="universe/sp500.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_sp500_csv(args.out)
    print(f"saved: {args.out}")

if __name__ == "__main__":
    main()
