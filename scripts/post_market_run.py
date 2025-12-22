from __future__ import annotations
import argparse
from datetime import date
import pandas as pd
from viserai.config import load_config
from viserai.post_market import run_post_market
from viserai.utils import previous_trading_day

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: previous trading day)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(cfg["universe_path"])
    if args.limit: df = df.head(args.limit)

    trading_date = date.fromisoformat(args.date) if args.date else previous_trading_day(date.today())

    path = run_post_market(
        universe_df=df,
        trading_date=trading_date,
        data_dir=cfg["data_dir"],
        market_provider_name=cfg["intraday"]["provider"],
        H=int(cfg["dataset"]["horizon_points"]),
        lookback_days=int(cfg["intraday"]["sigma_lookback_days"]),
    )
    print(f"wrote: {path}")

if __name__ == "__main__":
    main()
