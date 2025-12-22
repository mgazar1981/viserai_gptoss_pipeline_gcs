from __future__ import annotations
import argparse, os
from datetime import date
import pandas as pd
from viserai.config import load_config
from viserai.pre_market import run_pre_market
from viserai.utils import next_trading_day, is_trading_day

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: next trading day)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n_passes", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(cfg["universe_path"])
    if args.limit: df = df.head(args.limit)

    if args.date:
        trading_date = date.fromisoformat(args.date)
    else:
        today = date.today()
        trading_date = next_trading_day(today) if is_trading_day(today) else next_trading_day(today)

    n_passes = int(args.n_passes or cfg["research"]["n_passes"])
    path = run_pre_market(
        universe_df=df,
        trading_date=trading_date,
        data_dir=cfg["data_dir"],
        market_provider_name=cfg["intraday"]["provider"],
        fred_series=cfg["macro"]["fred_series"],
        n_passes=n_passes,
        web_provider=cfg["research"]["web_provider"],
        max_web_snippets=int(cfg["research"]["max_web_snippets"]),
        serpapi_key_env=str(cfg["research"]["serpapi_api_key_env"]),
        H=int(cfg["dataset"]["horizon_points"]),
    )
    print(f"wrote: {path}")

if __name__ == "__main__":
    main()
