from __future__ import annotations
from datetime import datetime, timedelta, date
from typing import Dict, Any, List
import os, json
import pandas as pd

from .providers import get_market_provider, fetch_macro_bundle, search_web, summarize_intraday
from .gpt_client import get_gpt_client
from .dataset_store import PassRecord, LocalStore

def build_prompt(ticker: str, static: Dict[str, Any], hist_tail: List[Dict[str, Any]], macro: Dict[str, Any], web: List[Dict[str, Any]], H: int) -> str:
    obj = {
        "ticker": ticker,
        "static": static,
        "historical_intraday_30m_tail": hist_tail,
        "macro": macro,
        "web": web,
        "task": {"interval_minutes": 30, "horizon_points": H, "output": "y1,y2 vectors"},
    }
    return json.dumps(obj, ensure_ascii=False)

def run_pre_market(
    universe_df: pd.DataFrame,
    trading_date: date,
    data_dir: str,
    market_provider_name: str,
    fred_series: Dict[str, str],
    n_passes: int,
    web_provider: str,
    max_web_snippets: int,
    serpapi_key_env: str,
    H: int,
) -> str:
    store = LocalStore(os.path.join(data_dir, "dataset"))
    market = get_market_provider(market_provider_name)
    gpt = get_gpt_client()

    # historical window ending at trading_date (exclusive)
    start = trading_date - timedelta(days=15)
    end = trading_date

    macro = fetch_macro_bundle(fred_series)

    records: List[PassRecord] = []
    for _, row in universe_df.iterrows():
        ticker = str(row["Symbol"]).strip()
        static = row.to_dict()

        try:
            df_hist = market.fetch_intraday_30m(ticker, start=start, end=end)
            hist_tail = summarize_intraday(df_hist, max_rows=60)
        except Exception:
            hist_tail = []

        query = f"{ticker} stock news earnings guidance macro"
        try:
            web = search_web(query, provider=web_provider, max_results=max_web_snippets, serpapi_key_env=serpapi_key_env)
        except Exception:
            web = []

        prompt = build_prompt(ticker, static, hist_tail, macro, web, H=H)

        for pid in range(n_passes):
            resp = gpt.generate_pass(prompt, H=H)
            records.append(PassRecord(
                schema_version=1,
                trading_date=str(trading_date),
                ticker=ticker,
                pass_id=pid,
                x=static,
                h={"intraday_hist_tail": hist_tail, "macro": macro},
                w=web,
                r=resp.reasoning,
                a=resp.analysis,
                y1=resp.y1,
                y2=resp.y2,
                y=None,
                sigma_x=None,
                created_utc=datetime.utcnow().isoformat() + "Z",
                meta={"market_provider": market_provider_name, "web_provider": web_provider},
            ))

    return store.append("passes", str(trading_date), records)
