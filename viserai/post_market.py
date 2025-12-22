from __future__ import annotations
from datetime import date
from typing import Dict, Any, List
import os
import numpy as np
import pandas as pd

from .providers import get_market_provider, extract_session_vector
from .dataset_store import LocalStore, PassRecord

def compute_sigma(recent: List[np.ndarray], eps: float = 1e-6) -> float:
    if not recent:
        return 1.0
    arr = np.stack(recent, axis=0)
    return float(max(np.std(arr), eps))

def run_post_market(
    universe_df: pd.DataFrame,
    trading_date: date,
    data_dir: str,
    market_provider_name: str,
    H: int,
    lookback_days: int,
) -> str:
    store = LocalStore(os.path.join(data_dir, "dataset"))
    market = get_market_provider(market_provider_name)

    records = store.read("passes", str(trading_date))
    if not records:
        return ""

    sigma_cache: Dict[str, float] = {}
    updated: List[PassRecord] = []

    # realized for the same trading_date
    for rec in records:
        ticker = rec.ticker
        try:
            df_real = market.fetch_intraday_30m(ticker, start=trading_date, end=trading_date)
            y = extract_session_vector(df_real, trading_date=trading_date, H=H)
        except Exception:
            y = None

        if y is None:
            updated.append(rec)
            continue

        if ticker not in sigma_cache:
            # best-effort sigma from prior completed records
            recent_vecs: List[np.ndarray] = []
            complete_dir = os.path.join(data_dir, "dataset", "complete")
            if os.path.exists(complete_dir):
                days = sorted([d for d in os.listdir(complete_dir) if os.path.isdir(os.path.join(complete_dir, d))])[-lookback_days:]
                for d in days:
                    for r in store.read("complete", d):
                        if r.ticker == ticker and r.y is not None:
                            try:
                                recent_vecs.append(np.array(r.y, dtype=np.float32))
                            except Exception:
                                pass
            sigma_cache[ticker] = compute_sigma(recent_vecs)

        rec.y = y.tolist()
        rec.sigma_x = float(sigma_cache[ticker])
        updated.append(rec)

    return store.overwrite("complete", str(trading_date), updated)
