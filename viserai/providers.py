from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import requests
from io import StringIO

# -------- Market (intraday 30m) --------
class MarketDataProvider:
    def fetch_intraday_30m(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError

class YFinanceProvider(MarketDataProvider):
    def __init__(self):
        import yfinance as yf
        self.yf = yf

    def fetch_intraday_30m(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        df = self.yf.download(
            tickers=ticker,
            start=str(start),
            end=str(end + timedelta(days=1)),
            interval="30m",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if df is None or len(df) == 0:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df

def get_market_provider(name: str) -> MarketDataProvider:
    name = (name or "yfinance").lower()
    if name == "yfinance":
        return YFinanceProvider()
    raise ValueError(f"Unknown market provider: {name}")

def summarize_intraday(df: pd.DataFrame, max_rows: int = 60) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    cols = [c for c in ["datetime","open","high","low","close","volume"] if c in df.columns]
    return jsonable(df[cols].tail(max_rows).to_dict(orient="records"))

def extract_session_vector(df: pd.DataFrame, trading_date: date, H: int) -> Optional[np.ndarray]:
    if df is None or df.empty or "close" not in df.columns:
        return None
    # infer timestamp column
    ts_col = "datetime" if "datetime" in df.columns else df.columns[0]
    ts = pd.to_datetime(df[ts_col])
    sub = df.copy()
    sub["_d"] = ts.dt.date
    sub = sub[sub["_d"] == trading_date]
    if sub.empty:
        return None
    closes = sub["close"].astype(float).to_numpy()
    if len(closes) >= H:
        closes = closes[:H]
    else:
        closes = np.pad(closes, (0, H-len(closes)), mode="edge")
    return closes.astype(np.float32)

# -------- Macro (FRED CSV) --------
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

def fetch_fred_series(series_id: str) -> List[Dict[str, Any]]:
    url = FRED_CSV.format(series=series_id)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text)).tail(300)
    return jsonable(df.to_dict(orient="records"))

def fetch_macro_bundle(series: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for sid in series.keys():
        try:
            out[sid] = fetch_fred_series(sid)
        except Exception as e:
            out[sid] = {"error": str(e)}
    return out

# -------- Web search (optional) --------
def serpapi_search(query: str, api_key: str, max_results: int = 10) -> List[Dict[str, Any]]:
    url = "https://serpapi.com/search.json"
    params = {"q": query, "engine": "google", "api_key": api_key, "num": max_results}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    out = []
    for item in data.get("organic_results", [])[:max_results]:
        out.append({
            "title": str(item.get("title","")),
            "link": str(item.get("link","")),
            "snippet": str(item.get("snippet","")),
        })
    return out

def search_web(query: str, provider: str, max_results: int, serpapi_key_env: str) -> List[Dict[str, Any]]:
    provider = (provider or "none").lower()
    if provider == "none":
        return []
    if provider == "serpapi":
        import os
        key = os.environ.get(serpapi_key_env, "")
        if not key:
            return []
        return serpapi_search(query, key, max_results=max_results)
    raise ValueError(f"Unknown web provider: {provider}")

# -------- helpers --------
def jsonable(x: Any) -> Any:
    # best-effort conversion for numpy types
    if isinstance(x, dict):
        return {k: jsonable(v) for k,v in x.items()}
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return x
