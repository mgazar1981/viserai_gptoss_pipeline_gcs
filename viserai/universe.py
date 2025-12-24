"""
viserai/universe.py

Utilities for managing the asset universe (e.g., S&P 500 constituents).

This module fetches the current S&P 500 constituents table from Wikipedia and
writes a *minimal* JSONL file containing only:

  - symbol  (ticker)
  - name    (company name)

Why the extra HTTP handling?
---------------------------
Wikipedia (and some corporate networks) will sometimes block "bot-like" HTTP
clients. `pandas.read_html(url)` uses urllib under the hood, which can trigger
403s. We therefore fetch the HTML ourselves with a browser-like User-Agent and
then parse it locally via `pandas.read_html(StringIO(html))`.
"""

from __future__ import annotations

import json
from io import StringIO
from typing import Iterable

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Browser-like UA to reduce 403 blocks from Wikipedia.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """
    Pick the first matching column from `candidates`.

    We try exact matching first, then fall back to case-insensitive matching.
    This makes the code resilient to small column-name changes.
    """
    cols = list(df.columns)

    # Exact match
    for c in candidates:
        if c in cols:
            return c

    # Case-insensitive match
    lower_map = {str(col).strip().lower(): col for col in cols}
    for c in candidates:
        key = str(c).strip().lower()
        if key in lower_map:
            return lower_map[key]

    raise KeyError(f"Could not find any of {list(candidates)}. Available columns: {cols}")


def fetch_sp500() -> pd.DataFrame:
    """
    Fetch the *full* S&P 500 constituents table from Wikipedia.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the table columns from Wikipedia (e.g. Symbol,
        Security, sector, sub-industry, HQ location, founded, etc.).
    """
    resp = requests.get(WIKI_URL, headers=DEFAULT_HEADERS, timeout=30)
    resp.raise_for_status()

    # Parse all tables and select the one that has the expected core columns.
    tables = pd.read_html(StringIO(resp.text))
    for t in tables:
        norm = {str(c).strip().lower() for c in t.columns}
        if "symbol" in norm and ("security" in norm or "name" in norm):
            return t

    # Fallback: historically the constituents table is often the first one.
    if tables:
        return tables[0]

    raise ValueError("No tables found on the S&P 500 Wikipedia page.")


def sp500_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a full Wikipedia constituents DataFrame into a minimal one
    with only the two fields we want to persist.

    Output columns:
      - symbol
      - name
    """
    symbol_col = _pick_col(df, ["Symbol", "Ticker", "ticker", "symbol"])
    name_col = _pick_col(df, ["Security", "Name", "Company", "company", "security"])

    out = df[[symbol_col, name_col]].rename(
        columns={symbol_col: "symbol", name_col: "name"}
    )

    # Light cleanup: strip whitespace and coerce to string.
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["name"] = out["name"].astype(str).str.strip()

    return out


def save_sp500_jsonl(path: str) -> None:
    """
    Save a minimal JSONL file of S&P 500 constituents to `path`.

    Each line is a JSON object with ONLY:
      - symbol
      - name
    """
    df_full = fetch_sp500()
    df_min = sp500_minimal(df_full)

    records = df_min.to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            # ensure_ascii=False keeps company names readable; sort_keys=True makes diffs stable.
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
