from __future__ import annotations
import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def fetch_sp500() -> pd.DataFrame:
    tables = pd.read_html(WIKI_URL)
    df = tables[0].copy()
    df.columns = [c.strip() for c in df.columns]
    df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return df.sort_values("Symbol").reset_index(drop=True)

def save_sp500_csv(path: str) -> None:
    df = fetch_sp500()
    df.to_csv(path, index=False)
