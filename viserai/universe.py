import pandas as pd
import requests
from io import StringIO

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def fetch_sp500():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(WIKI_URL, headers=headers, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))   # now parsing local HTML
    return tables[0]

def save_sp500_csv(path: str) -> None:
    df = fetch_sp500()
    df.to_csv(path, index=False)
