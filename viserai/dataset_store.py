from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Iterable
import os, json
from glob import glob

@dataclass
class PassRecord:
    schema_version: int
    trading_date: str
    ticker: str
    pass_id: int
    x: Dict[str, Any]
    h: Dict[str, Any]
    w: List[Dict[str, Any]]
    r: str
    a: str
    y1: List[float]
    y2: List[float]
    y: Optional[List[float]] = None
    sigma_x: Optional[float] = None
    created_utc: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PassRecord":
        return PassRecord(**d)

class LocalStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _path(self, kind: str, trading_date: str) -> str:
        d = os.path.join(self.root, kind, trading_date)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "records.jsonl")

    def append(self, kind: str, trading_date: str, records: List[PassRecord]) -> str:
        path = self._path(kind, trading_date)
        with open(path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        return path

    def overwrite(self, kind: str, trading_date: str, records: List[PassRecord]) -> str:
        path = self._path(kind, trading_date)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        return path

    def read(self, kind: str, trading_date: str) -> List[PassRecord]:
        path = self._path(kind, trading_date)
        if not os.path.exists(path):
            return []
        out=[]
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                out.append(PassRecord.from_dict(json.loads(line)))
        return out

    def iter_last_n_days(self, kind: str, n_days: int) -> Iterable[PassRecord]:
        base = os.path.join(self.root, kind)
        if not os.path.exists(base):
            return []
        days = sorted([os.path.basename(p) for p in glob(os.path.join(base, "*"))])[-n_days:]
        for d in days:
            for rec in self.read(kind, d):
                yield rec
