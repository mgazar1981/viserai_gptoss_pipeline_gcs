from __future__ import annotations
import os
import yaml
from typing import Any, Dict

def load_config(path: str = "configs/pipeline.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # env override
    cfg["data_dir"] = os.environ.get("VISERAI_DATA_DIR", cfg.get("data_dir", "./data"))
    return cfg
