from __future__ import annotations
import argparse, os
from typing import List, Dict, Any
from viserai.config import load_config
from viserai.dataset_store import LocalStore
from viserai.training import train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--output_dir", default="./checkpoints/latest")
    ap.add_argument("--max_seq_len", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--recent_days", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = LocalStore(os.path.join(cfg["data_dir"], "dataset"))
    n_days = int(args.recent_days or cfg["training"]["recent_days"])

    records: List[Dict[str, Any]] = [r.to_dict() for r in store.iter_last_n_days("complete", n_days)]
    if args.limit: records = records[:args.limit]

    max_seq = int(args.max_seq_len or cfg["training"]["max_seq_len"])
    os.makedirs(args.output_dir, exist_ok=True)
    train(records, args.model_name_or_path, args.output_dir, max_seq, args.max_steps, args.batch_size, args.lr, cfg["training"]["lora"])
    print(f"saved checkpoint: {args.output_dir}")

if __name__ == "__main__":
    main()
