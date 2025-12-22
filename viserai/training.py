from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# -------- text formatting --------
def format_text_and_spans(rec: Dict[str, Any]) -> Tuple[str, Dict[str, Tuple[int,int]]]:
    ctx = {
        "ticker": rec["ticker"],
        "static": {k: rec.get("x", {}).get(k) for k in ["Symbol","Security","GICS Sector","GICS Sub-Industry"] if k in rec.get("x", {})},
        "h": rec.get("h", {}),
        "w": (rec.get("w", []) or [])[:10],
    }
    ctx_block = "<CTX>\n" + json.dumps(ctx, ensure_ascii=False) + "\n</CTX>\n"
    r_block = "<REASONING>\n" + str(rec.get("r","")) + "\n</REASONING>\n"
    a_block = "<ANALYSIS>\n" + str(rec.get("a","")) + "\n</ANALYSIS>\n"
    y1_block = "<PREDICTION_1>\n" + json.dumps(rec.get("y1", [])) + "\n</PREDICTION_1>\n"
    y2_block = "<PREDICTION_2>\n" + json.dumps(rec.get("y2", [])) + "\n</PREDICTION_2>\n"
    full = ctx_block + r_block + a_block + y1_block + y2_block

    def span(sub: str) -> Tuple[int,int]:
        i = full.find(sub)
        return (i, i+len(sub)) if i >= 0 else (-1,-1)

    spans = {
        "ctx": span(ctx_block),
        "r": span(r_block),
        "a": span(a_block),
        "y1": span(y1_block),
        "y2": span(y2_block),
    }
    spans["ar"] = (spans["r"][0], spans["a"][1]) if spans["r"][0] >= 0 and spans["a"][1] >= 0 else (-1,-1)
    return full, spans

# -------- dataset --------
class ViseraiDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.items=[]
        for r in records:
            if r.get("y") is None or r.get("sigma_x") is None:
                continue
            text, spans = format_text_and_spans(r)
            self.items.append({
                "text": text,
                "spans": spans,
                "y": np.array(r["y"], dtype=np.float32),
                "y1": np.array(r.get("y1", r["y"]), dtype=np.float32),
                "y2": np.array(r.get("y2", r["y"]), dtype=np.float32),
                "sigma_x": float(r["sigma_x"]),
            })

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def build_labels(enc_one: Dict[str, Any], spans: Dict[str, Tuple[int,int]], which: str) -> torch.Tensor:
    input_ids = torch.tensor(enc_one["input_ids"], dtype=torch.long)
    offsets = enc_one.get("offset_mapping")
    labels = torch.full_like(input_ids, -100)
    if offsets is None:
        return labels
    start, end = spans.get(which, (-1,-1))
    if start < 0 or end < 0:
        return labels
    for i,(s,e) in enumerate(offsets):
        if s==0 and e==0:
            continue
        if e <= start or s >= end:
            continue
        labels[i] = input_ids[i]
    return labels

class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [b["text"] for b in batch]
        spans = [b["spans"] for b in batch]
        enc = self.tok(texts, max_length=self.max_length, truncation=True, padding=True, return_offsets_mapping=True)
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attn = torch.tensor(enc["attention_mask"], dtype=torch.long)

        labels_ar, labels_y1, labels_y2 = [], [], []
        for i in range(len(batch)):
            one = {k: enc[k][i] for k in enc.keys()}
            labels_ar.append(build_labels(one, spans[i], "ar"))
            labels_y1.append(build_labels(one, spans[i], "y1"))
            labels_y2.append(build_labels(one, spans[i], "y2"))
        labels_ar = torch.stack(labels_ar, 0)
        labels_y1 = torch.stack(labels_y1, 0)
        labels_y2 = torch.stack(labels_y2, 0)

        y = torch.tensor(np.stack([b["y"] for b in batch], 0), dtype=torch.float32)
        y1 = torch.tensor(np.stack([b["y1"] for b in batch], 0), dtype=torch.float32)
        y2 = torch.tensor(np.stack([b["y2"] for b in batch], 0), dtype=torch.float32)
        sigma_x = torch.tensor([b["sigma_x"] for b in batch], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels_ar": labels_ar,
            "labels_y1": labels_y1,
            "labels_y2": labels_y2,
            "y": y,
            "y1": y1,
            "y2": y2,
            "sigma_x": sigma_x,
        }

# -------- custom trainer (kernel-weighted) --------
def kernel(u: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dist2 = torch.mean((u-v)**2, dim=-1)
    return torch.exp(-0.5 * dist2 / (sigma**2 + eps))

class ViseraiTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask")
        labels_ar = inputs["labels_ar"]
        labels_y1 = inputs["labels_y1"]
        labels_y2 = inputs["labels_y2"]
        y = inputs["y"].to(input_ids.device)
        y1 = inputs["y1"].to(input_ids.device)
        y2 = inputs["y2"].to(input_ids.device)
        sigma = inputs["sigma_x"].to(input_ids.device)

        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits  # (B,T,V)

        shift_logits = logits[:, :-1, :].contiguous()
        V = shift_logits.size(-1)

        def per_ex_nll(labels: torch.Tensor) -> torch.Tensor:
            labels = labels[:, 1:].contiguous()
            loss_tok = F.cross_entropy(
                shift_logits.view(-1, V),
                labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(labels.size(0), -1)
            return torch.sum(loss_tok, dim=-1)

        nll_ar = per_ex_nll(labels_ar)
        nll_1 = per_ex_nll(labels_y1)
        nll_2 = per_ex_nll(labels_y2)

        w = kernel(y,y1,sigma) + kernel(y,y2,sigma) - kernel(y1,y2,sigma)
        # w = torch.clamp(w, min=0.0)  # relu

        loss_vec = nll_ar + (nll_1 + nll_2) * w
        loss = torch.mean(loss_vec)
        return (loss, out) if return_outputs else loss

# -------- training entrypoint --------
def maybe_lora(model, enabled: bool, r: int, alpha: int, dropout: float):
    if not enabled:
        return model
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model

def train(records: List[Dict[str, Any]], model_name_or_path: str, output_dir: str, max_seq_len: int,
          max_steps: int, batch_size: int, lr: float, lora_cfg: Dict[str, Any]):
    ds = ViseraiDataset(records)
    if len(ds)==0:
        raise ValueError("No completed records (need y and sigma_x).")

    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = maybe_lora(model,
                       enabled=bool(lora_cfg.get("enabled", True)),
                       r=int(lora_cfg.get("r",16)),
                       alpha=int(lora_cfg.get("alpha",32)),
                       dropout=float(lora_cfg.get("dropout",0.05)))

    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
    )

    trainer = ViseraiTrainer(model=model, args=args, train_dataset=ds, data_collator=Collator(tok, max_seq_len), tokenizer=tok)
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
