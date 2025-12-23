# Viserai GPT-OSS daily pipeline (PyTorch scaffold)

Implements the daily loop:
- **Universe**: S&P 500 tickers
- **Pre-market**: ingest intraday(30m)+macro, run **n** deep-research passes per ticker, output `(x,h,w,r,a,y1,y2)`
- **Post-close**: ingest realized intraday vector `y`, compute `sigma_x`, update to `(x,h,w,r,a,y,y1,y2)`
- **Train**: SFT (LoRA recommended) with custom kernel-weighted prediction loss, save new checkpoint

This is a scaffold meant to run end-to-end on small models locally, and be extended for 120B-scale infra.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Universe
```bash
python -m scripts.update_universe
```

## 2) Pre-market (use dummy GPT client for testing)
```bash
export VISERAI_GPT_MODE=dummy
python -m scripts.pre_market_run --limit 10 --n_passes 2
```

## 3) Post-market (fills realized y for previous trading day)
```bash
python -m script.spost_market_run.py --limit 10
```

## 4) Train (use small model for smoke test)
```bash
python -m scripts.train_daily --model_name_or_path gpt2 --max_steps 50 --limit 200
```

## GPT-OSS inference wiring
Default client expects an OpenAI-compatible endpoint:
- `VISERAI_OPENAI_BASE_URL` (e.g. http://localhost:8000)
- `VISERAI_OPENAI_API_KEY` (optional)
- `VISERAI_OPENAI_MODEL` (e.g. gpt-oss-120b)

## Data layout
- `data/dataset/passes/YYYY-MM-DD/records.jsonl`
- `data/dataset/complete/YYYY-MM-DD/records.jsonl`

## Custom loss (implemented)
Token NLL over:
- reasoning+analysis segment: `NLL_ar`
- prediction segments: `NLL_y1`, `NLL_y2`

Kernel:
`K(u,v)=exp(-0.5*||u-v||^2/(sigma_x^2+eps))`
`w = K(y,y1) + K(y,y2) - K(y1,y2)`

Loss:
`Loss = NLL_ar + (NLL_y1 + NLL_y2) * relu(w)`

Adjust in `viserai/training.py` if you want a different parenthesization.
