
**Table 4.2** — DistilBERT fine-tuning configuration

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| Base model | `distilbert-base-uncased` | Warmup steps | 1,000 |
| Split | 70/15/15, seed 42 | Weight decay | 0.01 |
| Epochs | 4 (early stopping, patience 2) | Precision | fp16 |
| Batch size | 32 train / 64 eval | Class weights | Balanced, weighted cross-entropy |
| Learning rate | 2e-5 | Selection metric | Validation macro F1 |

**Table 4.11** — VADER neutral-band sweep, diagnostic sample *(source: `data/results/sentiment_diagnostics/vader_threshold_sweep.csv`)*

| Neutral band | Accuracy | Macro F1 | Neutral F1 | Predicted neutral (%) |
|---:|---:|---:|---:|---:|
| ±0.05 (default) | 0.5539 | 0.5094 | 0.2526 | 10.2 |
| ±0.10 | 0.5567 | 0.5179 | 0.2812 | 12.1 |
| ±0.15 | 0.5594 | 0.5258 | 0.3075 | 14.0 |
| ±0.20 | **0.5628** | 0.5332 | 0.3311 | 16.3 |
| ±0.25 | 0.5606 | **0.5353** | 0.3486 | 18.9 |
| ±0.30 | 0.5561 | 0.5335 | 0.3638 | 21.9 |
| ±0.40 | 0.5478 | 0.5297 | 0.3909 | 27.5 |
| ±0.50 | 0.5406 | 0.5259 | 0.4413 | 37.2 |
| ±0.60 | 0.5272 | 0.5099 | 0.4649 | 44.3 |
