# Qwen2.5-0.5B reasoning adapter — CPU pilot

This LoRA adapter is an early CPU pilot for concise general Q&A and worked
grade-school math answers. It was trained for one epoch on 64 examples from
the curated dataset. It is kept separate from the full-data adapter.

## Training

- Base checkpoint: local Qwen2.5-0.5B
- Method: LoRA, rank 8, alpha 16, applied to `q_proj` and `v_proj`
- Training examples: 64
- Validation examples: 8
- Sequence limit: 256 tokens
- Learning rate: 0.0001
- Device: CPU, float32

The recorded mean training loss is 1.481 and mean validation loss is 1.538.
These small pilot metrics are not a broad capability or quality evaluation.
See `training_metrics.json` for the complete record and `data/README.md` for
dataset provenance and licenses.

## Use

```sh
uv run python scripts/ask_model.py \
  --adapter-dir models/qwen2.5-0.5b-reasoning-adapter \
  "Why does dividing by a fraction increase a number?"
```

## Limitations

The model has only 0.5B parameters and was fine-tuned on a small dataset. It can
produce incorrect or biased responses and should not be used for high-stakes
decisions or assumed to have reliable general reasoning ability. Check its
answers independently.
