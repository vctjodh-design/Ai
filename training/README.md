# Fine-tuning

This project uses a small LoRA adapter so the original pretrained weights stay
unchanged. The curated set has 540 training and 60 validation examples from
openly licensed general Q&A and multi-step math sources; see `data/README.md`
for source licenses and preparation details.

## Train a local CPU pilot

```sh
uv run python scripts/train_lora.py
```

The default run uses 64 examples to keep a first CPU run bounded. To train on
all examples, use `--max-train-examples 0`. Training on this CPU-only machine
will be much slower than on a supported GPU.

The adapter and its loss report are saved in
`models/qwen2.5-0.5b-reasoning-adapter/`. After training, ask a question with:

```sh
uv run python scripts/ask_model.py "Why does dividing by a fraction increase a number?"
```

This setup teaches answer formatting and selected multi-step solution
patterns. A 0.5B model and a small fine-tuning set cannot guarantee reliable
or general-purpose deep reasoning; evaluate the adapter on held-out tasks
before relying on it.
