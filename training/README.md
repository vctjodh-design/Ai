# Fine-tuning

This project uses a small LoRA adapter so the original pretrained weights stay
unchanged. The curated set has 540 training and 60 validation examples from
openly licensed general Q&A and multi-step math sources; see `data/README.md`
for source licenses and preparation details.

## Train a local CPU pilot

```sh
uv run python scripts/train_lora.py
```

The default run uses 64 examples to keep a first CPU run bounded. Training on
all examples takes longer, but uses the complete curated training split and
held-out validation split:

```sh
uv run python scripts/train_lora.py \
  --max-train-examples 0 \
  --eval-examples 0 \
  --max-length 512 \
  --output-dir models/qwen2.5-0.5b-reasoning-adapter-full
```

The full adapter is saved separately so the existing 64-example pilot remains
available. Training on this CPU-only machine is much slower than on a supported
GPU. By default, the training script evaluates every validation example that
fits the selected token limit and reports token-weighted validation loss.
Examples longer than the token limit are skipped; the 512-token run includes
523 of 540 training rows and all 60 validation rows.

## Compare the base model and adapter

Run the held-out comparison after training:

```sh
uv run python scripts/evaluate_model.py
```

This reports teacher-forced token loss and perplexity, broken down by dataset
source, and greedy exact-match accuracy on all 30 held-out GSM8K questions. It
compares the base model with the adapter and writes an auditable JSON report to
`models/qwen2.5-0.5b-reasoning-adapter-full/evaluation_metrics.json`.
Generation exact match is a narrow score for these grade-school math answers;
it does not measure general reasoning ability or open-ended answer quality.
The evaluator's answer-normalization and token-weighted-loss helpers can be
checked with `uv run python scripts/test_evaluation.py`.

After training, ask a question with:

```sh
uv run python scripts/ask_model.py "Why does dividing by a fraction increase a number?"
```

This setup teaches answer formatting and selected multi-step solution
patterns. A 0.5B model and a small fine-tuning set cannot guarantee reliable
or general-purpose deep reasoning; evaluate the adapter on held-out tasks
before relying on it.
