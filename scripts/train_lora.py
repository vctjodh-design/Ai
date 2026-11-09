#!/usr/bin/env python3
"""CPU-friendly LoRA fine-tuning for the local Qwen2.5 base checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b"
DEFAULT_TRAIN_FILE = ROOT / "data" / "train.jsonl"
DEFAULT_VALIDATION_FILE = ROOT / "data" / "validation.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "qwen2.5-0.5b-reasoning-adapter"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer accurately. For multi-step problems, "
    "explain the key steps briefly and state the final answer. Be clear when "
    "information is uncertain."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=64,
        help="CPU pilot limit; use 0 to train on the complete training file.",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--eval-examples",
        type=int,
        default=0,
        help="Maximum validation examples; use 0 to evaluate the complete split.",
    )
    parser.add_argument("--checkpoint-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def get_token_ids(encoded: Any) -> list[int]:
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Expected one tokenized conversation at a time.")
        encoded = encoded[0]
    return encoded


def encode_example(
    tokenizer: Any, record: dict[str, Any], max_length: int
) -> dict[str, list[int]] | None:
    messages = record["messages"]
    if len(messages) < 3 or messages[-1].get("role") != "assistant":
        return None

    prompt_ids = get_token_ids(
        tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = get_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "The rendered prompt is not a prefix of the full conversation; "
            "refusing to train on incorrectly masked labels."
        )
    if len(full_ids) > max_length or len(full_ids) <= len(prompt_ids):
        return None

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "labels": labels}


def make_batch(
    example: dict[str, list[int]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([example["input_ids"]], dtype=torch.long)
    labels = torch.tensor([example["labels"]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def evaluate(
    model: Any,
    examples: list[dict[str, list[int]]],
    pad_token_id: int,
) -> float | None:
    if not examples:
        return None
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for example in examples:
            batch = make_batch(example, pad_token_id)
            target_tokens = int((batch["labels"] != -100).sum().item())
            total_loss += float(model(**batch).loss.item()) * target_tokens
            total_tokens += target_tokens
    model.train()
    return total_loss / total_tokens if total_tokens else None


def main() -> None:
    args = parse_args()
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {args.model_dir}")
    if args.epochs < 1 or args.max_length < 32:
        raise ValueError("--epochs must be positive and --max-length at least 32.")
    if (
        args.max_train_examples < 0
        or args.max_steps < 0
        or args.eval_examples < 0
        or args.checkpoint_steps < 1
    ):
        raise ValueError("Limits must be nonnegative; checkpoint steps must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    try:
        torch.set_num_interop_threads(min(2, max(1, args.cpu_threads)))
    except RuntimeError:
        pass

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_train = load_jsonl(args.train_file)
    raw_validation = load_jsonl(args.validation_file)
    train_examples = [
        encoded
        for row in raw_train
        if (encoded := encode_example(tokenizer, row, args.max_length)) is not None
    ]
    validation_examples = [
        encoded
        for row in raw_validation
        if (encoded := encode_example(tokenizer, row, args.max_length)) is not None
    ]
    if not train_examples:
        raise ValueError(
            "No training examples fit --max-length. Increase it or shorten the data."
        )

    rng = random.Random(args.seed)
    rng.shuffle(train_examples)
    if args.max_train_examples:
        train_examples = train_examples[: args.max_train_examples]
    if args.eval_examples:
        validation_examples = validation_examples[: args.eval_examples]

    print(
        f"Training on CPU with {len(train_examples)} examples; "
        f"{len(validation_examples)} validation examples fit the token limit."
    )
    print(f"Max sequence length: {args.max_length}; epochs: {args.epochs}.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        ),
    )
    model.print_trainable_parameters()
    model.train()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pad_token_id = int(tokenizer.pad_token_id)
    started = time.monotonic()
    losses = []
    step = 0
    stop_training = False

    for epoch in range(args.epochs):
        rng.shuffle(train_examples)
        for example in train_examples:
            optimizer.zero_grad(set_to_none=True)
            batch = make_batch(example, pad_token_id)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()

            step += 1
            losses.append(float(loss.detach().item()))
            if step == 1 or step % 10 == 0:
                print(f"step={step} loss={losses[-1]:.4f}")
            if step % args.checkpoint_steps == 0:
                model.save_pretrained(args.output_dir, safe_serialization=True)
                tokenizer.save_pretrained(args.output_dir)
                print(f"Saved adapter checkpoint at step {step}")
            if args.max_steps and step >= args.max_steps:
                stop_training = True
                break
        if stop_training:
            break

    validation_loss = evaluate(model, validation_examples, pad_token_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    metrics = {
        "base_model": "Qwen/Qwen2.5-0.5B",
        "training_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "steps": step,
        "epochs_requested": args.epochs,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "train_loss_mean": sum(losses) / len(losses),
        "validation_loss_mean": validation_loss,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "device": "cpu",
        "adapter": "LoRA r=8, alpha=16; q_proj and v_proj",
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
