#!/usr/bin/env python3
"""Compare the base Qwen model and its LoRA adapter on held-out examples."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora import encode_example, load_jsonl, make_batch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b"
DEFAULT_ADAPTER_DIR = ROOT / "models" / "qwen2.5-0.5b-reasoning-adapter-full"
DEFAULT_VALIDATION_FILE = ROOT / "data" / "validation.jsonl"

NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
FRACTION_PATTERN = re.compile(
    r"(-?\d[\d,]*(?:\.\d+)?)\s*/\s*(\d[\d,]*(?:\.\d+)?)"
)
FINAL_ANSWER_PATTERN = re.compile(
    r"(?:final\s+answer|answer)\s*:\s*(.+)", re.IGNORECASE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--validation-file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--limit-math-examples",
        type=int,
        default=0,
        help="Maximum GSM8K generation examples; use 0 for all held-out math items.",
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def normalize_final_answer(text: str) -> str:
    """Normalize the answer portion of a response for GSM8K exact matching."""
    matches = list(FINAL_ANSWER_PATTERN.finditer(text))
    stripped = text.strip()
    candidate = (
        matches[-1].group(1)
        if matches
        else stripped.splitlines()[-1] if stripped else ""
    )
    if not candidate:
        return ""
    fraction = FRACTION_PATTERN.search(candidate)
    if fraction:
        try:
            numerator = Decimal(fraction.group(1).replace(",", ""))
            denominator = Decimal(fraction.group(2).replace(",", ""))
            if denominator:
                return str((numerator / denominator).normalize())
        except InvalidOperation:
            pass
    numeric = NUMBER_PATTERN.search(candidate)
    if numeric:
        try:
            return str(Decimal(numeric.group(0).replace(",", "")).normalize())
        except InvalidOperation:
            pass
    return re.sub(r"[^a-z0-9]+", " ", candidate.lower()).strip()


def encode_validation_rows(
    tokenizer: Any, rows: list[dict[str, Any]], max_length: int
) -> tuple[list[tuple[str, dict[str, list[int]]]], dict[str, int]]:
    encoded: list[tuple[str, dict[str, list[int]]]] = []
    skipped: dict[str, int] = defaultdict(int)
    for index, row in enumerate(rows, start=1):
        source = str(row.get("source", "unknown"))
        example = encode_example(tokenizer, row, max_length)
        if example is None:
            skipped[source] += 1
        else:
            encoded.append((source, example))
    return encoded, dict(skipped)


def score_language_model(
    model: Any,
    examples: list[tuple[str, dict[str, list[int]]]],
    pad_token_id: int,
) -> dict[str, Any]:
    totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"nll_sum": 0.0, "target_tokens": 0, "examples": 0}
    )
    model.eval()
    with torch.inference_mode():
        for source, example in examples:
            batch = make_batch(example, pad_token_id)
            target_tokens = int((batch["labels"] != -100).sum().item())
            if target_tokens == 0:
                continue
            loss = float(model(**batch).loss.item())
            totals[source]["nll_sum"] += loss * target_tokens
            totals[source]["target_tokens"] += target_tokens
            totals[source]["examples"] += 1

    def summarize(values: dict[str, float | int]) -> dict[str, Any]:
        token_count = int(values["target_tokens"])
        nll = float(values["nll_sum"]) / token_count if token_count else None
        try:
            perplexity = math.exp(nll) if nll is not None else None
        except OverflowError:
            perplexity = None
        return {
            "examples": int(values["examples"]),
            "target_tokens": token_count,
            "mean_token_nll": nll,
            "perplexity": perplexity,
        }

    all_totals: dict[str, float | int] = {
        "nll_sum": sum(float(value["nll_sum"]) for value in totals.values()),
        "target_tokens": sum(int(value["target_tokens"]) for value in totals.values()),
        "examples": sum(int(value["examples"]) for value in totals.values()),
    }
    return {
        **summarize(all_totals),
        "by_source": {
            source: summarize(values) for source, values in sorted(totals.items())
        },
    }


def generate_math_answers(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    results = []
    correct = 0
    model.eval()
    for index, row in enumerate(rows, start=1):
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        prompt_ids = getattr(prompt, "input_ids", None)
        if prompt_ids is None:
            prompt_ids = prompt["input_ids"] if isinstance(prompt, dict) else prompt
        with torch.inference_mode():
            output = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prediction_text = tokenizer.decode(
            output[0, prompt_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        expected_text = messages[-1]["content"]
        expected = normalize_final_answer(expected_text)
        predicted = normalize_final_answer(prediction_text)
        is_correct = bool(expected) and expected == predicted
        correct += int(is_correct)
        results.append(
            {
                "question": messages[-2]["content"],
                "expected_final_answer": expected,
                "predicted_final_answer": predicted,
                "generated_text": prediction_text,
                "correct": is_correct,
            }
        )
        if index == 1 or index % 5 == 0 or index == len(rows):
            print(f"Generated {index}/{len(rows)} held-out math answers.", flush=True)
    return {
        "examples": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else None,
        "items": results,
    }


def main() -> None:
    args = parse_args()
    if args.max_length < 32 or args.max_new_tokens < 1 or args.cpu_threads < 1:
        raise ValueError("Token limits and CPU thread count must be positive.")
    if args.limit_math_examples < 0:
        raise ValueError("--limit-math-examples must be nonnegative.")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {args.model_dir}")
    if not (args.adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"LoRA adapter not found: {args.adapter_dir}")

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(min(2, args.cpu_threads))
    except RuntimeError:
        pass

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_rows = load_jsonl(args.validation_file)
    validation_examples, skipped = encode_validation_rows(
        tokenizer, raw_rows, args.max_length
    )
    math_rows = [
        row for row in raw_rows if row.get("source") == "openai/gsm8k"
    ]
    if args.limit_math_examples:
        math_rows = math_rows[: args.limit_math_examples]
    if not validation_examples:
        raise ValueError("No validation examples fit the selected token limit.")
    if not math_rows:
        raise ValueError("No held-out GSM8K examples are available.")

    print(
        f"Scoring {len(validation_examples)} validation examples "
        f"({sum(skipped.values())} too long for --max-length)."
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    pad_token_id = int(tokenizer.pad_token_id)

    print("Evaluating base model...")
    base_loss = score_language_model(base_model, validation_examples, pad_token_id)
    base_math = generate_math_answers(
        base_model, tokenizer, math_rows, args.max_new_tokens
    )

    print("Evaluating LoRA adapter...")
    adapter_model = PeftModel.from_pretrained(
        base_model, args.adapter_dir, is_trainable=False
    )
    adapted_loss = score_language_model(
        adapter_model, validation_examples, pad_token_id
    )
    adapted_math = generate_math_answers(
        adapter_model, tokenizer, math_rows, args.max_new_tokens
    )

    report = {
        "base_model": str(args.model_dir),
        "adapter": str(args.adapter_dir),
        "validation_file": str(args.validation_file),
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "validation_rows": len(raw_rows),
        "validation_examples_scored": len(validation_examples),
        "validation_examples_skipped_by_source": skipped,
        "held_out_math_examples_generated": len(math_rows),
        "base_model_metrics": {
            "teacher_forced": base_loss,
            "gsm8k_greedy_exact_match": base_math,
        },
        "adapter_metrics": {
            "teacher_forced": adapted_loss,
            "gsm8k_greedy_exact_match": adapted_math,
        },
    }
    output_path = args.output or args.adapter_dir / "evaluation_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved evaluation report to {output_path}")


if __name__ == "__main__":
    main()
