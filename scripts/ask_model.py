#!/usr/bin/env python3
"""Ask a question using the base model and its fine-tuned LoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b"
DEFAULT_ADAPTER_DIR = ROOT / "models" / "qwen2.5-0.5b-reasoning-adapter-full"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer accurately. For multi-step problems, "
    "explain the key steps briefly and state the final answer. Be clear when "
    "information is uncertain."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="Question to ask.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=DEFAULT_ADAPTER_DIR,
        help="Directory containing the trained LoRA adapter.",
    )
    args = parser.parse_args()

    if not (args.adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"The LoRA adapter is missing: {args.adapter_dir}. "
            "Run scripts/train_lora.py first."
        )
    question = " ".join(args.question).strip()
    if not question:
        question = input("Question: ").strip()
    if not question:
        raise ValueError("Enter a question.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_dir,
        local_files_only=True,
        use_fast=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = getattr(prompt, "input_ids", None)
    if input_ids is None:
        input_ids = prompt["input_ids"] if isinstance(prompt, dict) else prompt
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(
        output[0, input_ids.shape[1] :],
        skip_special_tokens=True,
    ).strip()
    print(answer)


if __name__ == "__main__":
    main()
