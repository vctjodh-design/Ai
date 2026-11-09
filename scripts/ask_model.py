#!/usr/bin/env python3
"""Ask a question using the base model and its fine-tuned LoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b"
DEFAULT_ADAPTER_DIR = ROOT / "models" / "qwen2.5-0.5b-reasoning-adapter-full"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Give accurate, grounded answers. For multi-step "
    "problems, explain the key steps briefly, state important assumptions, and "
    "give a direct final answer. Do not present hidden chain-of-thought. If you "
    "do not know or the evidence is uncertain, say so rather than inventing details."
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
    parser.add_argument(
        "--messages-stdin",
        action="store_true",
        help="Read a JSON object with conversation messages from standard input.",
    )
    args = parser.parse_args()

    if not (args.adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"The LoRA adapter is missing: {args.adapter_dir}. "
            "Run scripts/train_lora.py first."
        )

    if args.messages_stdin:
        try:
            payload = json.load(sys.stdin)
        except json.JSONDecodeError as exc:
            raise ValueError("Expected a JSON object on standard input.") from exc
        raw_messages = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("The JSON input must contain a non-empty messages list.")

        messages = []
        for message in raw_messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be an object.")
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise ValueError("Messages need a user or assistant role and text content.")
            if not content.strip():
                raise ValueError("Message content cannot be empty.")
            messages.append({"role": role, "content": content})
        system_prompt = payload.get("systemPrompt", "") if isinstance(payload, dict) else ""
        if not isinstance(system_prompt, str) or len(system_prompt) > 12_000:
            raise ValueError("The optional system prompt must be a string under 12,000 characters.")
        messages.insert(
            0,
            {
                "role": "system",
                "content": "\n\n".join(part for part in (SYSTEM_PROMPT, system_prompt.strip()) if part),
            },
        )
    else:
        question = " ".join(args.question).strip()
        if not question:
            question = input("Question: ").strip()
        if not question:
            raise ValueError("Enter a question.")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

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
            repetition_penalty=1.12,
            no_repeat_ngram_size=3,
            renormalize_logits=True,
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
