#!/usr/bin/env python3
"""Build a small, licensed Q&A and multi-step math SFT dataset."""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data"
SEED = 23
SAMPLES_PER_SOURCE = 300
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer accurately. For multi-step problems, "
    "explain the key steps briefly and state the final answer. Be clear when "
    "information is uncertain."
)

SOURCES = {
    "oasst": {
        "path": RAW_DIR / "oasst1-train.parquet",
        "url": (
            "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/"
            "fdf72ae0827c1cda404aff25b6603abec9e3399b/"
            "data/train-00000-of-00001-b42a775f407cee45.parquet"
        ),
        "sha256": "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e",
    },
    "gsm8k": {
        "path": RAW_DIR / "gsm8k-train.parquet",
        "url": (
            "https://huggingface.co/datasets/openai/gsm8k/resolve/"
            "740312add88f781978c0658806c59bc2815b9866/"
            "main/train-00000-of-00001.parquet"
        ),
        "sha256": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
    },
}

CALCULATION = re.compile(
    r"<<([^=<>]+)=([^<>]+)>>\s*\$?[\d,]+(?:\.\d+)?"
)
FINAL_ANSWER = re.compile(r"\s*####\s*")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(name: str, source: dict[str, Any]) -> Path:
    path = source["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256(path) == source["sha256"]:
        return path

    partial = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(
        source["url"], headers={"User-Agent": "llm-fine-tuning-starter/1.0"}
    )
    print(f"Downloading pinned {name} training data...")
    with urllib.request.urlopen(request, timeout=60) as response:
        with partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)

    actual = sha256(partial)
    if actual != source["sha256"]:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"{name} checksum mismatch: expected {source['sha256']}, got {actual}"
        )
    partial.replace(path)
    return path


def label_scores(row: dict[str, Any]) -> dict[str, float]:
    labels = row.get("labels") or {}
    return dict(zip(labels.get("name") or [], labels.get("value") or []))


def is_reviewed(row: dict[str, Any]) -> bool:
    return (
        row.get("review_result") is True
        and not row.get("deleted", False)
        and row.get("review_count", 0) >= 1
    )


def oasst_pairs(path: Path) -> list[dict[str, Any]]:
    columns = [
        "message_id",
        "parent_id",
        "text",
        "role",
        "lang",
        "review_count",
        "review_result",
        "deleted",
        "rank",
        "synthetic",
        "detoxify",
        "labels",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    by_id = {row["message_id"]: row for row in rows}
    samples = []

    for reply in rows:
        if (
            reply["role"] != "assistant"
            or reply["lang"] != "en"
            or reply.get("rank") != 0
            or reply.get("synthetic")
            or not is_reviewed(reply)
        ):
            continue

        toxicity = reply.get("detoxify") or {}
        if max(
            toxicity.get("toxicity", 0),
            toxicity.get("severe_toxicity", 0),
            toxicity.get("sexual_explicit", 0),
        ) >= 0.2:
            continue

        scores = label_scores(reply)
        if scores.get("quality", 1) < 0.6 or any(
            scores.get(label, 0) > 0
            for label in (
                "spam",
                "lang_mismatch",
                "pii",
                "not_appropriate",
                "hate_speech",
                "sexual_content",
            )
        ):
            continue

        prompt = by_id.get(reply.get("parent_id"))
        if (
            not prompt
            or prompt["role"] != "prompter"
            or prompt["lang"] != "en"
            or prompt.get("synthetic")
            or not is_reviewed(prompt)
        ):
            continue

        question = (prompt.get("text") or "").strip()
        answer = (reply.get("text") or "").strip()
        if not (20 <= len(question) <= 1800 and 80 <= len(answer) <= 2600):
            continue

        samples.append(
            {
                "source": "OpenAssistant/oasst1",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{question}\n\nPlease answer clearly and explain "
                            "the key steps when useful."
                        ),
                    },
                    {"role": "assistant", "content": answer},
                ],
            }
        )

    return samples


def gsm8k_pairs(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    samples = []
    for row in rows:
        question = (row.get("question") or "").strip()
        raw_answer = (row.get("answer") or "").strip()
        if not question or "####" not in raw_answer:
            continue

        steps, final = FINAL_ANSWER.split(raw_answer, maxsplit=1)
        steps = CALCULATION.sub(
            lambda match: f"{match.group(2).strip()}",
            steps,
        )
        steps = re.sub(r"[ \t]+", " ", steps).strip()
        final = final.strip()
        if not final:
            continue

        answer = f"{steps}\n\nFinal answer: {final}"
        samples.append(
            {
                "source": "openai/gsm8k",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{question}\n\nSolve it step by step. Keep the "
                            "explanation concise and state the final answer."
                        ),
                    },
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    return samples


def select_and_split(
    source_name: str,
    samples: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(samples) < SAMPLES_PER_SOURCE:
        raise ValueError(
            f"Only {len(samples)} usable {source_name} examples were found; "
            f"expected at least {SAMPLES_PER_SOURCE}."
        )
    rng.shuffle(samples)
    selected = samples[:SAMPLES_PER_SOURCE]
    validation_count = max(1, round(len(selected) * 0.1))
    validation = selected[:validation_count]
    training = selected[validation_count:]
    return training, validation


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for sample in samples:
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    downloaded = {
        name: download_source(name, source) for name, source in SOURCES.items()
    }
    rng = random.Random(SEED)

    source_samples = {
        "OpenAssistant": oasst_pairs(downloaded["oasst"]),
        "GSM8K": gsm8k_pairs(downloaded["gsm8k"]),
    }
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for name, samples in source_samples.items():
        train, validation = select_and_split(name, samples, rng)
        train_rows.extend(train)
        validation_rows.extend(validation)

    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)
    write_jsonl(ROOT / "data" / "train.jsonl", train_rows)
    write_jsonl(ROOT / "data" / "validation.jsonl", validation_rows)
    print(
        f"Created {len(train_rows)} training and {len(validation_rows)} "
        "validation examples."
    )
    for name, samples in source_samples.items():
        print(f"  {name}: selected {SAMPLES_PER_SOURCE}")


if __name__ == "__main__":
    main()
