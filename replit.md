# Qwen Fine-Tuning Lab

A local Qwen LoRA training and evaluation project for concise general Q&A and
grade-school math explanations.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string
- `uv run python scripts/train_lora.py --max-train-examples 0 --eval-examples 0 --max-length 512 --output-dir models/qwen2.5-0.5b-reasoning-adapter-full` — train on all examples that fit the token limit
- `uv run python scripts/evaluate_model.py` — compare the base model and full adapter on held-out validation data
- `uv run python scripts/ask_model.py "Your question"` — ask the trained full adapter

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Python 3.13, uv, CPU PyTorch, Transformers, and PEFT
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `data/` — curated training and validation examples; see `data/README.md` for provenance and licenses
- `scripts/prepare-training-data.py` — reproducible, checksum-verified dataset preparation
- `scripts/train_lora.py` — local LoRA fine-tuning with validation loss reporting
- `scripts/evaluate_model.py` — base-versus-adapter validation and held-out math exact match
- `scripts/ask_model.py` — command-line inference with the trained adapter
- `models/` — local base checkpoint and saved LoRA adapters

## Architecture decisions

- Use LoRA so training updates a small adapter instead of rewriting the pretrained base model.
- Keep data preparation, training, evaluation, and inference runnable locally without a hosted model API.
- Keep the original 64-example pilot separate from the full-data adapter.
- Treat held-out loss and math exact match as narrow indicators, not proof of general reasoning ability.

## Product

This project prepares a small licensed dataset, fine-tunes a local Qwen 0.5B
checkpoint, compares the adapter with the base model, and provides a simple
command-line prompt flow. The model is experimental and should not be relied on
for high-stakes or general-purpose reasoning.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

_Populate as you build — sharp edges, "always run X before Y" rules._

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
