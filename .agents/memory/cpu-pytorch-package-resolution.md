---
name: CPU-only PyTorch package resolution
description: Replit Python package-index behavior when installing CPU PyTorch and Hugging Face libraries.
---

When CPU PyTorch is added through Replit's Python package manager, it can generate `tool.uv.sources` mappings that route unrelated packages such as Transformers and Accelerate to the PyTorch CPU index. Those packages are unavailable there, so `uv` reports no compatible Linux versions. Keep only Torch mapped to the CPU index, use an array source record with a Linux marker, and mark the CPU Torch dependency as Linux-only; leave other Hugging Face dependencies on the default PyPI index.

**Why:** the generated source mappings made an otherwise valid training dependency set unsatisfiable; restricting the index mapping allowed installation.

**How to apply:** after setting up CPU training dependencies in this workspace, inspect `pyproject.toml` for unintended PyTorch-index mappings before resolving packages.
