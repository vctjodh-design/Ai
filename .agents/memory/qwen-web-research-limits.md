---
name: Qwen web research limits
description: The evidence and citation limits of Qwen Chat's integration-free research mode.
---

Research mode reads bounded excerpts from public HTTPS pages when accessible and otherwise falls back to a clearly labeled search summary. The user chose to keep local Qwen and avoid a billable remote model. Research answers should therefore quote relevant source passages with citations rather than rely on Qwen to paraphrase evidence it may distort. Keep the limitations visible; do not claim a page was read when only its search summary was available.

Local generation is fail-closed: reject prompt echoes, runaway repetition, and long off-topic answers rather than displaying a cleaned fragment as if it were reliable. Simple greetings and match-score predictions receive direct responses instead of being sent to the weak local model.

**Why:** The user declined the search integration and selected local Qwen over a stronger model that would use Replit credits. Source-based excerpts avoid inventing a paraphrase when the local model produces malformed answers.

**How to apply:** Keep local Qwen available, fail explicitly on malformed output, display whether each source is a page excerpt or search summary, and do not add a billable remote model without asking first.
