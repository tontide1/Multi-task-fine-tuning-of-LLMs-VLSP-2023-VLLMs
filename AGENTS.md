---
description: 
alwaysApply: true
---

# AGENTS.md

## Core Working Rules (STRICT AND DO NOT DELETE OR REMOTE IT)

- **Minimal Impact:** Make the smallest correct change. NEVER refactor unrelated code, formatting, or docs.
- **No Guessing:** If requirements, contracts, or scope are ambiguous, stop and ask.
- **Verify First:** Prove your changes work via direct reads, checks, or tests before finishing.
- **Keep It Simple:** Do not over-engineer. Write straightforward solutions.

## Environment (source of truth)

- `conda activate nlp` (Python 3.11)

## High-signal commands

- Parser tests: `python -m unittest scripts.test_baseline_parsers -v`
- Seed ETL (MCQ):
  - `python scripts/load_exams_mcq_seed.py`
  - `python scripts/recheck_exams_mcq_seed.py`
  - `python scripts/load_wiki_mcq_seed.py`
  - `python scripts/recheck_wiki_mcq_seed.py`
- Seed ETL (comprehension raw + short answer train):
  - `python scripts/load_comprehension_seed_raw.py`
  - `python scripts/recheck_comprehension_seed_raw.py`
  - `python scripts/filter_comprehension_raw_uit.py`
  - `python scripts/load_comprehension_short_answer_seed.py`

## Current status

- `comprehension_seed_raw` spec: `docs/superpowers/specs/2026-04-29-comprehension-seed-raw-design.md`
- `comprehension_seed_raw` plan: `docs/superpowers/plans/2026-04-29-comprehension-seed-raw.md`
- Train đọc hiểu dùng **`comprehension_short_answer`**: `scripts/filter_comprehension_raw_uit.py`, `scripts/load_comprehension_short_answer_seed.py` → `seed_exports/comprehension_short_answer_seed.jsonl`
- Lý do định hướng: hiện không có API LLM/distractor generator đáng tin cậy để chuyển short answer thành MCQ sạch; **Better clean extractive QA than noisy generated MCQ.**
- Pipeline **comprehension MCQ** (spec/plan ngày 2026-04-30) không còn là đường train mặc định; script vẫn trong repo để tham chiếu lịch sử
