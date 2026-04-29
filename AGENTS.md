---
description: 
alwaysApply: true
---

# AGENTS.md

## Environment (source of truth)

- `conda activate nlp` (Python 3.11)

## High-signal commands

- Parser tests: `python -m unittest scripts.test_baseline_parsers -v`
- Seed ETL (MCQ):
  - `python scripts/load_exams_mcq_seed.py`
  - `python scripts/recheck_exams_mcq_seed.py`
  - `python scripts/load_wiki_mcq_seed.py`
  - `python scripts/recheck_wiki_mcq_seed.py`

## Current status

- `comprehension_seed_raw` spec: `docs/superpowers/specs/2026-04-29-comprehension-seed-raw-design.md`
- `comprehension_seed_raw` plan: `docs/superpowers/plans/2026-04-29-comprehension-seed-raw.md`
- ETL script not implemented yet (seed file not built)
