# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
conda activate nlp
```

- **Parser unit tests:** `python -m unittest scripts.test_baseline_parsers -v`
- **Training split pipeline:** `python scripts/build_training_split.py`
- **Seed ETL (exams_mcq):** `scripts/load_exams_mcq_seed.py` + `scripts/recheck_exams_mcq_seed.py`
- **Seed ETL (wiki_mcq):** `scripts/load_wiki_mcq_seed.py` + `scripts/recheck_wiki_mcq_seed.py`
- **Seed ETL (comprehension raw):** `scripts/load_comprehension_seed_raw.py` + `scripts/recheck_comprehension_seed_raw.py`
- **Comprehension short answer:** `scripts/filter_comprehension_raw_uit.py` → `scripts/load_comprehension_short_answer_seed.py`
- **Full fine-tune:** run `notebooks/fine-tune-qwen2-5-unsloth.ipynb` on Kaggle (T4 GPU)
- **Baseline eval:** `notebooks/baseline-vlsp-2023.ipynb` on Colab (T4 GPU)

## Project Overview

Multi-task QLoRA fine-tuning of `unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit` on the VLSP 2023 VLLMs benchmark (Vietnamese). 4 benchmark tasks: `lambada_vi` (next-word), `wikipediaqa_vi` (5-shot MCQ), `exams_vi` (7-subject MCQ), `comprehension_vi` (0-shot MCQ).

## Architecture

### Data Pipeline (5 seed files → training split)

All seed data uses unified `messages` + `metadata` container schema.

| Bucket | File | Role |
|--------|------|------|
| `exams_mcq` | `seed_exports/exams_mcq_seed.jsonl` | MCQ exam questions (11.9k) |
| `wiki_mcq` | `seed_exports/wiki_mcq_seed_final.jsonl` | MCQ general knowledge (7.9k) |
| `comprehension_short_answer` | `seed_exports/comprehension_short_answer_seed.jsonl` | Extractive QA from UIT-ViQuAD2.0 (21.7k) |
| `instruction_retention` | `seed_exports/instruction_retention_seed.jsonl` | Vietnamese instruct/chat retention (20k) |
| `cloze_lm_retention` | `seed_exports/cloze_lm_retention_seed.jsonl` | LM-style cloze retention (10k) |

Training split pipeline (`scripts/build_training_split.py`) produces: `train_all.jsonl`, `val_all.jsonl`, `shadow_eval.jsonl`, `instruction_probe.jsonl`, `cloze_probe.jsonl` — uploaded to HF dataset repo.

### Training Design

- **QLoRA** with Unsloth on Kaggle (primary) / Colab Free (smoke test)
- **Lora config:** r=16, alpha=32, dropout=0, targets = q/k/v/o/gate/up/down proj
- **Max seq length:** 2048, data packing enabled
- **Training args:** adamw_8bit, linear schedule, 2e-4 LR, 2 epochs
- **Loss:** assistant-only completion loss via `DataCollatorForCompletionOnlyLM`
- **Eval:** per-task loss on val set + generation metrics (accuracy for MCQ, EM/F1/RougeL for short answer) + retention probes (cloze perplexity, instruction loss)
- **Experiment tracking:** W&B; **Model registry:** Hugging Face Hub

### Training/Evaluation Split Strategy

- **val_all.jsonl:** per-task representative set, no overlap with train (grouped by passage/source hash for comprehension)
- **shadow_eval.jsonl:** harder subset, gate before official VLSP runs
- **instruction_probe.jsonl / cloze_probe.jsonl:** retention guardrails, not merged into main eval metric
- **Official VLSP 2023:** holdout only, run at major milestones only

### Reading Comprehension Training Approach

Train uses `comprehension_short_answer` (extractive QA from UIT-ViQuAD2.0), NOT MCQ. Official VLSP `comprehension` benchmark is MCQ — this format gap is tracked explicitly. Rationale: no reliable LLM API / distractor generator to convert short answer → clean MCQ. Better clean extractive QA than noisy generated MCQ.

### Script Categories

- **ETL scripts** (`scripts/load_*.py`): download from HF, transform to `messages`+`metadata` schema, export to `seed_exports/`
- **QC scripts** (`scripts/recheck_*.py`): validate exported seed files, produce JSON reports
- **Comprehension MCQ pipeline** (historical, not in active training flow): `scripts/build_comprehension_mcq_candidates.py`, `scripts/prepare_comprehension_mcq_generation.py`, etc.
- **Training split:** `scripts/build_training_split.py` + `scripts/test_build_training_split.py`
- **Fine-tune notebook:** `notebooks/fine-tune-qwen2-5-unsloth.ipynb` (Kaggle, run_mode = "smoke" → "full")

## Behavioral Guidelines

### 1. Think Before Coding

State assumptions explicitly. If multiple interpretations exist, present them — don't pick silently. If a simpler approach exists, say so. Push back when warranted. If something is unclear, stop and ask.

### 2. Simplicity First

Minimum code that solves the problem. No features beyond what was asked. No abstractions for single-use code. No error handling for impossible scenarios. If it's 200 lines and could be 50, rewrite it.

### 3. Surgical Changes

Touch only what you must. Don't improve adjacent code, comments, or formatting. Don't refactor things that aren't broken. Match existing style. When your changes create orphans (unused imports/variables/functions), remove them. Don't remove pre-existing dead code unless asked.

### 4. Goal-Driven Execution

Define success criteria before starting. For multi-step tasks, state a brief plan with verify checks per step. Loop until verification passes.

---

### Key Design Rules

- VLSP 2023 is holdout-only, never used in training
- No synthetic data generation without QC pipeline
- `comprehension_mcq` pipeline removed from training flow (scripts kept as history)
- Only 1 variable changed per improvement round
- Best checkpoint selected by eval loss on main tasks (exams_mcq, wiki_mcq, comprehension_short_answer)
