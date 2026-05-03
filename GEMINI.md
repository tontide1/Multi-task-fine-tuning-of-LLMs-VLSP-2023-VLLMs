# GEMINI.md

## Core Working Rules (STRICT)

- **Minimal Impact:** Make the smallest correct change. NEVER refactor unrelated code, formatting, or docs.
- **No Guessing:** If requirements, contracts, or scope are ambiguous, stop and ask.
- **Verify First:** Prove your changes work via direct reads, checks, or tests before finishing.
- **Keep It Simple:** Do not over-engineer. Write straightforward solutions.

## Environment

- Use conda for primary
- "conda activate nlp" is a command to activate environment named "nlp"
- python version is 3.12

## 5 Seed Files cho Train/Val

| Seed File | Size | Mô tả |
|-----------|------|-------|
| `comprehension_short_answer_seed.jsonl` | 35M | Short answer đọc hiểu (UIT) |
| `exams_mcq_seed.jsonl` | 7.2M | MCQ từ đề thi |
| `wiki_mcq_seed_final.jsonl` | 7.1M | MCQ từ Wikipedia |
| `cloze_lm_retention_seed.jsonl` | 8.2M | Cloze task (language modeling) |
| `instruction_retention_seed.jsonl` | 25M | Instruction retention |

Location: `seed_exports/`

## Training Split Pipeline

- Script: `scripts/build_training_split.py`
- Output: `seed_exports/splits/`
- Status: Completed (deterministic 90/5/5 split for benchmarks, 95/5 for retention)
