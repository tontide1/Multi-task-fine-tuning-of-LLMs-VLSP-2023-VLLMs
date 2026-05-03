# Build Training Splits Design

**Goal:** Combine 5 seed datasets into clean, deterministic splits (`train`, `val`, `shadow_eval`, and `probes`) for fine-tuning, using fixed percentages and a monolithic script approach without VLSP 2023 leakage checks.

**Architecture:** A single monolithic Python script (`scripts/build_training_split.py`) will load all seed files in-memory, deduplicate them based on prompt/response hashes, shuffle them deterministically, and split them by fixed percentage ratios into separate output JSONL files.

**Tech Stack:** Python 3.11, standard library (`json`, `random`, `pathlib`, `hashlib`).

---

## 1. Input Sources

The script will ingest the following 5 seed files from `seed_exports/`:
- `comprehension_short_answer_seed.jsonl`
- `exams_mcq_seed.jsonl`
- `wiki_mcq_seed_final.jsonl`
- `instruction_retention_seed.jsonl`
- `cloze_lm_retention_seed.jsonl`

## 2. Data Flow & Deduplication

### In-Memory Loading & Hashing
- Read all JSONL files into a single list of dictionaries.
- Compute an MD5/SHA256 hash for every record based on a concatenation of `user` content and `assistant` content.
- Deduplication: Iterate through the loaded data and maintain a `seen_hashes` set. Keep only the first occurrence of any record to prevent the same question from appearing in both training and evaluation splits due to cross-source overlaps.

### Re-bucketing
Group the clean, deduplicated records back into 5 buckets based on the value of `metadata["task"]`.

## 3. Deterministic Splitting Strategy

- Set `random.seed(42)` at the beginning of the script to ensure reproducible shuffling.
- Shuffle each bucket independently.
- Define split configurations as percentages (e.g., `TRAIN_PCT = 0.90`, `VAL_PCT = 0.05`, `SHADOW_PCT = 0.05`).

### Splitting Logic by Task Type
**Group A: Benchmark Proxies (`exams_mcq`, `wiki_mcq`, `comprehension_short_answer`)**
- Calculate integer indices based on the length of the bucket and the percentages.
- Slicing:
  - `train_slice = bucket[:train_idx]`
  - `val_slice = bucket[train_idx:val_idx]`
  - `shadow_slice = bucket[val_idx:]`

**Group B: Retention Tasks (`instruction_retention`, `cloze_lm_retention`)**
- Since these are for retention tracking, they are split into `train` and a `probe` set instead of val/shadow.
- e.g., 95% to `train`, 5% to their respective `probe`.

## 4. Mixing and Export

### Task-Balanced Mix (Optional Limiting)
- The script should support a simple mechanism to optionally cap the number of items any single bucket contributes to the final `train_all.jsonl` (e.g., capping short answer at 10,000 if needed). By default, it will perform a "natural mix" (use all available train records).

### Output Files
Write the final lists into the `seed_exports/splits/` directory (creating it if it doesn't exist):
1. `train_all.jsonl` (Combined `train` slices from all 5 buckets)
2. `val_all.jsonl` (Combined `val` slices from Group A)
3. `shadow_eval.jsonl` (Combined `shadow` slices from Group A)
4. `instruction_probe.jsonl` (Probe slice from instruction retention)
5. `cloze_probe.jsonl` (Probe slice from cloze retention)

### Reporting
- The script must print a summary report to the console detailing the counts: Total loaded -> Total after deduplication -> Counts per split per task.
- Export this summary to `seed_exports/splits/split_report.json`.
