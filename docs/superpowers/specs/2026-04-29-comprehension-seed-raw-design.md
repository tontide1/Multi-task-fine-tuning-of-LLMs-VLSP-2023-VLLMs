# Design: comprehension_seed_raw.jsonl

## Goal
Build `seed_exports/comprehension_seed_raw.jsonl` from two public Hugging Face sources while preserving extractive QA fidelity and enabling later MCQ expansion.

Sources:
- `taidng/UIT-ViQuAD2.0` (primary)
- `ShynBui/Vietnamese_Reading_Comprehension_Dataset` (supplementary)

## Output Contract

### Primary output
`seed_exports/comprehension_seed_raw.jsonl`

Each record uses the internal `messages` container. This file remains extractive QA (not MCQ).

```json
{
  "messages": [
    {"role": "user", "content": "Đoạn văn: ...\n\nCâu hỏi: ..."},
    {"role": "assistant", "content": "...answer text..."}
  ],
  "context": "...",
  "question": "...",
  "answer_text": "...",
  "metadata": {
    "task": "comprehension_raw",
    "source": "public",
    "source_dataset": "...",
    "source_split": "...",
    "source_id": "...",
    "answer_start": 456,
    "original_answer_start": 471,
    "answer_text": "...",
    "answer_variants": [
      {"text": "...", "answer_start": 123},
      {"text": "...", "answer_start": 125}
    ],
    "title": "...",
    "language": "vi",
    "difficulty": "medium",
    "span_check_mode": "raw_exact_then_normalized_find",
    "dedup_hash": "...",
    "qc_version": "comprehension_raw_source_specific_span_v1"
  }
}
```

Notes:
- `answer_variants` is only populated for UIT-ViQuAD2.0. For ShynBui, omit or set to an empty list.
- `original_answer_start` is stored only when needed (ShynBui normalization). For UIT-ViQuAD2.0, it matches `answer_start`.
- `assistant.content`, top-level `answer_text`, and `metadata.answer_text` must match.
- `messages[0].content` must equal `"Đoạn văn: {context}\n\nCâu hỏi: {question}"`.

### Rejects output
`seed_exports/comprehension_seed_raw_rejects.jsonl`

Reject record keeps debug fields:
```json
{
  "source_dataset": "...",
  "source_split": "...",
  "source_id": "...",
  "reason": "span_mismatch",
  "context_preview": "...",
  "question": "...",
  "answer": "...",
  "answer_start": 123
}
```

### Report output
`seed_exports/comprehension_seed_raw_report.json`

```json
{
  "total_loaded": 0,
  "total_kept": 0,
  "total_rejected": 0,
  "kept_by_dataset": {},
  "kept_by_split": {},
  "reject_reasons": {},
  "duplicate_count": 0,
  "split_policy": "raw_pool_all_source_splits_loaded; downstream split required",
  "qc_version": "comprehension_raw_source_specific_span_v1"
}
```

## Data Flow
1. Load all splits from both datasets (train/validation/test where available).
2. Normalize + validate source-specific schema into a shared intermediate:
   - `context`, `question`, `answer_text`, `answer_start`, `source_id`, `title`, `source_split`.
3. Apply source-specific QC (see below).
4. Build `messages` and `metadata`.
5. Deduplicate by a normalized hash of `(context, question, answer_text)`.
6. Export JSONL, rejects, and report.

This file is a raw pool. Downstream split/dedup by passage is required before training.

## Source-Specific QC

### UIT-ViQuAD2.0
- Drop `is_impossible=True`.
- Validate answer variants from `answers.text` + `answers.answer_start`:
  - `text` not empty
  - `answer_start` is int and >= 0
  - `answer_start + len(text) <= len(context)`
  - exact span match
- Keep only valid variants. If none pass, reject with `no_valid_answer_variant`.
- Primary answer = first valid variant.
- Store all valid variants in `metadata.answer_variants`.
- Strict span match on raw text:
  `context[answer_start:answer_start + len(answer_text)] == answer_text`
- If mismatch, reject with reason `span_mismatch`.
- `span_check_mode = "strict_exact"`.

### ShynBui/Vietnamese_Reading_Comprehension_Dataset
Rationale: the dataset may contain word segmentation with `_` which shifts offsets if normalized.

QC order:
1. Check strict span on raw text before normalization.
2. If pass, normalize `_` to spaces for `context`, `question`, `answer_text`.
3. Recompute `answer_start` using `.find(answer_text)` on normalized context.
4. If not found, reject with reason `answer_not_found_after_normalize`.
5. Store `original_answer_start` and `span_check_mode = "raw_exact_then_normalized_find"`.

## Length Filters
Apply lightweight length filters before QC:
- `min_context_chars = 80`
- `max_context_chars = 8000`
- `min_question_chars = 5`
- `max_question_chars = 400`
- `max_answer_chars = 300`

Reject reasons:
- `context_too_short`
- `context_too_long`
- `question_too_short`
- `question_too_long`
- `answer_too_long`

## Deduplication
Use SHA1 hash of normalized-light text:
- lowercase
- strip
- collapse whitespace
- replace `\u00a0`
- preserve punctuation

Hash input:
```
normalize_for_hash(context) + "\n" +
normalize_for_hash(question) + "\n" +
normalize_for_hash(answer_text)
```

## Error Handling
- If dataset load fails: print error and exit non-zero.
- If required columns missing: reject with reason `missing_field`.
- If answer text empty: reject with reason `missing_answer`.

## Reject Reasons (Enum)
- `is_impossible`
- `missing_field`
- `missing_context`
- `missing_question`
- `missing_answer`
- `invalid_answer_start`
- `answer_start_out_of_range`
- `span_mismatch`
- `no_valid_answer_variant`
- `answer_not_found_after_normalize`
- `context_too_short`
- `context_too_long`
- `question_too_short`
- `question_too_long`
- `answer_too_long`
- `duplicate`
- `load_error`
- `schema_error`

## Verification

Add `scripts/recheck_comprehension_seed_raw.py`:
- JSONL parse
- messages length = 2
- roles are `user`, `assistant`
- `metadata.task == "comprehension_raw"`
- passage/question/answer not empty
- `answer_text == assistant.content`
- `messages[0].content` format check:
  - starts with `"Đoạn văn: "`
  - contains `"\n\nCâu hỏi: "`
- top-level `context`/`question`/`answer_text` exist
- `messages[0].content` equals formatted `context` + `question`
- top-level `answer_text` equals `metadata.answer_text`
- answer present in passage
- if `span_check_mode == "strict_exact"`, span matches exactly
- dedup hash uniqueness
- source/split distribution
- `--sample` flag (default 5) for random inspection

## Testing
- Run the loader
- Run the recheck script

## Out of Scope
- Generating distractors or MCQ conversion
- Any synthetic data expansion
