# Design: comprehension_mcq_seed_final.jsonl

## Goal

Build `seed_exports/comprehension_mcq_seed_final.jsonl` from the existing raw extractive QA pool:

```text
seed_exports/comprehension_seed_raw.jsonl
```

The final dataset converts UIT-ViQuAD2.0 extractive QA records:

```text
passage + question -> answer_text
```

into the train MCQ schema:

```text
passage + question + A/B/C/D -> Đáp án: X
```

This supports the project-level `comprehension_mcq` bucket while keeping VLSP 2023 benchmark data as holdout evaluation data only.

## Scope

V1 uses offline JSONL boundaries for model work.

In scope:
- Filter the raw comprehension pool to UIT-ViQuAD2.0 only.
- Prepare distractor-generation request JSONL.
- Parse externally generated distractor outputs.
- Build MCQ candidate records with deterministic answer-label placement.
- Apply rule-based QC.
- Prepare solver request JSONL.
- Parse externally generated solver outputs and apply solver QC.
- Run lexical leakage checks against benchmark data.
- Finalize and recheck the final JSONL.
- Produce reports and rejects for auditability.

Out of scope for v1:
- Calling an LLM/API provider directly.
- Managing API keys, secrets, provider retries, or rate limits.
- Semantic embedding leakage checks.
- Using `ShynBui/Vietnamese_Reading_Comprehension_Dataset` in the final MCQ seed.

## Source Policy

Input source:

```text
seed_exports/comprehension_seed_raw.jsonl
```

Keep only records with:

```text
metadata.source_dataset == "taidng/UIT-ViQuAD2.0"
metadata.span_check_mode == "strict_exact"
metadata.answer_variants is non-empty
```

Do not include `ShynBui/Vietnamese_Reading_Comprehension_Dataset` in `comprehension_mcq_seed_final.jsonl` v1. It remains available in the raw pool for future work.

Final MCQ rows use:

```text
metadata.task == "comprehension_mcq"
metadata.source == "synthetic"
metadata.source_dataset == "taidng/UIT-ViQuAD2.0"
```

Rationale: the passage, question, and gold answer come from a public source, but the A/B/C/D MCQ item contains LLM-generated distractors.

## Pipeline Architecture

The pipeline is a sequence of deterministic scripts separated by two model-output boundaries:

1. Raw UIT filtering.
2. Distractor generation request preparation.
3. External distractor generation.
4. Candidate MCQ construction.
5. Rule-based QC.
6. Solver request preparation.
7. External solver/model QC.
8. Leakage check.
9. Finalization and recheck.

The repository scripts do not call model providers. They read and write JSONL files only.

## Files

Scripts:

```text
scripts/filter_comprehension_raw_uit.py
scripts/prepare_comprehension_mcq_generation.py
scripts/build_comprehension_mcq_candidates.py
scripts/qc_comprehension_mcq_seed.py
scripts/prepare_comprehension_mcq_solver.py
scripts/apply_comprehension_mcq_solver_qc.py
scripts/check_comprehension_mcq_leakage.py
scripts/finalize_comprehension_mcq_seed.py
scripts/recheck_comprehension_mcq_seed.py
scripts/test_comprehension_mcq_seed.py
```

Primary outputs:

```text
seed_exports/comprehension_seed_raw_uit_only.jsonl
seed_exports/comprehension_mcq_generation_requests.jsonl
seed_exports/comprehension_mcq_generation_outputs_raw.jsonl
seed_exports/comprehension_mcq_seed_candidates.jsonl
seed_exports/comprehension_mcq_seed_rule_checked.jsonl
seed_exports/comprehension_mcq_solver_requests.jsonl
seed_exports/comprehension_mcq_solver_outputs_raw.jsonl
seed_exports/comprehension_mcq_seed_solver_checked.jsonl
seed_exports/comprehension_mcq_seed_no_leak.jsonl
seed_exports/comprehension_mcq_seed_final.jsonl
```

Audit outputs:

```text
seed_exports/comprehension_seed_raw_uit_only_report.json
seed_exports/comprehension_mcq_generation_request_rejects.jsonl
seed_exports/comprehension_mcq_generation_request_report.json
seed_exports/comprehension_mcq_seed_candidate_rejects.jsonl
seed_exports/comprehension_mcq_seed_candidate_report.json
seed_exports/comprehension_mcq_seed_rule_rejects.jsonl
seed_exports/comprehension_mcq_seed_rule_report.json
seed_exports/comprehension_mcq_seed_solver_rejects.jsonl
seed_exports/comprehension_mcq_seed_solver_report.json
seed_exports/comprehension_mcq_seed_leakage_rejects.jsonl
seed_exports/comprehension_mcq_seed_leakage_report.json
seed_exports/comprehension_mcq_seed_final_report.json
seed_exports/comprehension_mcq_seed_final_samples.jsonl
```

## Stage 1: UIT-Only Raw Filter

Script:

```text
scripts/filter_comprehension_raw_uit.py
```

Input:

```text
seed_exports/comprehension_seed_raw.jsonl
```

Outputs:

```text
seed_exports/comprehension_seed_raw_uit_only.jsonl
seed_exports/comprehension_seed_raw_uit_only_report.json
```

Required checks:
- Only `taidng/UIT-ViQuAD2.0` appears in `metadata.source_dataset`.
- Only `strict_exact` appears in `metadata.span_check_mode`.
- `metadata.answer_variants` is non-empty for every kept record.
- Existing raw schema remains unchanged.
- Invalid records count is `0` for an accepted run.

The report includes:
- `total_input`
- `total_kept`
- `total_rejected`
- `kept_by_split`
- `source_datasets`
- `span_check_modes`
- `empty_answer_variants`
- `invalid_records`

## Stage 2: Distractor Generation Requests

Script:

```text
scripts/prepare_comprehension_mcq_generation.py
```

Input:

```text
seed_exports/comprehension_seed_raw_uit_only.jsonl
```

Outputs:

```text
seed_exports/comprehension_mcq_generation_requests.jsonl
seed_exports/comprehension_mcq_generation_request_rejects.jsonl
seed_exports/comprehension_mcq_generation_request_report.json
```

Request schema:

```json
{
  "request_id": "cmcq-gen-<source_id-or-hash>",
  "source_id": "...",
  "source_split": "...",
  "context": "...",
  "question": "...",
  "gold_answer_text": "...",
  "answer_variants": ["...", "..."],
  "title": "...",
  "raw_dedup_hash": "...",
  "context_hash": "...",
  "generation_prompt_version": "comprehension_mcq_distractors_v1",
  "filter_version": "comprehension_mcq_generation_filter_v1"
}
```

Initial filters:
- `min_answer_chars = 2`
- `max_answer_chars = 220`
- `max_context_chars = 8000`
- non-empty question and context
- reject short generic answers where a rule can identify them reliably

Reject reasons:
- `answer_too_short`
- `answer_too_long`
- `context_too_long`
- `missing_context`
- `missing_question`
- `question_too_short`
- `generic_answer_text`
- `invalid_raw_record`

Generation prompt constraints:
- Generate exactly 3 incorrect options.
- Incorrect options must be plausible in the passage context.
- Incorrect options must not be synonyms or equivalents of the gold answer.
- Incorrect options must not contain the gold answer as a substring.
- Do not use options like `Không có thông tin`, `Tất cả các đáp án trên`, or `Cả A và B`.
- Keep option length close to the gold answer when possible.
- Return valid JSON only.

Expected external response payload:

```json
{
  "distractors": ["...", "...", "..."],
  "notes": "optional short reason"
}
```

## Stage 3: External Distractor Output Boundary

External file:

```text
seed_exports/comprehension_mcq_generation_outputs_raw.jsonl
```

Accepted raw output row schema:

```json
{
  "request_id": "cmcq-gen-...",
  "source_id": "...",
  "model": "...",
  "raw_response": "{\"distractors\":[\"...\",\"...\",\"...\"],\"notes\":\"...\"}",
  "parsed_response": {
    "distractors": ["...", "...", "..."],
    "notes": "..."
  },
  "created_at": "optional"
}
```

`parsed_response` is preferred when valid. Otherwise, the parser attempts to parse `raw_response` as JSON or fenced JSON. Rows are rejected when request IDs are missing, mismatched, malformed, duplicated, or the parsed response is invalid.

## Stage 4: Candidate MCQ Construction

Script:

```text
scripts/build_comprehension_mcq_candidates.py
```

Inputs:

```text
seed_exports/comprehension_seed_raw_uit_only.jsonl
seed_exports/comprehension_mcq_generation_outputs_raw.jsonl
```

Outputs:

```text
seed_exports/comprehension_mcq_seed_candidates.jsonl
seed_exports/comprehension_mcq_seed_candidate_rejects.jsonl
seed_exports/comprehension_mcq_seed_candidate_report.json
```

Candidate schema:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Đọc đoạn văn sau và chọn đáp án đúng.\n\nĐoạn văn: ...\n\nCâu hỏi: ...\nA. ...\nB. ...\nC. ...\nD. ..."
    },
    {
      "role": "assistant",
      "content": "Đáp án: C"
    }
  ],
  "metadata": {
    "task": "comprehension_mcq",
    "source": "synthetic",
    "source_dataset": "taidng/UIT-ViQuAD2.0",
    "source_split": "...",
    "source_id": "...",
    "title": "...",
    "context_hash": "...",
    "raw_dedup_hash": "...",
    "mcq_dedup_hash": "...",
    "gold_answer_text": "...",
    "answer": "C",
    "choices": {
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    },
    "generation_method": "llm_distractor_generation_v1",
    "generation_prompt_version": "comprehension_mcq_distractors_v1",
    "qc_version": "comprehension_mcq_uit_rule_qc_v1",
    "language": "vi",
    "difficulty": "medium"
  }
}
```

Answer placement is deterministic. Shuffle the gold answer and three distractors using a stable seed derived from `raw_dedup_hash` or `request_id`, not process-global random state.

Reject reasons:
- `missing_generation_output`
- `malformed_generation_json`
- `invalid_distractors`
- `wrong_distractor_count`
- `empty_distractor`
- `duplicate_distractors`
- `distractor_matches_gold`
- `distractor_contains_gold`
- `missing_raw_record`
- `duplicate_mcq_dedup_hash`

## Stage 5: Rule-Based QC

Script:

```text
scripts/qc_comprehension_mcq_seed.py
```

Input:

```text
seed_exports/comprehension_mcq_seed_candidates.jsonl
```

Outputs:

```text
seed_exports/comprehension_mcq_seed_rule_checked.jsonl
seed_exports/comprehension_mcq_seed_rule_rejects.jsonl
seed_exports/comprehension_mcq_seed_rule_report.json
```

Hard reject reasons:
- `invalid_messages`
- `invalid_assistant_answer_format`
- `invalid_answer_label`
- `missing_choices`
- `empty_choice`
- `duplicate_choices_normalized`
- `gold_answer_missing_from_choices`
- `gold_answer_label_mismatch`
- `distractor_matches_gold`
- `distractor_contains_gold`
- `banned_option_text`
- `missing_context_or_question`
- `invalid_task`
- `invalid_source_dataset`
- `choice_too_long`
- `choice_length_imbalance`
- `duplicate_mcq_dedup_hash`

V1 near-equivalence checks are conservative:
- lowercase
- strip surrounding whitespace
- collapse whitespace
- strip simple punctuation
- exact normalized equality
- substring containment
- simple token overlap/Jaccard only for obvious duplicates

Do not use semantic similarity in rule QC v1.

## Stage 6: Solver Request Boundary

Script:

```text
scripts/prepare_comprehension_mcq_solver.py
```

Input:

```text
seed_exports/comprehension_mcq_seed_rule_checked.jsonl
```

Output:

```text
seed_exports/comprehension_mcq_solver_requests.jsonl
```

Request rows include enough data for an external solver model:

```json
{
  "mcq_dedup_hash": "...",
  "source_id": "...",
  "context": "...",
  "question": "...",
  "choices": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "gold_answer": "C",
  "solver_prompt_version": "comprehension_mcq_solver_v1"
}
```

The solver prompt asks the model to return:

```json
{
  "predicted_answer": "C",
  "is_unambiguous": true,
  "bad_reason": null
}
```

## Stage 7: Solver QC

External file:

```text
seed_exports/comprehension_mcq_solver_outputs_raw.jsonl
```

Accepted raw output row schema:

```json
{
  "mcq_dedup_hash": "...",
  "source_id": "...",
  "model": "...",
  "raw_response": "{\"predicted_answer\":\"C\",\"is_unambiguous\":true,\"bad_reason\":null}",
  "parsed_response": {
    "predicted_answer": "C",
    "is_unambiguous": true,
    "bad_reason": null
  }
}
```

Script:

```text
scripts/apply_comprehension_mcq_solver_qc.py
```

Inputs:

```text
seed_exports/comprehension_mcq_seed_rule_checked.jsonl
seed_exports/comprehension_mcq_solver_outputs_raw.jsonl
```

Outputs:

```text
seed_exports/comprehension_mcq_seed_solver_checked.jsonl
seed_exports/comprehension_mcq_seed_solver_rejects.jsonl
seed_exports/comprehension_mcq_seed_solver_report.json
```

Keep only if:

```text
parsed_response.predicted_answer == metadata.answer
parsed_response.is_unambiguous == true
parsed_response.bad_reason is null or empty
```

Reject reasons:
- `missing_solver_output`
- `malformed_solver_json`
- `invalid_predicted_answer`
- `solver_answer_mismatch`
- `solver_marked_ambiguous`
- `solver_bad_reason`
- `duplicate_solver_output`

## Stage 8: Leakage Check

Script:

```text
scripts/check_comprehension_mcq_leakage.py
```

Default input:

```text
seed_exports/comprehension_mcq_seed_solver_checked.jsonl
```

Outputs:

```text
seed_exports/comprehension_mcq_seed_no_leak.jsonl
seed_exports/comprehension_mcq_seed_leakage_rejects.jsonl
seed_exports/comprehension_mcq_seed_leakage_report.json
```

The script accepts a benchmark path argument:

```bash
python scripts/check_comprehension_mcq_leakage.py \
  --input seed_exports/comprehension_mcq_seed_solver_checked.jsonl \
  --benchmark benchmark/comprehension \
  --output seed_exports/comprehension_mcq_seed_no_leak.jsonl
```

If the benchmark path is absent or missing, the script fails with a clear error unless `--allow-missing-benchmark` is explicitly passed. This prevents accidentally treating unchecked data as no-leak data.

V1 checks:
- exact normalized question match
- high fuzzy question overlap
- high context overlap by character or token shingles
- same normalized choices set
- same normalized answer plus highly similar question

Reject reasons:
- `exact_question_match`
- `near_duplicate_question`
- `high_context_overlap`
- `same_choices_pattern`
- `same_answer_similar_question`
- `invalid_benchmark_record`

## Stage 9: Finalization and Recheck

Script:

```text
scripts/finalize_comprehension_mcq_seed.py
```

Input:

```text
seed_exports/comprehension_mcq_seed_no_leak.jsonl
```

Outputs:

```text
seed_exports/comprehension_mcq_seed_final.jsonl
seed_exports/comprehension_mcq_seed_final_report.json
seed_exports/comprehension_mcq_seed_final_samples.jsonl
```

Final hard invariants:
- JSONL parse succeeds.
- `messages` length is exactly `2`.
- user content contains passage, question, and A/B/C/D.
- assistant content matches `^Đáp án: [ABCD]$`.
- `metadata.task == "comprehension_mcq"`.
- `metadata.source == "synthetic"`.
- `metadata.source_dataset == "taidng/UIT-ViQuAD2.0"`.
- choices contain exactly A/B/C/D.
- normalized choices are unique.
- answer label is valid.
- `metadata.choices[metadata.answer] == metadata.gold_answer_text`.
- `metadata.mcq_dedup_hash` is unique.
- `metadata.context_hash` exists.
- no leakage reject remains.

`scripts/recheck_comprehension_mcq_seed.py` is a validator/reporting script. It does not transform candidates into checked data. It should support validating the final file or any intermediate MCQ JSONL path via an argument.

## Error Handling

All scripts:
- Exit non-zero for missing required input files.
- Exit non-zero for malformed JSONL unless the malformed rows are the expected object of a rejects-producing parser.
- Write rejects with source identifiers and reason strings where possible.
- Write reports with counts, reject reasons, source split distribution, and answer label distribution where applicable.
- Avoid writing partial final files after hard invariant failures.

## Testing

Add:

```text
scripts/test_comprehension_mcq_seed.py
```

Unit tests cover:
- UIT-only filter invariants.
- Generation prefilter reasons.
- Generation request shape.
- Raw LLM JSON parsing, including fenced JSON.
- Candidate shuffle determinism.
- Answer label maps to the gold answer.
- Choice normalization and duplicate detection.
- Banned option detection.
- Rule QC reject reasons.
- Solver output parsing.
- Solver answer mismatch rejection.
- Lexical leakage exact and fuzzy helpers.
- Final schema validation.

Tests use small fixture records and do not require the full 20k-record dataset or real model outputs.

## Verification Commands

Implementation verification:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
python scripts/filter_comprehension_raw_uit.py
python scripts/prepare_comprehension_mcq_generation.py
python scripts/build_comprehension_mcq_candidates.py --help
python scripts/qc_comprehension_mcq_seed.py --help
python scripts/prepare_comprehension_mcq_solver.py --help
python scripts/apply_comprehension_mcq_solver_qc.py --help
python scripts/check_comprehension_mcq_leakage.py --help
python scripts/finalize_comprehension_mcq_seed.py --help
python scripts/recheck_comprehension_mcq_seed.py --help
```

Full dataset completion additionally requires real external generation and solver outputs:

```text
seed_exports/comprehension_mcq_generation_outputs_raw.jsonl
seed_exports/comprehension_mcq_solver_outputs_raw.jsonl
```

## Completion Levels

Pipeline implementation complete:
- All deterministic scripts exist.
- Unit tests pass.
- UIT-only filter runs on the existing raw file.
- Generation request export runs on the UIT-only file.
- Candidate parser, QC, solver parser, leakage, finalizer, and recheck pass on fixtures or sample external-output files.

Dataset final complete:
- Real distractor generation outputs exist.
- Real solver outputs exist.
- Rule QC passes.
- Solver QC passes.
- Leakage check runs against benchmark data.
- Final JSONL, report, and sample files are produced.
- Final volume is in the expected clean range, or any shortfall is explained by reject reports.

## Acceptance Criteria

`comprehension_mcq_seed_final.jsonl` is accepted only when:

1. It uses only `taidng/UIT-ViQuAD2.0`.
2. Every row has `metadata.task == "comprehension_mcq"`.
3. Every row has passage, question, and A/B/C/D in the user message.
4. Every assistant message is exactly `Đáp án: A/B/C/D`.
5. Every row has four unique choices.
6. The gold answer is exactly at `metadata.answer`.
7. Rule QC passes.
8. Solver/model QC passes.
9. Leakage check against VLSP benchmark data passes.
10. Final report and reject reports exist.

Expected v1 volume:

```text
raw UIT: about 21.7k
after generation prefilter: about 15k-21k
after generation parse: about 12k-20k
after rule QC: about 10k-18k
after solver QC + leakage: about 8k-15k
```

The target is 8k-15k clean final samples for the first training round.
