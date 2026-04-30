import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Allow running this script directly from the repo root while keeping shared imports working.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.comprehension_mcq_seed_common import (
    build_mcq_user_content,
    canonical_choice_text,
    compute_context_hash,
    extract_json_object,
    is_near_duplicate_text,
    make_mcq_dedup_hash,
    stable_choice_labels,
)


DEFAULT_RAW_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_seed_raw_uit_only.jsonl"
DEFAULT_GENERATION_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_generation_outputs_raw.jsonl"
DEFAULT_OUTPUT_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_candidates.jsonl"
DEFAULT_REJECTS_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_candidate_rejects.jsonl"
DEFAULT_REPORT_JSON = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_candidate_report.json"

GENERATION_PROMPT_VERSION = "comprehension_mcq_distractors_v1"
GENERATION_METHOD = "llm_distractor_generation_v1"
QC_VERSION = "comprehension_mcq_uit_rule_qc_v1"
SOURCE_DATASET = "taidng/UIT-ViQuAD2.0"
TASK_NAME = "comprehension_mcq"
SOURCE_NAME = "synthetic"
LANGUAGE = "vi"
DIFFICULTY = "medium"


def read_jsonl(path):
    records = []
    malformed_rejects = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                malformed_rejects.append(
                    {
                        "reason": "invalid_json",
                        "source_id": f"line-{line_number}",
                        "source_split": "unknown",
                        "context_preview": "",
                        "question": "",
                        "answer_text": "",
                        "raw_preview": line[:200],
                    }
                )
    return records, malformed_rejects


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _metadata(record):
    if not isinstance(record, dict):
        return None
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else None


def _string_field(record, key):
    if not isinstance(record, dict):
        return None
    value = record.get(key)
    return value if isinstance(value, str) else None


def _preview(value):
    return "" if value is None else str(value)[:200]


def _source_id(record):
    metadata = _metadata(record)
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("source_id")
    return value if isinstance(value, str) and value else None


def _source_split(record):
    metadata = _metadata(record)
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("source_split")
    return value if isinstance(value, str) and value else None


def _generation_text(row):
    if not isinstance(row, dict):
        return None
    for key in ("raw_response", "generation_output", "output", "response_text", "response", "content"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
    parsed = row.get("parsed_response")
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    return None


def _raw_record_from_generation_row(row):
    if not isinstance(row, dict):
        return None
    raw_record = row.get("raw_record")
    if isinstance(raw_record, dict):
        return raw_record
    return None


def _join_key_candidates(record):
    key = _source_id(record)
    if key:
        return [key, f"cmcq-gen-{key}"]
    metadata = _metadata(record)
    if isinstance(metadata, dict):
        request_id = metadata.get("request_id")
        if isinstance(request_id, str) and request_id:
            return [request_id]
    return []


def parse_generation_output(raw_text):
    parsed = raw_text if isinstance(raw_text, dict) else extract_json_object(raw_text)
    if not isinstance(parsed, dict):
        raise TypeError("generation output must be a JSON object")
    distractors = parsed.get("distractors")
    if not isinstance(distractors, list):
        raise ValueError("distractors must be a list")
    if len(distractors) != 3:
        raise ValueError("expected exactly 3 distractors")
    return parsed


def build_candidate_record(raw_record, distractors):
    context = raw_record["context"]
    question = raw_record["question"]
    gold_answer = raw_record["answer_text"]
    labels = stable_choice_labels(raw_record["metadata"]["dedup_hash"])
    choices_in_label_order = [gold_answer, *distractors]
    rng = random.Random(raw_record["metadata"]["dedup_hash"])
    rng.shuffle(choices_in_label_order)
    answer_label = labels[choices_in_label_order.index(gold_answer)]
    choices = {label: value for label, value in zip(labels, choices_in_label_order)}
    choice_list = [choices["A"], choices["B"], choices["C"], choices["D"]]
    return {
        "messages": [
            {"role": "user", "content": build_mcq_user_content(context, question, choice_list)},
            {"role": "assistant", "content": f"Đáp án: {answer_label}"},
        ],
        "metadata": {
            "task": TASK_NAME,
            "source": SOURCE_NAME,
            "source_dataset": SOURCE_DATASET,
            "source_split": raw_record["metadata"]["source_split"],
            "source_id": raw_record["metadata"]["source_id"],
            "title": raw_record["metadata"].get("title"),
            "context_hash": compute_context_hash(context),
            "raw_dedup_hash": raw_record["metadata"]["dedup_hash"],
            "mcq_dedup_hash": make_mcq_dedup_hash(context, question, choice_list),
            "gold_answer_text": gold_answer,
            "answer": answer_label,
            "choices": choices,
            "generation_method": GENERATION_METHOD,
            "generation_prompt_version": GENERATION_PROMPT_VERSION,
            "qc_version": QC_VERSION,
            "language": LANGUAGE,
            "difficulty": DIFFICULTY,
        },
    }


def _reject_record(raw_record, reason, generation_row=None, generation_text=None):
    metadata = _metadata(raw_record)
    return {
        "source_id": _source_id(raw_record) if _source_id(raw_record) is not None else "unknown",
        "source_split": _source_split(raw_record) if _source_split(raw_record) is not None else "unknown",
        "reason": reason,
        "context_preview": _preview(_string_field(raw_record, "context")),
        "question": _preview(_string_field(raw_record, "question")),
        "answer_text": _preview(_string_field(raw_record, "answer_text")),
        "request_id": generation_row.get("request_id") if isinstance(generation_row, dict) else None,
        "generation_preview": _preview(generation_text),
        "title": metadata.get("title") if isinstance(metadata, dict) else None,
    }


def _validate_distractors(gold_answer_text, distractors):
    normalized = []
    for distractor in distractors:
        if not isinstance(distractor, str):
            return False, "invalid_distractors"
        cleaned = distractor.strip()
        if not cleaned:
            return False, "empty_distractor"
        normalized.append(cleaned)

    if len(set(canonical_choice_text(text) for text in normalized)) != 3:
        return False, "duplicate_distractors"

    gold_norm = canonical_choice_text(gold_answer_text)
    for distractor in normalized:
        distractor_norm = canonical_choice_text(distractor)
        if distractor_norm == gold_norm:
            return False, "distractor_matches_gold"
        if gold_norm and gold_norm in distractor_norm:
            return False, "distractor_contains_gold"
        if distractor_norm and is_near_duplicate_text(distractor_norm, gold_norm):
            return False, "distractor_matches_gold"

    return True, normalized


def _find_generation_row(generation_index, raw_record):
    for key in _join_key_candidates(raw_record):
        row = generation_index.get(key)
        if row is not None:
            return row
    return None


def build_candidates(raw_records, generation_rows):
    generation_index = {}
    for row in generation_rows:
        if not isinstance(row, dict):
            continue
        for key in ("source_id", "request_id"):
            value = row.get(key)
            if isinstance(value, str) and value:
                generation_index.setdefault(value, row)

    kept = []
    rejects = []
    reject_reasons = Counter()
    kept_by_split = Counter()

    for raw_record in raw_records:
        generation_row = _find_generation_row(generation_index, raw_record)
        if generation_row is None:
            reject_reasons["missing_generation_output"] += 1
            rejects.append(_reject_record(raw_record, "missing_generation_output"))
            continue

        generation_text = _generation_text(generation_row)
        if generation_text is None:
            reject_reasons["missing_generation_output"] += 1
            rejects.append(_reject_record(raw_record, "missing_generation_output", generation_row))
            continue

        try:
            parsed = parse_generation_output(generation_text)
        except json.JSONDecodeError:
            reject_reasons["malformed_generation_json"] += 1
            rejects.append(_reject_record(raw_record, "malformed_generation_json", generation_row, generation_text))
            continue
        except TypeError:
            reject_reasons["invalid_distractors"] += 1
            rejects.append(_reject_record(raw_record, "invalid_distractors", generation_row, generation_text))
            continue
        except ValueError:
            reject_reasons["wrong_distractor_count"] += 1
            rejects.append(_reject_record(raw_record, "wrong_distractor_count", generation_row, generation_text))
            continue

        distractors = parsed.get("distractors") if isinstance(parsed, dict) else None
        if not isinstance(distractors, list):
            reject_reasons["invalid_distractors"] += 1
            rejects.append(_reject_record(raw_record, "invalid_distractors", generation_row, generation_text))
            continue

        valid, validated_distractors_or_reason = _validate_distractors(raw_record["answer_text"], distractors)
        if not valid:
            reason = validated_distractors_or_reason
            reject_reasons[reason] += 1
            rejects.append(_reject_record(raw_record, reason, generation_row, generation_text))
            continue

        candidate = build_candidate_record(raw_record, validated_distractors_or_reason)
        kept.append(candidate)
        source_split = candidate["metadata"]["source_split"]
        kept_by_split[source_split if source_split else "unknown"] += 1

    report = {
        "input_raw_jsonl": None,
        "input_generation_jsonl": None,
        "output_jsonl": None,
        "rejects_jsonl": None,
        "report_json": None,
        "total_loaded": len(raw_records),
        "total_kept": len(kept),
        "total_rejected": len(rejects),
        "kept_by_split": dict(kept_by_split),
        "reject_reasons": dict(reject_reasons),
        "generation_prompt_version": GENERATION_PROMPT_VERSION,
        "generation_method": GENERATION_METHOD,
        "qc_version": QC_VERSION,
        "source_dataset": SOURCE_DATASET,
    }
    return kept, rejects, report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-jsonl", default=DEFAULT_RAW_JSONL)
    parser.add_argument("--generation-jsonl", default=DEFAULT_GENERATION_JSONL)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--rejects-jsonl", default=DEFAULT_REJECTS_JSONL)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    args = parser.parse_args(argv)

    raw_path = Path(args.raw_jsonl)
    generation_path = Path(args.generation_jsonl)
    if not raw_path.exists():
        print(f"[error] missing raw input file: {raw_path}")
        return 1
    if not generation_path.exists():
        print(f"[error] missing generation input file: {generation_path}")
        return 1

    raw_records, raw_malformed_rejects = read_jsonl(raw_path)
    generation_records, generation_malformed_rejects = read_jsonl(generation_path)

    kept, rejects, report = build_candidates(raw_records, generation_records)
    rejects = raw_malformed_rejects + generation_malformed_rejects + rejects
    report["total_loaded"] = len(raw_records) + len(raw_malformed_rejects)
    report["total_rejected"] = len(rejects)
    if raw_malformed_rejects:
        report["reject_reasons"]["invalid_raw_json"] = len(raw_malformed_rejects)
    if generation_malformed_rejects:
        report["reject_reasons"]["invalid_generation_json"] = len(generation_malformed_rejects)
    report["generation_rows_loaded"] = len(generation_records) + len(generation_malformed_rejects)

    output_path = Path(args.output_jsonl)
    rejects_path = Path(args.rejects_jsonl)
    report_path = Path(args.report_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_path, kept)
    write_jsonl(rejects_path, rejects)

    report.update(
        {
            "input_raw_jsonl": str(raw_path),
            "input_generation_jsonl": str(generation_path),
            "output_jsonl": str(output_path),
            "rejects_jsonl": str(rejects_path),
            "report_json": str(report_path),
        }
    )
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"[ok] wrote {output_path}")
    print(f"[ok] wrote {rejects_path}")
    print(f"[ok] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
