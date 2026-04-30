import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.comprehension_mcq_seed_common import (  # noqa: E402
    ANSWER_LABELS,
    canonical_choice_text,
    split_mcq_user_content,
)


DEFAULT_INPUT_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_candidates.jsonl"
DEFAULT_OUTPUT_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_rule_checked.jsonl"
DEFAULT_REJECTS_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_rule_rejects.jsonl"
DEFAULT_REPORT_JSON = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_rule_report.json"

QC_VERSION = "comprehension_mcq_rule_qc_v1"
TASK_NAME = "comprehension_mcq"
SOURCE_DATASET = "taidng/UIT-ViQuAD2.0"
MAX_CHOICE_CHARS = 300
MAX_CHOICE_LENGTH_RATIO = 4
MIN_CHOICE_LENGTH_FOR_IMBALANCE = 50
BANNED_OPTION_TEXTS = {
    "không có thông tin",
    "tất cả các đáp án trên",
    "cả a và b",
}

_ASSISTANT_RE = re.compile(r"^Đáp án: ([ABCD])$")


def read_jsonl(path):
    records = []
    malformed_rejects = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                malformed_rejects.append(
                    {
                        "reason": "invalid_json",
                        "source_id": f"line-{line_number}",
                        "source_split": "unknown",
                        "raw_preview": text[:200],
                    }
                )
    return records, malformed_rejects


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _preview(value):
    return "" if value is None else str(value)[:200]


def _metadata(record):
    if not isinstance(record, dict):
        return {}
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _choice_texts(choices):
    if not isinstance(choices, dict):
        return None
    if sorted(choices.keys()) != ANSWER_LABELS:
        return None
    return [choices[label] for label in ANSWER_LABELS]


def _choice_length_stats(choice_texts):
    lengths = [len(text) for text in choice_texts]
    shortest = min(lengths)
    longest = max(lengths)
    return shortest, longest


def _normalized_tokens(text):
    cleaned = re.sub(r"[^\w\s]+", " ", canonical_choice_text(text))
    return [token for token in cleaned.split() if token]


def _looks_near_duplicate(left, right):
    left_norm = canonical_choice_text(left)
    right_norm = canonical_choice_text(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if left_norm in right_norm or right_norm in left_norm:
        return True
    left_tokens = set(_normalized_tokens(left_norm))
    right_tokens = set(_normalized_tokens(right_norm))
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return union > 0 and overlap / union >= 0.8 and overlap >= 2


def _assistant_answer_label(record):
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        return None
    assistant = messages[1]
    if not isinstance(assistant, dict):
        return None
    content = assistant.get("content")
    if not isinstance(content, str):
        return None
    match = _ASSISTANT_RE.match(content.strip())
    if not match:
        return None
    return match.group(1)


def _record_reject_reason(record, seen_mcq_dedup_hashes=None):
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        return "invalid_messages"
    if not all(isinstance(message, dict) for message in messages):
        return "invalid_messages"
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        return "invalid_messages"

    metadata = _metadata(record)
    if metadata.get("task") != TASK_NAME:
        return "invalid_task"
    if metadata.get("source_dataset") != SOURCE_DATASET:
        return "invalid_source_dataset"

    assistant_label = _assistant_answer_label(record)
    if assistant_label is None:
        return "invalid_assistant_answer_format"

    answer_label = metadata.get("answer")
    if answer_label not in ANSWER_LABELS:
        return "invalid_answer_label"

    choices = _choice_texts(metadata.get("choices"))
    if choices is None:
        return "missing_choices"
    if any(not isinstance(choice, str) or not choice.strip() for choice in choices):
        return "empty_choice"

    normalized_choices = [canonical_choice_text(choice) for choice in choices]
    if len(set(normalized_choices)) != 4:
        return "duplicate_choices_normalized"

    gold_answer_text = metadata.get("gold_answer_text")
    if not isinstance(gold_answer_text, str) or not gold_answer_text.strip():
        return "gold_answer_missing_from_choices"

    normalized_gold = canonical_choice_text(gold_answer_text)
    if normalized_gold not in normalized_choices:
        return "gold_answer_missing_from_choices"

    if canonical_choice_text(choices[ANSWER_LABELS.index(answer_label)]) != normalized_gold:
        return "gold_answer_label_mismatch"

    for choice_text, normalized_choice in zip(choices, normalized_choices):
        if len(choice_text) > MAX_CHOICE_CHARS:
            return "choice_too_long"
        if normalized_choice in BANNED_OPTION_TEXTS:
            return "banned_option_text"

    gold_choice = choices[ANSWER_LABELS.index(answer_label)]
    for idx, choice_text in enumerate(choices):
        if idx == ANSWER_LABELS.index(answer_label):
            continue
        if normalized_gold and normalized_gold in canonical_choice_text(choice_text):
            return "distractor_contains_gold"
        if _looks_near_duplicate(choice_text, gold_choice):
            return "distractor_matches_gold"

    shortest, longest = _choice_length_stats(choices)
    if longest >= MIN_CHOICE_LENGTH_FOR_IMBALANCE and longest >= shortest * MAX_CHOICE_LENGTH_RATIO:
        return "choice_length_imbalance"

    user_content = messages[0].get("content")
    if not isinstance(user_content, str) or not user_content.strip():
        return "missing_context_or_question"

    try:
        context, question, _choice_lines = split_mcq_user_content(user_content)
    except Exception:
        return "missing_choices"

    if not context.strip() or not question.strip():
        return "missing_context_or_question"

    dedup_hash = metadata.get("mcq_dedup_hash")
    if not isinstance(dedup_hash, str) or not dedup_hash.strip():
        return "duplicate_mcq_dedup_hash"
    if seen_mcq_dedup_hashes is not None:
        if dedup_hash in seen_mcq_dedup_hashes:
            return "duplicate_mcq_dedup_hash"

    return None


def rule_check_record(record, seen_mcq_dedup_hashes=None):
    reason = _record_reject_reason(record, seen_mcq_dedup_hashes=seen_mcq_dedup_hashes)
    return reason is None, reason


def _reject_row(record, reason):
    metadata = _metadata(record)
    messages = record.get("messages") if isinstance(record, dict) else []
    user_content = messages[0].get("content") if isinstance(messages, list) and messages else None
    assistant_content = messages[1].get("content") if isinstance(messages, list) and len(messages) > 1 else None
    return {
        "reason": reason,
        "source_id": metadata.get("source_id", "unknown"),
        "source_split": metadata.get("source_split", "unknown"),
        "source_dataset": metadata.get("source_dataset", "unknown"),
        "mcq_dedup_hash": metadata.get("mcq_dedup_hash"),
        "answer": metadata.get("answer"),
        "gold_answer_text": metadata.get("gold_answer_text"),
        "user_preview": _preview(user_content),
        "assistant_preview": _preview(assistant_content),
    }


def _soft_check_row(record):
    metadata = _metadata(record)
    choice_texts = _choice_texts(metadata.get("choices"))
    if choice_texts is None:
        return {}

    lengths = [len(choice) for choice in choice_texts]
    return {
        "long_choice": sum(length > 80 for length in lengths),
        "choice_length_imbalance": int(max(lengths) - min(lengths) >= 20),
        "long_gold_answer": int(len(metadata.get("gold_answer_text", "")) > 80),
    }


def process_records(records):
    kept = []
    rejects = []
    reject_reasons = Counter()
    kept_by_split = Counter()
    answer_labels = Counter()
    soft_checks = Counter()
    seen_mcq_dedup_hashes = set()

    for record in records:
        reason = _record_reject_reason(record, seen_mcq_dedup_hashes=seen_mcq_dedup_hashes)
        if reason is None:
            kept.append(record)
            metadata = _metadata(record)
            source_split = metadata.get("source_split")
            kept_by_split[str(source_split) if source_split not in (None, "") else "unknown"] += 1
            answer_labels[str(metadata.get("answer", "unknown"))] += 1
            dedup_hash = metadata.get("mcq_dedup_hash")
            if isinstance(dedup_hash, str) and dedup_hash:
                seen_mcq_dedup_hashes.add(dedup_hash)
            for key, count in _soft_check_row(record).items():
                soft_checks[key] += count
            continue

        reject_reasons[reason] += 1
        rejects.append(_reject_row(record, reason))

    report = {
        "input_jsonl": None,
        "output_jsonl": None,
        "rejects_jsonl": None,
        "report_json": None,
        "total_loaded": len(records),
        "total_kept": len(kept),
        "total_rejected": len(rejects),
        "kept_by_split": dict(kept_by_split),
        "answer_labels": dict(answer_labels),
        "reject_reasons": dict(reject_reasons),
        "soft_checks": dict(soft_checks),
        "qc_version": QC_VERSION,
    }
    return kept, rejects, report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--rejects-jsonl", default=DEFAULT_REJECTS_JSONL)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    args = parser.parse_args(argv)

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        print(f"[error] missing input file: {input_path}")
        return 1

    records, malformed_rejects = read_jsonl(input_path)
    kept, rejects, report = process_records(records)
    rejects = malformed_rejects + rejects
    report["total_loaded"] = len(records) + len(malformed_rejects)
    report["total_rejected"] = len(rejects)
    report["reject_reasons"]["invalid_json"] = len(malformed_rejects)

    output_path = Path(args.output_jsonl)
    rejects_path = Path(args.rejects_jsonl)
    report_path = Path(args.report_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_path, kept)
    write_jsonl(rejects_path, rejects)
    report["input_jsonl"] = str(input_path)
    report["output_jsonl"] = str(output_path)
    report["rejects_jsonl"] = str(rejects_path)
    report["report_json"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] kept={len(kept)} rejected={len(rejects)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
