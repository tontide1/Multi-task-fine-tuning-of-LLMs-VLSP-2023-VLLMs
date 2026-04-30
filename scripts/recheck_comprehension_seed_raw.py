import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


JSONL_PATH = Path(__file__).resolve().parent.parent / "seed_exports" / "comprehension_seed_raw.jsonl"
REQUIRED_METADATA_FIELDS = [
    "task",
    "source",
    "source_dataset",
    "source_split",
    "source_id",
    "answer_start",
    "original_answer_start",
    "answer_text",
    "answer_variants",
    "title",
    "language",
    "difficulty",
    "span_check_mode",
    "dedup_hash",
    "qc_version",
]


def format_user_content(context, question):
    return f"Đoạn văn: {context}\n\nCâu hỏi: {question}"


def validate_record_schema(record):
    errors = []
    if not isinstance(record, dict):
        return ["record is not a dict"]

    context = record.get("context")
    question = record.get("question")
    answer_text = record.get("answer_text")
    messages = record.get("messages")
    metadata = record.get("metadata")

    if not isinstance(messages, list):
        errors.append("messages is not a list")
    elif len(messages) != 2:
        errors.append("messages must have exactly 2 elements")
    else:
        first, second = messages
        if not isinstance(first, dict):
            errors.append("first message is not a dict")
        else:
            if first.get("role") != "user":
                errors.append("first role is not user")
            if first.get("content") != format_user_content(context, question):
                errors.append("user content does not match context/question")
        if not isinstance(second, dict):
            errors.append("second message is not a dict")
        else:
            if second.get("role") != "assistant":
                errors.append("second role is not assistant")
            if second.get("content") != answer_text:
                errors.append("assistant content does not match answer_text")

    if not isinstance(metadata, dict):
        errors.append("metadata is not a dict")
        return errors

    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            errors.append(f"missing metadata field: {field}")

    if metadata.get("task") != "comprehension_raw":
        errors.append("metadata.task is not comprehension_raw")

    if not isinstance(context, str) or not context.strip():
        errors.append("context is not a non-empty string")
    if not isinstance(question, str) or not question.strip():
        errors.append("question is not a non-empty string")
    if not isinstance(answer_text, str) or not answer_text.strip():
        errors.append("answer_text is not a non-empty string")
    elif isinstance(context, str) and context.strip() and answer_text not in context:
        errors.append("answer_text does not appear in context")

    if metadata.get("answer_text") != answer_text:
        errors.append("metadata.answer_text does not match top-level answer_text")

    dedup_hash = metadata.get("dedup_hash")
    if not isinstance(dedup_hash, str) or not dedup_hash.strip():
        errors.append("metadata.dedup_hash is not a non-empty string")

    span_check_mode = metadata.get("span_check_mode")
    answer_start = metadata.get("answer_start")
    if (
        span_check_mode == "strict_exact"
        and isinstance(context, str)
        and isinstance(answer_text, str)
        and isinstance(answer_start, int)
        and not isinstance(answer_start, bool)
    ):
        if context[answer_start : answer_start + len(answer_text)] != answer_text:
            errors.append("strict_exact span does not match context slice")

    return errors


def _load_records(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _print_distribution(title, counts):
    print(title)
    if counts:
        for key in sorted(counts):
            print(f"{key}: {counts[key]}")
    else:
        print("(empty)")
    print()


def _sample_message_content(record, index):
    if not isinstance(record, dict):
        return ""
    messages = record.get("messages")
    if not isinstance(messages, list) or index >= len(messages):
        return ""
    message = messages[index]
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5)
    args = parser.parse_args(argv)

    if not JSONL_PATH.exists():
        print(f"[error] missing input file: {JSONL_PATH}")
        return 1

    records = _load_records(JSONL_PATH)
    source_counts = Counter()
    split_counts = Counter()
    invalid_records = []
    seen_dedup_hashes = set()

    for index, record in enumerate(records):
        metadata = record.get("metadata") if isinstance(record, dict) else {}
        if isinstance(metadata, dict):
            source_counts[str(metadata.get("source_dataset", "unknown"))] += 1
            split_counts[str(metadata.get("source_split", "unknown"))] += 1
        errors = validate_record_schema(record)
        dedup_hash = metadata.get("dedup_hash") if isinstance(metadata, dict) else None
        if isinstance(dedup_hash, str) and dedup_hash.strip():
            if dedup_hash in seen_dedup_hashes:
                errors.append("duplicate metadata.dedup_hash")
            else:
                seen_dedup_hashes.add(dedup_hash)
        if errors:
            invalid_records.append((index, metadata.get("source_id", "unknown") if isinstance(metadata, dict) else "unknown", errors))

    print(f"=== Total records: {len(records)} ===")
    print()
    _print_distribution("=== Source distribution ===", source_counts)
    _print_distribution("=== Split distribution ===", split_counts)

    print("=== Schema validation ===")
    print(f"Invalid records: {len(invalid_records)}")
    if invalid_records:
        for index, source_id, errors in invalid_records[:5]:
            print(f"- [{index}] source_id={source_id}: {', '.join(errors)}")
    print()

    print(f"=== Random sample ({args.sample}) ===")
    rng = random.Random(42)
    sample_size = min(args.sample, len(records))
    sample = rng.sample(records, sample_size) if sample_size else []
    for index, record in enumerate(sample, 1):
        metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
        print(f"--- sample {index} ---")
        print("[user]")
        print(_sample_message_content(record, 0))
        print("[assistant]")
        print(_sample_message_content(record, 1))
        print("[metadata]")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        print()

    return 1 if invalid_records else 0


if __name__ == "__main__":
    sys.exit(main())
