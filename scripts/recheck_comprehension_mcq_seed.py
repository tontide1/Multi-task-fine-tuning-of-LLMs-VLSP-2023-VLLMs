import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.comprehension_mcq_seed_common import ANSWER_LABELS, canonical_choice_text, split_mcq_user_content  # noqa: E402


DEFAULT_INPUT_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_final.jsonl"
VALID_TASK = "comprehension_mcq"
VALID_SOURCE = "synthetic"
VALID_SOURCE_DATASET = "taidng/UIT-ViQuAD2.0"
VALID_ASSISTANT_RE = re.compile(r"^Đáp án: [ABCD]$")


def read_jsonl(path):
    records = []
    malformed = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                malformed.append((line_number, text))
    return records, malformed


def _preview(value):
    return "" if value is None else str(value)[:200]


def _metadata(record):
    if not isinstance(record, dict):
        return {}
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _format_counter(counter):
    return json.dumps(dict(counter), ensure_ascii=False, indent=2) if counter else "N/A"


def validate_record_schema(record):
    errors = []
    if not isinstance(record, dict):
        return ["record_is_not_object"]

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        return ["invalid_messages"]
    if not all(isinstance(message, dict) for message in messages):
        errors.append("invalid_messages")
        return errors

    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        errors.append("invalid_messages")

    user_content = messages[0].get("content")
    assistant_content = messages[1].get("content")
    if not isinstance(user_content, str) or not user_content.strip():
        errors.append("empty_user_content")
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        errors.append("empty_assistant_content")
    if isinstance(assistant_content, str) and not VALID_ASSISTANT_RE.match(assistant_content.strip()):
        errors.append("invalid_assistant_answer_format")

    metadata = _metadata(record)
    required_fields = [
        "task",
        "source",
        "source_dataset",
        "source_split",
        "source_id",
        "gold_answer_text",
        "answer",
        "choices",
        "mcq_dedup_hash",
        "context_hash",
    ]
    for field in required_fields:
        if field not in metadata:
            errors.append(f"missing_metadata_{field}")

    if metadata.get("task") != VALID_TASK:
        errors.append("invalid_task")
    if metadata.get("source") != VALID_SOURCE:
        errors.append("invalid_source")
    if metadata.get("source_dataset") != VALID_SOURCE_DATASET:
        errors.append("invalid_source_dataset")

    answer_label = metadata.get("answer")
    if answer_label not in ANSWER_LABELS:
        errors.append("invalid_answer_label")

    choices = metadata.get("choices")
    if not isinstance(choices, dict) or sorted(choices.keys()) != ANSWER_LABELS:
        errors.append("missing_choices")
    else:
        choice_values = [choices[label] for label in ANSWER_LABELS]
        if any(not isinstance(choice, str) or not choice.strip() for choice in choice_values):
            errors.append("empty_choice")
        normalized = [canonical_choice_text(choice) for choice in choice_values]
        if len(set(normalized)) != 4:
            errors.append("duplicate_choices_normalized")
        gold_answer_text = metadata.get("gold_answer_text")
        if not isinstance(gold_answer_text, str) or not gold_answer_text.strip():
            errors.append("gold_answer_missing_from_choices")
        else:
            normalized_gold = canonical_choice_text(gold_answer_text)
            if normalized_gold not in normalized:
                errors.append("gold_answer_missing_from_choices")
            elif answer_label in ANSWER_LABELS:
                answer_choice = choices.get(answer_label)
                if canonical_choice_text(answer_choice) != normalized_gold:
                    errors.append("gold_answer_label_mismatch")

    if isinstance(user_content, str) and user_content.strip():
        try:
            context, question, _choices = split_mcq_user_content(user_content)
        except Exception:
            errors.append("missing_choices")
        else:
            if not context.strip() or not question.strip():
                errors.append("missing_context_or_question")

    if metadata.get("mcq_dedup_hash") in (None, ""):
        errors.append("missing_metadata_mcq_dedup_hash")

    return errors


def _reject_row(record, errors, line_number=None):
    metadata = _metadata(record)
    messages = record.get("messages") if isinstance(record, dict) else []
    user_content = messages[0].get("content") if isinstance(messages, list) and messages else None
    assistant_content = messages[1].get("content") if isinstance(messages, list) and len(messages) > 1 else None
    return {
        "line_number": line_number,
        "source_id": metadata.get("source_id", "unknown"),
        "source_split": metadata.get("source_split", "unknown"),
        "source_dataset": metadata.get("source_dataset", "unknown"),
        "errors": errors,
        "user_preview": _preview(user_content),
        "assistant_preview": _preview(assistant_content),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--sample", type=int, default=5)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] missing input file: {input_path}")
        return 1

    records, malformed = read_jsonl(input_path)
    invalid_records = []
    source_datasets = Counter()
    source_splits = Counter()
    answer_labels = Counter()

    for index, record in enumerate(records):
        errors = validate_record_schema(record)
        if errors:
            invalid_records.append(_reject_row(record, errors, line_number=index + 1))
            continue
        metadata = _metadata(record)
        source_datasets[str(metadata.get("source_dataset", "unknown"))] += 1
        source_splits[str(metadata.get("source_split", "unknown"))] += 1
        answer_labels[str(metadata.get("answer", "unknown"))] += 1

    print(f"=== [1] Total records in JSONL: {len(records) + len(malformed)} ===")
    print(f"   Valid records: {len(records) - len(invalid_records)}")
    print(f"   Invalid records: {len(invalid_records)}")
    print()
    print("=== [2] Source dataset distribution ===")
    print(_format_counter(source_datasets))
    print()
    print("=== [3] Source split distribution ===")
    print(_format_counter(source_splits))
    print()
    print("=== [4] Answer label distribution ===")
    print(_format_counter(answer_labels))
    print()
    print("=== [5] Invalid records ===")
    if invalid_records:
        for row in invalid_records[:5]:
            print(f"  - line {row['line_number']}: {row['source_id']} :: {', '.join(row['errors'])}")
    else:
        print("  none")
    if malformed:
        print()
        print(f"=== [6] Malformed JSON rows: {len(malformed)} ===")
        print(f"  first line: {malformed[0][0]}")

    if args.sample > 0 and records:
        print()
        print("=== [7] Sample ===")
        sample = records[: min(args.sample, len(records))]
        for index, record in enumerate(sample, start=1):
            print(f"--- sample {index} ---")
            print(record["messages"][0]["content"])
            print(record["messages"][1]["content"])

    return 1 if invalid_records or malformed else 0


if __name__ == "__main__":
    raise SystemExit(main())
