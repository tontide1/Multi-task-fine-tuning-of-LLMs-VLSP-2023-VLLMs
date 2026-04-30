import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.comprehension_mcq_seed_common import canonical_choice_text, normalize_for_hash, split_mcq_user_content


DEFAULT_INPUT_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_solver_checked.jsonl"
DEFAULT_BENCHMARK_PATH = _REPO_ROOT / "benchmark" / "comprehension"
DEFAULT_OUTPUT_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_no_leak.jsonl"
DEFAULT_REJECTS_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_leakage_rejects.jsonl"
DEFAULT_REPORT_JSON = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_leakage_report.json"
ANSWER_LABELS = {"A", "B", "C", "D"}


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
    return records


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_text(value):
    return normalize_for_hash(value)


def _normalize_question(value):
    text = _normalize_text(value)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_context(value):
    return _normalize_text(value)


def _sequence_ratio(left, right):
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _token_ngrams(text, size):
    tokens = [token for token in text.split() if token]
    if len(tokens) < size:
        return set()
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _jaccard(left, right):
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


def has_exact_question_match(left, right):
    left_text = _normalize_question(left)
    right_text = _normalize_question(right)
    return bool(left_text) and left_text == right_text


def is_near_duplicate_question(left, right):
    left_text = _normalize_question(left)
    right_text = _normalize_question(right)
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return False
    left_tokens = sorted(left_text.split())
    right_tokens = sorted(right_text.split())
    if left_tokens and left_tokens == right_tokens:
        return True
    if _jaccard(set(left_tokens), set(right_tokens)) >= 0.8:
        return True
    return _sequence_ratio(left_text, right_text) >= 0.88


def _choice_sequence(choices):
    if isinstance(choices, dict):
        return [canonical_choice_text(choices.get(label)) for label in ["A", "B", "C", "D"]]
    if isinstance(choices, (list, tuple)):
        return [canonical_choice_text(choice) for choice in choices]
    return []


def has_same_choices_pattern(left_choices, right_choices):
    left = _choice_sequence(left_choices)
    right = _choice_sequence(right_choices)
    return bool(left) and left == right


def has_high_context_overlap(left, right):
    left_text = _normalize_context(left)
    right_text = _normalize_context(right)
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    if left_text in right_text or right_text in left_text:
        return True
    if _sequence_ratio(left_text, right_text) >= 0.86:
        return True

    left_ngrams = _token_ngrams(left_text, 3) or _token_ngrams(left_text, 2)
    right_ngrams = _token_ngrams(right_text, 3) or _token_ngrams(right_text, 2)
    if _jaccard(left_ngrams, right_ngrams) >= 0.5:
        return True

    left_chars = {left_text[index : index + 5] for index in range(max(len(left_text) - 4, 0))}
    right_chars = {right_text[index : index + 5] for index in range(max(len(right_text) - 4, 0))}
    return _jaccard(left_chars, right_chars) >= 0.45


def _empty_text(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _record_content(record):
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")

    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        user_message = messages[0] if isinstance(messages[0], dict) else {}
        content = user_message.get("content")
        if not _empty_text(content):
            return str(content)

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    context = metadata.get("context", record.get("context"))
    question = metadata.get("question", record.get("question"))
    choices = metadata.get("choices", record.get("choices"))
    if not _empty_text(context) and not _empty_text(question) and choices is not None:
        if isinstance(choices, dict):
            ordered_choices = [choices.get(label, "") for label in ["A", "B", "C", "D"]]
        else:
            ordered_choices = list(choices) if isinstance(choices, (list, tuple)) else []
        if len(ordered_choices) == 4:
            return (
                "Đọc đoạn văn sau và chọn đáp án đúng.\n\n"
                f"Đoạn văn: {context}\n\n"
                f"Câu hỏi: {question}\n"
                f"A. {ordered_choices[0]}\n"
                f"B. {ordered_choices[1]}\n"
                f"C. {ordered_choices[2]}\n"
                f"D. {ordered_choices[3]}"
            )

    raise ValueError("unable to extract MCQ user content")


def _extract_profile(record):
    content = _record_content(record)
    context, question, choices = split_mcq_user_content(content)
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    answer = metadata.get("answer")
    if _empty_text(answer):
        answer = record.get("answer")
    return {
        "source_id": metadata.get("source_id") or record.get("source_id") or record.get("id") or "unknown",
        "context": context,
        "question": question,
        "choices": {label: choice for label, choice in zip(["A", "B", "C", "D"], choices)},
        "answer": None if _empty_text(answer) else str(answer).strip().upper(),
        "record": record,
    }


def _ensure_profile(record):
    if isinstance(record, dict) and {"context", "question", "choices"}.issubset(record.keys()):
        normalized = dict(record)
        if "record" not in normalized:
            normalized["record"] = record
        return normalized
    return _extract_profile(record)


def _extract_answer_fact_value(profile):
    answer = profile.get("answer")
    if isinstance(answer, str) and answer.strip().upper() in ANSWER_LABELS:
        return ("label", answer.strip().upper())
    metadata = profile.get("record", {}).get("metadata") if isinstance(profile.get("record"), dict) else {}
    for key in ["gold_answer_text", "answer_text", "correct_answer", "fact_answer"]:
        value = metadata.get(key)
        if not _empty_text(value):
            return ("text", _normalize_text(value))
    return None


def _load_benchmark_records(path, allow_missing_benchmark=False):
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        if allow_missing_benchmark:
            return [], [], 0
        raise FileNotFoundError(f"missing benchmark path: {benchmark_path}")

    if benchmark_path.is_file():
        jsonl_paths = [benchmark_path]
    else:
        jsonl_paths = sorted(
            candidate for candidate in benchmark_path.rglob("*.jsonl") if candidate.is_file()
        )

    if not jsonl_paths:
        if allow_missing_benchmark:
            return [], [], 0
        raise FileNotFoundError(f"no benchmark JSONL files found under: {benchmark_path}")

    benchmark_records = []
    invalid_records = 0
    for jsonl_path in jsonl_paths:
        for row in read_jsonl(jsonl_path):
            try:
                benchmark_records.append(_extract_profile(row))
            except ValueError:
                invalid_records += 1
    return benchmark_records, jsonl_paths, invalid_records


def _build_reject_row(record, reason, matched_benchmark=None):
    metadata = record.get("metadata") if isinstance(record, dict) else {}
    reject_row = {
        "source_id": metadata.get("source_id") if isinstance(metadata, dict) else "unknown",
        "reason": reason,
    }
    if isinstance(matched_benchmark, dict):
        reject_row["matched_benchmark_source_id"] = matched_benchmark.get("source_id", "unknown")
    return reject_row


def _detect_leakage_reason(candidate, benchmark):
    if has_exact_question_match(candidate["question"], benchmark["question"]):
        return "exact_question_match"
    if is_near_duplicate_question(candidate["question"], benchmark["question"]):
        return "near_duplicate_question"
    if has_same_choices_pattern(candidate["choices"], benchmark["choices"]):
        return "same_choices_pattern"
    if has_high_context_overlap(candidate["context"], benchmark["context"]):
        return "high_context_overlap"

    candidate_answer = _extract_answer_fact_value(candidate)
    benchmark_answer = _extract_answer_fact_value(benchmark)
    if candidate_answer and benchmark_answer and candidate_answer == benchmark_answer:
        return "same_answer_fact_pattern"
    return None


def check_leakage(records, benchmark_records, benchmark_missing_allowed=False):
    kept = []
    rejects = []
    reject_reasons = Counter()
    kept_by_split = Counter()
    candidate_models = Counter()

    for record in records:
        metadata = record.get("metadata") if isinstance(record, dict) else {}
        try:
            candidate = _ensure_profile(record)
        except ValueError:
            rejects.append(_build_reject_row(record, "invalid_input_record"))
            reject_reasons["invalid_input_record"] += 1
            continue

        candidate_models[str(metadata.get("model", "unknown"))] += 1
        matched_reason = None
        matched_benchmark = None
        for benchmark in benchmark_records:
            try:
                benchmark_profile = _ensure_profile(benchmark)
            except ValueError:
                continue
            reason = _detect_leakage_reason(candidate, benchmark_profile)
            if reason is not None:
                matched_reason = reason
                matched_benchmark = benchmark_profile
                break

        if matched_reason is None:
            kept.append(record)
            kept_by_split[str(metadata.get("source_split", "unknown"))] += 1
            continue

        rejects.append(_build_reject_row(record, matched_reason, matched_benchmark))
        reject_reasons[matched_reason] += 1

    report = {
        "input_jsonl": None,
        "benchmark_jsonl_inputs": [],
        "output_jsonl": None,
        "rejects_jsonl": None,
        "report_json": None,
        "total_loaded": len(records),
        "benchmark_records_loaded": len(benchmark_records),
        "invalid_benchmark_records": 0,
        "benchmark_missing_allowed": benchmark_missing_allowed,
        "total_kept": len(kept),
        "total_rejected": len(rejects),
        "kept_by_split": dict(kept_by_split),
        "candidate_models": dict(candidate_models),
        "reject_reasons": dict(reject_reasons),
    }
    return kept, rejects, report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--rejects-jsonl", default=DEFAULT_REJECTS_JSONL)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--allow-missing-benchmark", action="store_true")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] missing input file: {input_path}")
        return 1

    try:
        records = read_jsonl(input_path)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1

    try:
        benchmark_records, benchmark_jsonl_paths, invalid_benchmark_records = _load_benchmark_records(
            args.benchmark,
            allow_missing_benchmark=args.allow_missing_benchmark,
        )
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1

    kept, rejects, report = check_leakage(
        records,
        benchmark_records,
        benchmark_missing_allowed=args.allow_missing_benchmark,
    )
    report["benchmark_jsonl_inputs"] = [str(path) for path in benchmark_jsonl_paths]
    report["invalid_benchmark_records"] = invalid_benchmark_records

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
            "input_jsonl": str(input_path),
            "output_jsonl": str(output_path),
            "rejects_jsonl": str(rejects_path),
            "report_json": str(report_path),
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
