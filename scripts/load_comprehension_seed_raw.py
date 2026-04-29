import hashlib
import json
import sys
from collections import Counter
from numbers import Integral
import re
from pathlib import Path


MIN_CONTEXT_CHARS = 80
MAX_CONTEXT_CHARS = 8000
MIN_QUESTION_CHARS = 5
MAX_QUESTION_CHARS = 400
MAX_ANSWER_CHARS = 300

MISSING_CONTEXT = "missing_context"
MISSING_QUESTION = "missing_question"
MISSING_ANSWER = "missing_answer"

TASK_NAME = "comprehension_raw"
QC_VERSION = "comprehension_raw_source_specific_span_v1"
UIT_DATASET_ID = "taidng/UIT-ViQuAD2.0"
SHYNBUI_DATASET_ID = "ShynBui/Vietnamese_Reading_Comprehension_Dataset"

CONTEXT_TOO_SHORT = "context_too_short"
CONTEXT_TOO_LONG = "context_too_long"
QUESTION_TOO_SHORT = "question_too_short"
QUESTION_TOO_LONG = "question_too_long"
ANSWER_TOO_LONG = "answer_too_long"


def normalize_for_hash(text):
    if text is None:
        return ""
    normalized = str(text).replace("\u00a0", " ").lower().strip()
    return re.sub(r"\s+", " ", normalized)


def normalize_shynbui_text(text):
    if text is None or not isinstance(text, str):
        raise TypeError("ShynBui text fields must be strings")
    normalized = text.replace("\u00a0", " ").replace("_", " ").strip()
    if not normalized:
        raise ValueError("ShynBui text fields must be non-empty")
    return normalized


def recompute_answer_start(context, answer_text):
    if context is None or answer_text is None:
        raise TypeError("context and answer_text must be strings")
    if not isinstance(context, str) or not isinstance(answer_text, str):
        raise TypeError("context and answer_text must be strings")
    return context.find(answer_text)


def make_dedup_hash(context, question, answer_text):
    payload = "\n".join(
        [
            normalize_for_hash(context),
            normalize_for_hash(question),
            normalize_for_hash(answer_text),
        ]
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def build_user_content(context, question):
    return f"Đoạn văn: {context}\n\nCâu hỏi: {question}"


def apply_length_filters(context, question, answer_text):
    if context is None or not isinstance(context, str):
        return MISSING_CONTEXT
    if len(context) < MIN_CONTEXT_CHARS:
        return CONTEXT_TOO_SHORT
    if len(context) > MAX_CONTEXT_CHARS:
        return CONTEXT_TOO_LONG
    if question is None or not isinstance(question, str):
        return MISSING_QUESTION
    if len(question) < MIN_QUESTION_CHARS:
        return QUESTION_TOO_SHORT
    if len(question) > MAX_QUESTION_CHARS:
        return QUESTION_TOO_LONG
    if answer_text is None or not isinstance(answer_text, str):
        return MISSING_ANSWER
    if len(answer_text) > MAX_ANSWER_CHARS:
        return ANSWER_TOO_LONG
    return None


def strict_span_match(context, answer_text, answer_start):
    if not isinstance(context, str) or not isinstance(answer_text, str):
        raise TypeError("context and answer_text must be strings")
    if not isinstance(answer_start, Integral) or isinstance(answer_start, bool):
        raise TypeError("answer_start must be an integer")
    if answer_start < 0:
        raise ValueError("answer_start must be non-negative")
    end = answer_start + len(answer_text)
    if end > len(context):
        return False
    return context[answer_start:end] == answer_text


def extract_valid_answer_variants(context, answers_text, answers_start):
    variants = []
    if not isinstance(answers_text, (list, tuple)):
        raise ValueError("answers_text must be a list or tuple")
    if not isinstance(answers_start, (list, tuple)):
        raise ValueError("answers_start must be a list or tuple")
    if len(answers_text) != len(answers_start):
        raise ValueError("answers_text and answers_start must have the same length")

    for text, start in zip(answers_text, answers_start):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("answer text must be a non-empty string")
        if not isinstance(start, Integral) or isinstance(start, bool) or start < 0:
            raise ValueError("answer_start must be a non-negative integer")
        if strict_span_match(context, text, start):
            variants.append({"text": text, "answer_start": start})
    return variants


def select_primary_variant(variants):
    return variants[0] if variants else None


def dedup_records(records):
    deduped = []
    duplicates = []
    seen_hashes = set()

    for record in records:
        metadata = record.get("metadata") if isinstance(record, dict) else None
        if not isinstance(metadata, dict) or "dedup_hash" not in metadata:
            source_id = "unknown"
            if isinstance(metadata, dict):
                source_id = metadata.get("source_id", "unknown")
            raise ValueError(f"missing metadata.dedup_hash for source_id={source_id}")

        dedup_hash = metadata["dedup_hash"]
        source_id = metadata.get("source_id", "unknown")
        if not isinstance(dedup_hash, str) or not dedup_hash:
            raise ValueError(f"invalid metadata.dedup_hash for source_id={source_id}")
        if dedup_hash in seen_hashes:
            duplicates.append(record)
            continue
        seen_hashes.add(dedup_hash)
        deduped.append(record)

    return deduped, duplicates


def build_record(
    *,
    context,
    question,
    answer_text,
    answer_start,
    source_dataset,
    source_split,
    source_id,
    title,
    answer_variants,
    span_check_mode,
    original_answer_start=None,
):
    return {
        "messages": [
            {"role": "user", "content": build_user_content(context, question)},
            {"role": "assistant", "content": answer_text},
        ],
        "context": context,
        "question": question,
        "answer_text": answer_text,
        "metadata": {
            "task": TASK_NAME,
            "source": "public",
            "source_dataset": source_dataset,
            "source_split": source_split,
            "source_id": source_id,
            "answer_start": answer_start,
            "original_answer_start": answer_start if original_answer_start is None else original_answer_start,
            "answer_text": answer_text,
            "answer_variants": answer_variants,
            "title": title,
            "language": "vi",
            "difficulty": "medium",
            "span_check_mode": span_check_mode,
            "dedup_hash": make_dedup_hash(context, question, answer_text),
            "qc_version": QC_VERSION,
        },
    }


def build_reject_record(
    *,
    source_dataset,
    source_split,
    source_id,
    reason,
    context,
    question,
    answer,
    answer_start,
): 
    context_preview = "" if context is None else str(context)[:200]
    return {
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_id": source_id,
        "reason": reason,
        "context_preview": context_preview,
        "question": question,
        "answer": answer,
        "answer_start": answer_start,
    }


def _safe_text(value):
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _source_row_id(row, split_name, row_index):
    if isinstance(row, dict):
        for key in ("id", "uid", "source_id"):
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
    return f"{split_name}-{row_index}"


def _build_row_reject_record(dataset_id, split_name, row_index, row, reason):
    if isinstance(row, dict):
        context = row.get("context")
        question = row.get("question")
        answer = row.get("answer")
        answer_start = row.get("answer_start")
        answers = row.get("answers")
        if answer is None and isinstance(answers, dict):
            raw_answers = answers.get("text")
            if isinstance(raw_answers, (list, tuple)) and raw_answers:
                answer = raw_answers[0]
            elif raw_answers is not None and not isinstance(raw_answers, (list, tuple)):
                answer = raw_answers
        if answer_start is None and isinstance(answers, dict):
            raw_starts = answers.get("answer_start")
            if isinstance(raw_starts, (list, tuple)) and raw_starts:
                answer_start = raw_starts[0]
            elif raw_starts is not None and not isinstance(raw_starts, (list, tuple)):
                answer_start = raw_starts
    else:
        context = None
        question = None
        answer = None
        answer_start = None

    return build_reject_record(
        source_dataset=dataset_id,
        source_split=split_name,
        source_id=_source_row_id(row, split_name, row_index),
        reason=reason,
        context=_safe_text(context),
        question=_safe_text(question),
        answer=_safe_text(answer),
        answer_start=answer_start,
    )


def _collect_valid_uit_variants(context, answers_text, answers_start):
    variants = []
    if not isinstance(context, str):
        return variants
    if not isinstance(answers_text, (list, tuple)) or not isinstance(answers_start, (list, tuple)):
        return variants
    if len(answers_text) != len(answers_start):
        return variants

    for text, start in zip(answers_text, answers_start):
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(start, Integral) or isinstance(start, bool) or start < 0:
            continue
        try:
            if strict_span_match(context, text, start):
                variants.append({"text": text, "answer_start": start})
        except (TypeError, ValueError):
            continue
    return variants


def process_uit_row(row, split_name):
    if not isinstance(row, dict):
        return None, "schema_error"

    if bool(row.get("is_impossible")):
        return None, "is_impossible"

    context = row.get("context")
    question = row.get("question")
    answers = row.get("answers")
    if not isinstance(answers, dict):
        return None, "missing_field"

    answers_text = answers.get("text")
    answers_start = answers.get("answer_start")
    if answers_text is None or answers_start is None:
        return None, "missing_field"
    if not isinstance(answers_text, (list, tuple)) or not isinstance(answers_start, (list, tuple)):
        return None, "schema_error"

    length_reason = apply_length_filters(context, question, "")
    if length_reason:
        return None, length_reason

    variants = _collect_valid_uit_variants(context, answers_text, answers_start)

    if not variants:
        return None, "no_valid_answer_variant"

    primary = None
    selected_length_reason = None
    for variant in variants:
        answer_length_reason = apply_length_filters(context, question, variant["text"])
        if answer_length_reason is None:
            primary = variant
            break
        if selected_length_reason is None:
            selected_length_reason = answer_length_reason

    if primary is None:
        return None, selected_length_reason or "no_valid_answer_variant"

    source_id = _source_row_id(row, split_name, row.get("_row_index", 0))
    record = build_record(
        context=context,
        question=question,
        answer_text=primary["text"],
        answer_start=primary["answer_start"],
        source_dataset=UIT_DATASET_ID,
        source_split=split_name,
        source_id=source_id,
        title=row.get("title"),
        answer_variants=variants,
        span_check_mode="strict_exact",
        original_answer_start=primary["answer_start"],
    )
    return record, None


def process_shynbui_row(row, split_name):
    if not isinstance(row, dict):
        return None, "schema_error"

    context_raw = row.get("context")
    question_raw = row.get("question")
    answer_raw = row.get("answer")
    answer_start_raw = row.get("answer_start")

    length_reason = apply_length_filters(context_raw, question_raw, answer_raw)
    if length_reason:
        return None, length_reason
    if not isinstance(answer_raw, str) or not answer_raw.strip():
        return None, "missing_answer"

    if not isinstance(answer_start_raw, Integral) or isinstance(answer_start_raw, bool):
        return None, "invalid_answer_start"
    answer_start_raw = int(answer_start_raw)
    if answer_start_raw < 0 or answer_start_raw + len(answer_raw) > len(context_raw):
        return None, "answer_start_out_of_range"
    try:
        if not strict_span_match(context_raw, answer_raw, answer_start_raw):
            return None, "span_mismatch"
    except (TypeError, ValueError):
        return None, "schema_error"

    try:
        context_normalized = context_raw.replace("\u00a0", " ").replace("_", " ")
        leading_trim = len(context_normalized) - len(context_normalized.lstrip())
        context = context_normalized.strip()
        question = normalize_shynbui_text(question_raw)
        answer_text = normalize_shynbui_text(answer_raw)
    except (TypeError, ValueError):
        return None, "schema_error"

    answer_start = answer_start_raw - leading_trim
    if answer_start < 0 or answer_start + len(answer_text) > len(context) or context[answer_start : answer_start + len(answer_text)] != answer_text:
    if answer_start + len(answer_text) > len(context) or context[answer_start : answer_start + len(answer_text)] != answer_text:
        found = context.find(answer_text, max(0, answer_start))
        if found < 0 or context[found : found + len(answer_text)] != answer_text:
            return None, "answer_not_found_after_normalize"
        answer_start = found

    record = build_record(
        context=context,
        question=question,
        answer_text=answer_text,
        answer_start=answer_start,
        source_dataset=SHYNBUI_DATASET_ID,
        source_split=split_name,
        source_id=_source_row_id(row, split_name, row.get("_row_index", 0)),
        title=row.get("title"),
        answer_variants=[],
        span_check_mode="raw_exact_then_normalized_find",
        original_answer_start=answer_start_raw,
    )
    return record, None


def _dedup_records_with_rejects(records):
    try:
        deduped, duplicates = dedup_records(records)
        return deduped, duplicates, []
    except ValueError:
        valid_records = []
        reject_records = []
        for record in records:
            metadata = record.get("metadata") if isinstance(record, dict) else None
            dedup_hash = metadata.get("dedup_hash") if isinstance(metadata, dict) else None
            if isinstance(dedup_hash, str) and dedup_hash:
                valid_records.append(record)
                continue
            reject_records.append(
                build_reject_record(
                    source_dataset=metadata.get("source_dataset", "unknown") if isinstance(metadata, dict) else "unknown",
                    source_split=metadata.get("source_split", "unknown") if isinstance(metadata, dict) else "unknown",
                    source_id=metadata.get("source_id", "unknown") if isinstance(metadata, dict) else "unknown",
                    reason="schema_error",
                    context=record.get("context") if isinstance(record, dict) else "",
                    question=record.get("question") if isinstance(record, dict) else "",
                    answer=record.get("answer_text") if isinstance(record, dict) else "",
                    answer_start=metadata.get("answer_start") if isinstance(metadata, dict) else None,
                )
            )

        deduped, duplicates = dedup_records(valid_records)
        return deduped, duplicates, reject_records


def _counts_from_records(records, field_name):
    counts = Counter()
    for record in records:
        metadata = record.get("metadata", {})
        value = metadata.get(field_name)
        if value is not None:
            counts[str(value)] += 1
    return dict(counts)


def run_loader_pipeline(source_datasets):
    kept = []
    rejects = []
    reject_reasons = Counter()
    total_loaded = 0

    for dataset_id, splits in source_datasets.items():
        processor = process_uit_row if dataset_id == UIT_DATASET_ID else process_shynbui_row
        for split_name, rows in splits.items():
            for row_index, row in enumerate(rows):
                total_loaded += 1
                processor_row = dict(row) if isinstance(row, dict) else row
                if isinstance(processor_row, dict):
                    processor_row["_row_index"] = row_index
                record, reason = processor(processor_row, split_name)
                if reason:
                    reject_reasons[reason] += 1
                    rejects.append(_build_row_reject_record(dataset_id, split_name, row_index, row, reason))
                    continue
                kept.append(record)

    deduped, duplicates, bad_hash_rejects = _dedup_records_with_rejects(kept)
    for reject in bad_hash_rejects:
        reject_reasons[reject["reason"]] += 1
    for record in duplicates:
        reject_reasons["duplicate"] += 1
        metadata = record["metadata"]
        rejects.append(
            build_reject_record(
                source_dataset=metadata["source_dataset"],
                source_split=metadata["source_split"],
                source_id=metadata["source_id"],
                reason="duplicate",
                context=record["context"],
                question=record["question"],
                answer=record["answer_text"],
                answer_start=metadata["answer_start"],
            )
        )

    rejects.extend(bad_hash_rejects)

    report = {
        "total_loaded": total_loaded,
        "total_kept": len(deduped),
        "total_rejected": len(rejects),
        "kept_by_dataset": _counts_from_records(deduped, "source_dataset"),
        "kept_by_split": _counts_from_records(deduped, "source_split"),
        "reject_reasons": dict(reject_reasons),
        "duplicate_count": len(duplicates),
        "split_policy": "raw_pool_all_source_splits_loaded; downstream split required",
        "qc_version": QC_VERSION,
    }

    return {"kept": deduped, "rejects": rejects, "report": report}


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    try:
        from datasets import load_dataset
        from dotenv import load_dotenv
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:
        print(f"[error] missing runtime dependency: {exc}")
        return 1

    load_dotenv()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "seed_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "comprehension_seed_raw.jsonl"
    out_rejects_jsonl = out_dir / "comprehension_seed_raw_rejects.jsonl"
    out_report_json = out_dir / "comprehension_seed_raw_report.json"

    source_datasets = {}
    for dataset_id in (UIT_DATASET_ID, SHYNBUI_DATASET_ID):
        print(f"[load] {dataset_id}")
        try:
            dataset_dict = load_dataset(dataset_id)
        except Exception as exc:
            print(f"[error] failed to load {dataset_id}: {exc}")
            return 1

        source_datasets[dataset_id] = {}
        for split_name in dataset_dict.keys():
            source_datasets[dataset_id][split_name] = list(dataset_dict[split_name])

    pipeline = run_loader_pipeline(source_datasets)

    _write_jsonl(out_jsonl, tqdm(pipeline["kept"], desc="write kept", disable=True))
    _write_jsonl(out_rejects_jsonl, tqdm(pipeline["rejects"], desc="write rejects", disable=True))
    with out_report_json.open("w", encoding="utf-8") as handle:
        json.dump(pipeline["report"], handle, ensure_ascii=False, indent=2)

    print(f"[ok] wrote {out_jsonl}")
    print(f"[ok] wrote {out_rejects_jsonl}")
    print(f"[ok] wrote {out_report_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
