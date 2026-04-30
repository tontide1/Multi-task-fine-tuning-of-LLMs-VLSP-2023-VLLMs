import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.comprehension_mcq_seed_common import compute_context_hash


DEFAULT_INPUT_JSONL = Path(__file__).resolve().parent.parent / "seed_exports" / "comprehension_seed_raw_uit_only.jsonl"
DEFAULT_OUTPUT_JSONL = Path(__file__).resolve().parent.parent / "seed_exports" / "comprehension_mcq_generation_requests.jsonl"
DEFAULT_REJECTS_JSONL = (
    Path(__file__).resolve().parent.parent / "seed_exports" / "comprehension_mcq_generation_request_rejects.jsonl"
)
DEFAULT_REPORT_JSON = Path(__file__).resolve().parent.parent / "seed_exports" / "comprehension_mcq_generation_request_report.json"

MIN_ANSWER_CHARS = 2
MAX_ANSWER_CHARS = 220
MAX_CONTEXT_CHARS = 8000

GENERATION_PROMPT_VERSION = "comprehension_mcq_distractors_v1"
FILTER_VERSION = "comprehension_mcq_generation_filter_v1"


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


def _string_field(record, key):
    if not isinstance(record, dict):
        return None
    value = record.get(key)
    return value if isinstance(value, str) else None


def _metadata(record):
    if not isinstance(record, dict):
        return None
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else None


def _preview(value):
    return "" if value is None else str(value)[:200]


def _answer_variants(metadata):
    variants = metadata.get("answer_variants") if isinstance(metadata, dict) else None
    if not isinstance(variants, list):
        return []
    texts = []
    for variant in variants:
        if isinstance(variant, dict) and isinstance(variant.get("text"), str):
            texts.append(variant["text"])
    return texts


def should_keep_for_generation(record):
    context = _string_field(record, "context")
    question = _string_field(record, "question")
    answer_text = _string_field(record, "answer_text")

    if context is None or not context.strip():
        return False
    if question is None or not question.strip():
        return False
    if answer_text is None or not answer_text.strip():
        return False
    if len(answer_text) < MIN_ANSWER_CHARS:
        return False
    if len(answer_text) > MAX_ANSWER_CHARS:
        return False
    if len(context) > MAX_CONTEXT_CHARS:
        return False
    return True


def build_generation_request(record):
    metadata = _metadata(record)
    source_id = metadata.get("source_id") if isinstance(metadata, dict) else None
    raw_dedup_hash = metadata.get("dedup_hash") if isinstance(metadata, dict) else None
    request_suffix = source_id if isinstance(source_id, str) and source_id else raw_dedup_hash
    if not isinstance(request_suffix, str) or not request_suffix:
        request_suffix = compute_context_hash(record["context"])

    return {
        "request_id": f"cmcq-gen-{request_suffix}",
        "source_id": source_id,
        "source_split": metadata.get("source_split") if isinstance(metadata, dict) else None,
        "context": record["context"],
        "question": record["question"],
        "gold_answer_text": record["answer_text"],
        "answer_variants": _answer_variants(metadata),
        "title": metadata.get("title") if isinstance(metadata, dict) else None,
        "raw_dedup_hash": raw_dedup_hash,
        "context_hash": compute_context_hash(record["context"]),
        "generation_prompt_version": GENERATION_PROMPT_VERSION,
        "filter_version": FILTER_VERSION,
    }


def _filter_reason(record):
    if not isinstance(record, dict):
        return "invalid_raw_record"
    context = record.get("context")
    question = record.get("question")
    answer_text = record.get("answer_text")
    if not isinstance(context, str) or not context.strip():
        return "missing_context"
    if not isinstance(question, str) or not question.strip():
        return "missing_question"
    if not isinstance(answer_text, str) or not answer_text.strip():
        return "missing_answer"
    if len(answer_text) < MIN_ANSWER_CHARS:
        return "answer_too_short"
    if len(answer_text) > MAX_ANSWER_CHARS:
        return "answer_too_long"
    if len(context) > MAX_CONTEXT_CHARS:
        return "context_too_long"
    return None


def build_reject_record(record, reason):
    metadata = _metadata(record)
    source_id = metadata.get("source_id") if isinstance(metadata, dict) else None
    source_split = metadata.get("source_split") if isinstance(metadata, dict) else None
    return {
        "source_id": source_id if source_id is not None else "unknown",
        "source_split": source_split if source_split is not None else "unknown",
        "reason": reason,
        "context_preview": _preview(_string_field(record, "context")),
        "question": _preview(_string_field(record, "question")),
        "answer_text": _preview(_string_field(record, "answer_text")),
    }


def filter_generation_requests(records):
    kept = []
    rejects = []
    reject_reasons = Counter()
    kept_by_split = Counter()

    for record in records:
        reason = _filter_reason(record)
        if reason is None:
            kept.append(record)
            metadata = _metadata(record)
            source_split = metadata.get("source_split") if isinstance(metadata, dict) else None
            kept_by_split[str(source_split) if source_split not in (None, "") else "unknown"] += 1
            continue
        reject_reasons[reason] += 1
        rejects.append(build_reject_record(record, reason))

    report = {
        "input_jsonl": None,
        "output_jsonl": None,
        "rejects_jsonl": None,
        "report_json": None,
        "total_loaded": len(records),
        "total_kept": len(kept),
        "total_rejected": len(rejects),
        "kept_by_split": dict(kept_by_split),
        "reject_reasons": dict(reject_reasons),
        "filter": {
            "min_answer_chars": MIN_ANSWER_CHARS,
            "max_answer_chars": MAX_ANSWER_CHARS,
            "max_context_chars": MAX_CONTEXT_CHARS,
            "generation_prompt_version": GENERATION_PROMPT_VERSION,
            "filter_version": FILTER_VERSION,
        },
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
    kept, rejects, report = filter_generation_requests(records)
    rejects = malformed_rejects + rejects
    report["total_loaded"] = len(records) + len(malformed_rejects)
    report["total_rejected"] = len(rejects)
    if malformed_rejects:
        report["reject_reasons"]["invalid_json"] = len(malformed_rejects)

    output_path = Path(args.output_jsonl)
    rejects_path = Path(args.rejects_jsonl)
    report_path = Path(args.report_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_path, [build_generation_request(record) for record in kept])
    write_jsonl(rejects_path, rejects)

    report.update(
        {
            "input_jsonl": str(input_path),
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
