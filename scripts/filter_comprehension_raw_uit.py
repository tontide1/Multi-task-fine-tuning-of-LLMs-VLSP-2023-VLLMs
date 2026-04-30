import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_INPUT_JSONL = Path(__file__).resolve().parent.parent / "seed_exports" / "comprehension_seed_raw.jsonl"
DEFAULT_OUTPUT_JSONL = Path(__file__).resolve().parent.parent / "seed_exports" / "comprehension_seed_raw_uit_only.jsonl"
DEFAULT_REJECTS_JSONL = Path(__file__).resolve().parent.parent / "seed_exports" / "comprehension_seed_raw_uit_only_rejects.jsonl"
DEFAULT_REPORT_JSON = Path(__file__).resolve().parent.parent / "seed_exports" / "comprehension_seed_raw_uit_only_report.json"

UIT_DATASET_ID = "taidng/UIT-ViQuAD2.0"
STRICT_SPAN_CHECK_MODE = "strict_exact"


def read_jsonl(path):
    records = []
    malformed_rejects = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed_rejects.append(
                        {
                            "source_dataset": "unknown",
                            "source_split": "unknown",
                            "source_id": f"line-{line_number}",
                            "reason": "invalid_json",
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


def _source_id(record):
    metadata = record.get("metadata") if isinstance(record, dict) else None
    if not isinstance(metadata, dict):
        return "unknown"
    for key in ("source_id", "id", "uid"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def _source_split(record):
    metadata = record.get("metadata") if isinstance(record, dict) else None
    if not isinstance(metadata, dict):
        return "unknown"
    value = metadata.get("source_split")
    return "unknown" if value in (None, "") else str(value)


def _source_dataset(record):
    metadata = record.get("metadata") if isinstance(record, dict) else None
    if not isinstance(metadata, dict):
        return "unknown"
    value = metadata.get("source_dataset")
    return "unknown" if value in (None, "") else str(value)


def _context_preview(record):
    if not isinstance(record, dict):
        return ""
    context = record.get("context")
    return "" if context is None else str(context)[:200]


def _question(record):
    if not isinstance(record, dict):
        return ""
    value = record.get("question")
    return "" if value is None else str(value)


def _answer_text(record):
    if not isinstance(record, dict):
        return ""
    value = record.get("answer_text")
    return "" if value is None else str(value)


def _filter_reason(record):
    metadata = record.get("metadata") if isinstance(record, dict) else None
    if not isinstance(metadata, dict):
        return "missing_metadata"
    if metadata.get("source_dataset") != UIT_DATASET_ID:
        return "source_dataset"
    if metadata.get("span_check_mode") != STRICT_SPAN_CHECK_MODE:
        return "span_check_mode"
    answer_variants = metadata.get("answer_variants")
    if not isinstance(answer_variants, list) or not answer_variants:
        return "answer_variants_empty"
    return None


def build_reject_record(record, reason):
    return {
        "source_dataset": _source_dataset(record),
        "source_split": _source_split(record),
        "source_id": _source_id(record),
        "reason": reason,
        "context_preview": _context_preview(record),
        "question": _question(record),
        "answer_text": _answer_text(record),
    }


def filter_uit_only(records):
    kept = []
    rejects = []
    reject_reasons = Counter()
    kept_by_split = Counter()
    source_datasets = Counter()
    span_check_modes = Counter()
    empty_answer_variants = 0

    for record in records:
        reason = _filter_reason(record)
        if reason is None:
            kept.append(record)
            kept_by_split[_source_split(record)] += 1
            source_datasets[_source_dataset(record)] += 1
            span_check_modes[str(record["metadata"].get("span_check_mode"))] += 1
            continue
        reject_reasons[reason] += 1
        if reason == "answer_variants_empty":
            empty_answer_variants += 1
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
        "source_datasets": dict(source_datasets),
        "span_check_modes": dict(span_check_modes),
        "empty_answer_variants": empty_answer_variants,
        "invalid_records": 0,
        "reject_reasons": dict(reject_reasons),
        "filter": {
            "source_dataset": UIT_DATASET_ID,
            "span_check_mode": STRICT_SPAN_CHECK_MODE,
            "requires_non_empty_answer_variants": True,
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
    kept, rejects, report = filter_uit_only(records)
    rejects = malformed_rejects + rejects
    report["invalid_records"] = len(malformed_rejects)
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
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"[ok] wrote {output_path}")
    print(f"[ok] wrote {rejects_path}")
    print(f"[ok] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
