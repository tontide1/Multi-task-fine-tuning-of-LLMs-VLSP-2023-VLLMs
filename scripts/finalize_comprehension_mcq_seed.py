import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.recheck_comprehension_mcq_seed import validate_record_schema  # noqa: E402


DEFAULT_INPUT_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_no_leak.jsonl"
DEFAULT_OUTPUT_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_final.jsonl"
DEFAULT_REPORT_JSON = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_final_report.json"
DEFAULT_SAMPLES_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_final_samples.jsonl"


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
    return records


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _metadata(record):
    if not isinstance(record, dict):
        return {}
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _final_path_string(path):
    return str(Path(path))


def finalize_records(records, sample_size=5):
    kept = []
    seen_mcq_dedup_hashes = set()
    duplicate_mcq_dedup_hashes = []
    source_datasets = Counter()
    source_splits = Counter()
    answer_labels = Counter()

    for index, record in enumerate(records, start=1):
        errors = validate_record_schema(record)
        if errors:
            raise ValueError(f"invalid record at line {index}: {', '.join(errors)}")

        metadata = _metadata(record)
        dedup_hash = metadata.get("mcq_dedup_hash")
        if dedup_hash in seen_mcq_dedup_hashes:
            duplicate_mcq_dedup_hashes.append(dedup_hash)
            raise ValueError(f"duplicate mcq_dedup_hash at line {index}: {dedup_hash}")
        seen_mcq_dedup_hashes.add(dedup_hash)

        kept.append(record)
        source_datasets[str(metadata.get("source_dataset", "unknown"))] += 1
        source_splits[str(metadata.get("source_split", "unknown"))] += 1
        answer_labels[str(metadata.get("answer", "unknown"))] += 1

    samples = kept[: max(sample_size, 0)]
    report = {
        "input_jsonl": None,
        "output_jsonl": _final_path_string(DEFAULT_OUTPUT_JSONL),
        "report_json": _final_path_string(DEFAULT_REPORT_JSON),
        "samples_jsonl": _final_path_string(DEFAULT_SAMPLES_JSONL),
        "total_loaded": len(records),
        "total_kept": len(kept),
        "total_rejected": 0,
        "duplicate_mcq_dedup_hashes": duplicate_mcq_dedup_hashes,
        "source_dataset_distribution": dict(source_datasets),
        "source_split_distribution": dict(source_splits),
        "answer_label_distribution": dict(answer_labels),
    }
    return kept, report, samples


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--samples-jsonl", default=DEFAULT_SAMPLES_JSONL)
    parser.add_argument("--sample-size", type=int, default=5)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] missing input file: {input_path}")
        return 1

    try:
        records = read_jsonl(input_path)
        final_records, report, samples = finalize_records(records, sample_size=args.sample_size)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1

    output_path = Path(args.output_jsonl)
    report_path = Path(args.report_json)
    samples_path = Path(args.samples_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_path, final_records)
    write_jsonl(samples_path, samples)

    report.update(
        {
            "input_jsonl": str(input_path),
            "output_jsonl": str(output_path),
            "report_json": str(report_path),
            "samples_jsonl": str(samples_path),
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
