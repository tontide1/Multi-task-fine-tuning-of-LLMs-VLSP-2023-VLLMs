import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.comprehension_mcq_seed_common import split_mcq_user_content  # noqa: E402


DEFAULT_INPUT_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_rule_checked.jsonl"
DEFAULT_OUTPUT_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_solver_requests.jsonl"
DEFAULT_REJECTS_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_solver_request_rejects.jsonl"
DEFAULT_REPORT_JSON = _REPO_ROOT / "seed_exports" / "comprehension_mcq_solver_request_report.json"

SOLVER_PROMPT_VERSION = "comprehension_mcq_solver_v1"


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


def _metadata(record):
    if not isinstance(record, dict):
        return {}
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _reject_row(record, reason):
    metadata = _metadata(record)
    return {
        "reason": reason,
        "source_id": metadata.get("source_id", "unknown"),
        "source_split": metadata.get("source_split", "unknown"),
    }


def build_solver_request(record):
    metadata = _metadata(record)
    user_content = record["messages"][0]["content"]
    context, question, _choices = split_mcq_user_content(user_content)
    return {
        "mcq_dedup_hash": metadata["mcq_dedup_hash"],
        "source_id": metadata["source_id"],
        "source_split": metadata.get("source_split"),
        "context": context,
        "question": question,
        "choices": metadata["choices"],
        "gold_answer": metadata["answer"],
        "solver_prompt_version": SOLVER_PROMPT_VERSION,
        "request_id": metadata["mcq_dedup_hash"],
    }


def filter_solver_requests(records):
    kept = []
    rejects = []
    reject_reasons = Counter()
    kept_by_split = Counter()

    for record in records:
        metadata = _metadata(record)
        if metadata.get("task") != "comprehension_mcq":
            reject_reasons["invalid_task"] += 1
            rejects.append(_reject_row(record, "invalid_task"))
            continue
        if metadata.get("source") != "synthetic":
            reject_reasons["invalid_source"] += 1
            rejects.append(_reject_row(record, "invalid_source"))
            continue
        if metadata.get("source_dataset") != "taidng/UIT-ViQuAD2.0":
            reject_reasons["invalid_source_dataset"] += 1
            rejects.append(_reject_row(record, "invalid_source_dataset"))
            continue
        if metadata.get("answer") not in {"A", "B", "C", "D"}:
            reject_reasons["invalid_answer_label"] += 1
            rejects.append(_reject_row(record, "invalid_answer_label"))
            continue
        if not isinstance(record.get("messages"), list) or len(record["messages"]) != 2:
            reject_reasons["invalid_messages"] += 1
            rejects.append(_reject_row(record, "invalid_messages"))
            continue

        kept.append(record)
        source_split = metadata.get("source_split")
        kept_by_split[str(source_split) if source_split not in (None, "") else "unknown"] += 1

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
        "solver_prompt_version": SOLVER_PROMPT_VERSION,
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
    kept, rejects, report = filter_solver_requests(records)
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

    write_jsonl(output_path, [build_solver_request(record) for record in kept])
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
