import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.comprehension_mcq_seed_common import extract_json_object  # noqa: E402


DEFAULT_INPUT_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_rule_checked.jsonl"
DEFAULT_OUTPUT_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_solver_checked.jsonl"
DEFAULT_REJECTS_JSONL = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_solver_rejects.jsonl"
DEFAULT_REPORT_JSON = _REPO_ROOT / "seed_exports" / "comprehension_mcq_seed_solver_report.json"

VALID_ANSWER_RE = re.compile(r"^[ABCD]$")


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


def _reject_row(record, reason, raw_preview=None):
    metadata = _metadata(record)
    return {
        "reason": reason,
        "source_id": metadata.get("source_id", "unknown"),
        "source_split": metadata.get("source_split", "unknown"),
        "mcq_dedup_hash": metadata.get("mcq_dedup_hash"),
        "raw_preview": raw_preview,
    }


def parse_solver_output(raw_text):
    parsed = raw_text if isinstance(raw_text, dict) else extract_json_object(raw_text)
    if not isinstance(parsed, dict):
        raise TypeError("solver output must be a JSON object")
    if "predicted_answer" not in parsed:
        raise ValueError("missing predicted_answer")
    return parsed


def solver_keep_decision(record, parsed):
    metadata = _metadata(record)
    return (
        parsed.get("predicted_answer") == metadata.get("answer")
        and parsed.get("is_unambiguous") is True
        and not parsed.get("bad_reason")
    )


def _solver_output_text(row):
    if not isinstance(row, dict):
        return None
    for key in ("raw_response", "response_text", "response", "content", "output"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
    parsed = row.get("parsed_response")
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    return None


def _join_key_candidates(record):
    metadata = _metadata(record)
    keys = []
    value = metadata.get("mcq_dedup_hash")
    if isinstance(value, str) and value:
        keys.append(value)
    value = metadata.get("source_id")
    if isinstance(value, str) and value:
        keys.append(value)
    return keys


def build_solver_checked_record(record, parsed):
    return {
        **record,
        "metadata": {
            **_metadata(record),
            "solver_predicted_answer": parsed.get("predicted_answer"),
            "solver_is_unambiguous": parsed.get("is_unambiguous"),
            "solver_bad_reason": parsed.get("bad_reason"),
        },
    }


def _find_solver_row(index, record):
    for key in _join_key_candidates(record):
        row = index.get(key)
        if row is not None:
            return row
    return None


def process_records(records, solver_rows):
    index = {}
    for row in solver_rows:
        if not isinstance(row, dict):
            continue
        for key in ("mcq_dedup_hash", "source_id"):
            value = row.get(key)
            if isinstance(value, str) and value:
                index.setdefault(value, row)

    kept = []
    rejects = []
    reject_reasons = Counter()
    kept_by_split = Counter()

    for record in records:
        solver_row = _find_solver_row(index, record)
        if solver_row is None:
            reject_reasons["missing_solver_output"] += 1
            rejects.append(_reject_row(record, "missing_solver_output"))
            continue

        raw_text = _solver_output_text(solver_row)
        if raw_text is None:
            reject_reasons["missing_solver_output"] += 1
            rejects.append(_reject_row(record, "missing_solver_output"))
            continue

        try:
            parsed = parse_solver_output(raw_text)
        except json.JSONDecodeError:
            reject_reasons["malformed_solver_json"] += 1
            rejects.append(_reject_row(record, "malformed_solver_json", raw_text[:200]))
            continue
        except (TypeError, ValueError):
            reject_reasons["invalid_solver_output"] += 1
            rejects.append(_reject_row(record, "invalid_solver_output", raw_text[:200]))
            continue

        if not VALID_ANSWER_RE.match(str(parsed.get("predicted_answer", ""))):
            reject_reasons["invalid_solver_answer"] += 1
            rejects.append(_reject_row(record, "invalid_solver_answer", raw_text[:200]))
            continue

        if not solver_keep_decision(record, parsed):
            reason = "solver_mismatch"
            if parsed.get("is_unambiguous") is not True:
                reason = "solver_ambiguous"
            elif parsed.get("bad_reason"):
                reason = "solver_bad_reason"
            reject_reasons[reason] += 1
            rejects.append(_reject_row(record, reason, raw_text[:200]))
            continue

        kept_record = build_solver_checked_record(record, parsed)
        kept.append(kept_record)
        metadata = _metadata(record)
        source_split = metadata.get("source_split")
        kept_by_split[str(source_split) if source_split not in (None, "") else "unknown"] += 1

    report = {
        "input_jsonl": None,
        "solver_outputs_jsonl": None,
        "output_jsonl": None,
        "rejects_jsonl": None,
        "report_json": None,
        "total_loaded": len(records),
        "total_kept": len(kept),
        "total_rejected": len(rejects),
        "kept_by_split": dict(kept_by_split),
        "reject_reasons": dict(reject_reasons),
    }
    return kept, rejects, report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--solver-jsonl", default=None)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--rejects-jsonl", default=DEFAULT_REJECTS_JSONL)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    args = parser.parse_args(argv)

    input_path = Path(args.input_jsonl)
    solver_path = Path(args.solver_jsonl) if args.solver_jsonl else _REPO_ROOT / "seed_exports" / "comprehension_mcq_solver_outputs_raw.jsonl"
    if not input_path.exists():
        print(f"[error] missing input file: {input_path}")
        return 1
    if not solver_path.exists():
        print(f"[error] missing solver output file: {solver_path}")
        return 1

    records, malformed_rejects = read_jsonl(input_path)
    solver_rows, solver_malformed = read_jsonl(solver_path)
    kept, rejects, report = process_records(records, solver_rows)
    rejects = malformed_rejects + solver_malformed + rejects
    report["total_loaded"] = len(records) + len(malformed_rejects)
    report["total_rejected"] = len(rejects)
    report["solver_outputs_loaded"] = len(solver_rows) + len(solver_malformed)
    if malformed_rejects:
        report["reject_reasons"]["invalid_json"] = len(malformed_rejects)
    if solver_malformed:
        report["reject_reasons"]["invalid_solver_json"] = len(solver_malformed)

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
            "solver_outputs_jsonl": str(solver_path),
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
