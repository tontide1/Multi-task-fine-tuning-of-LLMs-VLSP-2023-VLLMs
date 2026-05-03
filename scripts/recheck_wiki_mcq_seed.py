"""
recheck_wiki_mcq_seed.py
=========================
Recheck / QC thống kê của wiki_mcq seed dataset SAU KHI đã chạy
load_wiki_mcq_seed.py.

Script này đọc lại file JSONL đã export (seed_exports/wiki_mcq_seed.jsonl)
và thực hiện các kiểm tra:
  [1] Tổng số record
  [2] Source / config distribution
  [3] Subject distribution
  [4] Schema validation (messages structure + metadata fields)
  [5] Answer format check ("Đáp án: X")
  [6] User content format check (Câu hỏi + 4 choices)
  [7] Choice prefix check (đảm bảo không còn prefix A. / B. ... trong nội dung đáp án)
  [8] Dedup hash uniqueness
  [9] Random sample spot-check
"""

import json
import random
import re
from pathlib import Path

import pandas as pd

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)


# ===========================================================================
# Helpers
# ===========================================================================

REQUIRED_METADATA_FIELDS = [
    "task", "source", "difficulty",
    "source_dataset", "source_config", "source_split",
    "source_id", "subject", "answer", "dedup_hash",
]

ANSWER_FORMAT_RE = re.compile(r"^Đáp án: [ABCD]$")
CHOICE_LINE_RE = re.compile(r"^\s*[A-D]\s*[\.\):\-]\s*.+$")
CHOICE_PREFIX_RE = re.compile(r"^\s*[A-Da-d]\s*[\.\):\-]\s*", re.IGNORECASE)


def count_choice_lines(user_content: str) -> int:
    """Đếm số dòng dạng 'A. ...' / 'B. ...' trong user content."""
    return sum(bool(CHOICE_LINE_RE.match(line)) for line in user_content.splitlines())


def extract_choices_from_user(user_content: str) -> dict:
    """Trích các đáp án A/B/C/D từ user_content."""
    choices = {}
    pattern = re.compile(r"^\s*([A-D])\s*[\.\):\-]\s*(.+)$")
    for line in user_content.split("\n"):
        m = pattern.match(line)
        if m:
            choices[m.group(1).upper()] = m.group(2).strip()
    return choices


# ===========================================================================
# Main
# ===========================================================================

def main():
    jsonl_path = Path(__file__).parent.parent / "seed_exports" / "wiki_mcq_seed_final.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path} không tồn tại. Hãy chạy load_wiki_mcq_seed.py trước."
        )

    # Load records
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # ===========================================================================
    # [1] Total records
    # ===========================================================================

    print(f"=== [1] Total records in JSONL: {len(records)} ===\n")

    # Flatten metadata
    rows = []
    for rec in records:
        meta = rec.get("metadata", {})
        user_msg = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
        asst_msg = next((m["content"] for m in rec["messages"] if m["role"] == "assistant"), "")
        rows.append({**meta, "user_content": user_msg, "assistant_content": asst_msg})

    df = pd.DataFrame(rows)

    # ===========================================================================
    # [2] Source / config distribution
    # ===========================================================================

    print("=== [2] Source distribution ===")
    print(df["source_dataset"].value_counts(dropna=False).to_string())
    print()
    print("   Config distribution:")
    print(df["source_config"].value_counts(dropna=False).to_string())
    print()
    print("   Split distribution:")
    print(df["source_split"].value_counts(dropna=False).to_string())
    print()

    # ===========================================================================
    # [3] Subject distribution
    # ===========================================================================

    print("=== [3] Subject distribution (top 30) ===")
    print(df["subject"].value_counts(dropna=False).head(30).to_string())
    print()

    # ===========================================================================
    # [4] Schema validation
    # ===========================================================================

    print("=== [4] Schema validation ===")
    invalid_records = []

    for i, rec in enumerate(records):
        errors = []

        # messages structure
        if not isinstance(rec.get("messages"), list):
            errors.append("messages is not a list")
        elif len(rec["messages"]) != 2:
            errors.append(f"messages has {len(rec['messages'])} elements (expected 2)")
        else:
            if rec["messages"][0].get("role") != "user":
                errors.append(f"first role='{rec['messages'][0].get('role')}' (expected 'user')")
            if rec["messages"][1].get("role") != "assistant":
                errors.append(f"second role='{rec['messages'][1].get('role')}' (expected 'assistant')")
            if not str(rec["messages"][0].get("content", "")).strip():
                errors.append("user content is empty")
            if not str(rec["messages"][1].get("content", "")).strip():
                errors.append("assistant content is empty")

        # metadata fields
        meta = rec.get("metadata", {})
        for field in REQUIRED_METADATA_FIELDS:
            if field not in meta:
                errors.append(f"missing metadata field: {field}")

        if errors:
            invalid_records.append({
                "index": i,
                "source_id": meta.get("source_id", "unknown"),
                "errors": errors,
            })

    print(f"   Total records  : {len(records)}")
    print(f"   Invalid records: {len(invalid_records)}")
    if invalid_records:
        print("   Sample (first 5):")
        for inv in invalid_records[:5]:
            print(f"     - source_id={inv['source_id']}: {', '.join(inv['errors'])}")
    print()

    # ===========================================================================
    # [5] Answer format check
    # ===========================================================================

    answer_format_ok = df["assistant_content"].str.match(r"^Đáp án: [ABCD]$", na=False)
    bad_answer_count = (~answer_format_ok).sum()
    print(f"=== [5] Invalid answer format rows: {bad_answer_count} ===")
    if bad_answer_count > 0:
        print(df.loc[~answer_format_ok, ["source_id", "assistant_content"]].head(10).to_string())
    print()

    # Answer distribution
    print("   Answer distribution:")
    print(df["answer"].value_counts(dropna=False).sort_index().to_string())
    print()

    # ===========================================================================
    # [6] User content format check
    # ===========================================================================

    df["num_choice_lines"] = df["user_content"].apply(count_choice_lines)
    bad_choice_count = (df["num_choice_lines"] != 4).sum()

    print(f"=== [6] User content: rows with choice lines != 4: {bad_choice_count} ===")
    if bad_choice_count > 0:
        print(df.loc[df["num_choice_lines"] != 4, ["source_id", "user_content"]].head(10).to_string())
    print()

    # Check "Câu hỏi:" prefix exists
    no_question_prefix = (~df["user_content"].str.contains("Câu hỏi:", na=False)).sum()
    print(f"   Rows missing 'Câu hỏi:' prefix: {no_question_prefix}")
    print()

    # ===========================================================================
    # [7] Choice prefix check (nội dung đáp án không được có A. / B. ... ở đầu)
    # ===========================================================================

    print("=== [7] Choice prefix check on final JSONL ===")
    bad_prefix_count = 0
    for rec in records:
        user_msg = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
        choices = extract_choices_from_user(user_msg)
        for v in choices.values():
            if CHOICE_PREFIX_RE.match(v):
                bad_prefix_count += 1
                break

    print(f"   Records with leftover choice prefix: {bad_prefix_count}")
    print()

    # ===========================================================================
    # [8] Dedup hash uniqueness
    # ===========================================================================

    print("=== [8] Dedup hash uniqueness ===")
    hashes = [rec["metadata"].get("dedup_hash") for rec in records]
    unique_hashes = set(hashes)
    duplicate_hashes = len(hashes) - len(unique_hashes)

    print(f"   Total hashes  : {len(hashes)}")
    print(f"   Unique hashes : {len(unique_hashes)}")
    print(f"   Duplicates    : {duplicate_hashes}")
    print()

    # ===========================================================================
    # [9] Random sample spot-check
    # ===========================================================================

    print("=== [9] Random sample (5 records) ===")
    random.seed(42)
    sample = random.sample(records, min(5, len(records)))
    for i, rec in enumerate(sample, 1):
        print(f"  --- sample {i} ---")
        print("[user]")
        print(rec["messages"][0]["content"])
        print("[assistant]")
        print(rec["messages"][1]["content"])
        print("[metadata]")
        print(json.dumps(rec["metadata"], ensure_ascii=False, indent=2))
        print()

    print("[DONE] recheck complete.")


if __name__ == "__main__":
    main()
