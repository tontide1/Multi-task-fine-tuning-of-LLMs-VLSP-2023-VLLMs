"""
recheck_exams_mcq_seed.py
==========================
Recheck / QC các thống kê của exams_mcq seed dataset SAU KHI đã chạy
load_exams_mcq_seed.py.

Script này đọc lại file JSONL đã export (seed_exports/exams_mcq_seed.jsonl)
và in ra một loạt kiểm tra:
  - Tổng số record
  - Phân bố source_dataset
  - Phân bố subject
  - Kiểm tra choice prefix còn sót
  - Kiểm tra answer_letter hợp lệ
  - Sample ngẫu nhiên từ hllj

Ngoài ra script còn re-run một số check trực tiếp trên intermediate
DataFrames (hllj_flat) để tiện so sánh với notebook gốc.
"""

import json
import random
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)


# ===========================================================================
# Helper (copy từ load script để recheck độc lập)
# ===========================================================================

def has_choice_prefix(x):
    if pd.isna(x):
        return False
    return bool(re.match(r"^\s*[A-Da-d]\s*[\.)\:\-]\s*", str(x), flags=re.IGNORECASE))


def clean_choice_text(x):
    if pd.isna(x):
        return None
    text = str(x).strip()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^\s*[A-Da-d]\s*[\.)\:\-]\s*", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_answer_letter(explanation):
    if pd.isna(explanation):
        return None
    text = str(explanation).strip()
    patterns = [
        r"Chọn\s+đáp\s+án\s*:?\s*([ABCD])\b",
        r"Đáp\s*án\s*cần\s*chọn\s*là\s*:?\s*([ABCD])\b",
        r"Đáp\s*án\s*đúng\s*là\s*:?\s*([ABCD])\b",
        r"Đáp\s*án\s*đúng\s*:?\s*([ABCD])\b",
        r"Đáp\s*án\s*:?\s*([ABCD])\b",
        r"Ta\s+chọn\s+([ABCD])\b",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None



def main():
    # ===========================================================================
    # 1. Đọc file JSONL đã export
    # ===========================================================================

    jsonl_path = Path(__file__).parent.parent / "seed_exports" / "exams_mcq_seed.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path} không tồn tại. Hãy chạy load_exams_mcq_seed.py trước."
        )

    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"=== [1] Total records in JSONL: {len(records)} ===\n")

    # Flatten metadata để dễ check
    rows = []
    for rec in records:
        meta = rec.get("metadata", {})
        user_msg = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
        asst_msg = next((m["content"] for m in rec["messages"] if m["role"] == "assistant"), "")
        rows.append({**meta, "user_content": user_msg, "assistant_content": asst_msg})

    df = pd.DataFrame(rows)


    # ===========================================================================
    # 2. Source distribution
    # ===========================================================================

    print("=== [2] Source distribution ===")
    print(df["source_dataset"].value_counts().to_string())
    print()


    # ===========================================================================
    # 3. Subject distribution
    # ===========================================================================

    print("=== [3] Subject distribution ===")
    print(df["subject"].value_counts(dropna=False).head(20).to_string())
    print()


    # ===========================================================================
    # 4. Schema validation
    # ===========================================================================

    print("=== [4] Schema validation ===")

    required_metadata_fields = ["task", "source", "source_dataset", "sample_id", "difficulty", "split", "language"]
    invalid_records = []

    for i, rec in enumerate(records):
        errors = []

        # Validate messages structure
        if not isinstance(rec.get("messages"), list):
            errors.append("messages is not a list")
        elif len(rec["messages"]) != 2:
            errors.append(f"messages has {len(rec['messages'])} elements, expected 2")
        else:
            if rec["messages"][0].get("role") != "user":
                errors.append(f"first message role is '{rec['messages'][0].get('role')}', expected 'user'")
            if rec["messages"][1].get("role") != "assistant":
                errors.append(f"second message role is '{rec['messages'][1].get('role')}', expected 'assistant'")
            if not str(rec["messages"][0].get("content", "")).strip():
                errors.append("user content is empty")
            if not str(rec["messages"][1].get("content", "")).strip():
                errors.append("assistant content is empty")

        # Validate metadata fields
        metadata = rec.get("metadata", {})
        for field in required_metadata_fields:
            if field not in metadata:
                errors.append(f"missing metadata field: {field}")

        if errors:
            invalid_records.append({"index": i, "sample_id": metadata.get("sample_id", "unknown"), "errors": errors})

    print(f"   Total records: {len(records)}")
    print(f"   Invalid records: {len(invalid_records)}")
    if len(invalid_records) > 0:
        print(f"   Sample of invalid records (first 5):")
        for inv in invalid_records[:5]:
            print(f"     - sample_id={inv['sample_id']}: {', '.join(inv['errors'])}")
    print()


    # ===========================================================================
    # 5. answer_letter distribution & validity
    # ===========================================================================

    valid_answers = {"A", "B", "C", "D"}
    answer_format_ok = df["assistant_content"].str.match(r"^Đáp án: [ABCD]$", na=False)
    invalid_mask = ~answer_format_ok
    print(f"=== [5] Invalid assistant answer format rows: {invalid_mask.sum()} ===\n")
    if invalid_mask.sum() > 0:
        print(df.loc[invalid_mask, ["sample_id", "assistant_content"]].head(20).to_string())

    # ===========================================================================
    # 6. Re-check hllj pipeline trực tiếp từ HF
    # ===========================================================================

    print("=== [6] Re-checking hllj/vi_grade_school_math_mcq pipeline ===")

    df_hllj_raw = pd.read_json("hf://datasets/hllj/vi_grade_school_math_mcq/vietjack.json")

    df_hllj_flat = df_hllj_raw.copy()
    df_hllj_flat = df_hllj_flat.explode("problems", ignore_index=True)
    problem_cols = pd.json_normalize(df_hllj_flat["problems"])
    df_hllj = pd.concat(
        [df_hllj_flat.drop(columns=["problems"]).reset_index(drop=True), problem_cols.reset_index(drop=True)],
        axis=1,
    )

    df_hllj["answer_letter"] = df_hllj["explanation"].apply(extract_answer_letter)
    hllj_has_4_choices = df_hllj["choices"].apply(lambda x: isinstance(x, list) and len(x) == 4)
    hllj_has_answer = df_hllj["answer_letter"].isin({"A", "B", "C", "D"})

    print(f"   Raw page rows       : {len(df_hllj_raw)}")
    print(f"   After explode       : {len(df_hllj_flat)}")
    # Check if question column exists before accessing
    if 'question' in df_hllj_flat.columns:
        print(f"   Non-null problems   : {df_hllj_flat['question'].notna().sum()}")
    else:
        print(f"   Non-null problems   : N/A (column 'question' not found)")
    print(f"   Has 4 choices       : {hllj_has_4_choices.sum()}")
    print(f"   Has parsed answer   : {hllj_has_answer.sum()}")
    print(f"   Pass both (4 + ans) : {(hllj_has_4_choices & hllj_has_answer).sum()}")
    print()

    # Mẫu rows không parse được answer
    df_hllj_failed = df_hllj[df_hllj["answer_letter"].isna()].copy()
    print(f"   Rows missing answer_letter: {len(df_hllj_failed)}")
    print("   Sample (5 rows) of missing:")
    # Check available columns before trying to access
    available_cols = [col for col in ["question", "choices", "explanation"] if col in df_hllj_failed.columns]
    if available_cols:
        print(df_hllj_failed[available_cols].head(5).to_string())
    else:
        print("   No columns available for display")
    print()

    # choice prefix check trên df_hllj intermediate (trước khi clean)
    _choice_cols = ["choice_A", "choice_B", "choice_C", "choice_D"]
    df_hllj[_choice_cols] = df_hllj["choices"].apply(
        lambda lst: pd.Series(
            [clean_choice_text(c) if isinstance(lst, list) and i < len(lst) else None for i, c in enumerate(lst)]
            if isinstance(lst, list) and len(lst) == 4
            else [None, None, None, None]
        )
    )

    bad_prefix_rows = df_hllj[
        df_hllj[_choice_cols]
        .apply(lambda col: col.apply(has_choice_prefix))
        .any(axis=1)
    ]
    print(f"   Rows with raw choice prefix before cleaning (hllj intermediate): {len(bad_prefix_rows)}")
    print()


    # ===========================================================================
    # 7. Choice prefix check trên df cuối (đọc lại từ JSONL)
    # ===========================================================================

    print("=== [7] Choice prefix check on final JSONL ===")

    # Extract choice lines từ user_content
    prefix_pattern = r"^\s*[A-Da-d]\s*[\.)\:\-]\s*"

    def extract_choices_from_user(user_content: str):
        lines = user_content.split("\n")
        choices = {}
        choice_pattern = re.compile(r"^\s*([A-D])\s*[\.)\:\-]\s*(.+)$")
        for line in lines:
            match = choice_pattern.match(line)
            if match:
                label = match.group(1).upper()
                text = match.group(2).strip()
                choices[f"choice_{label}"] = text
        return choices

    bad_prefix_count = 0
    for rec in records:
        user_msg = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
        choices = extract_choices_from_user(user_msg)
        for v in choices.values():
            if re.match(prefix_pattern, v, flags=re.IGNORECASE):
                bad_prefix_count += 1
                break

    print(f"   Records with choice prefix still present: {bad_prefix_count}")
    print()

    def count_choice_lines(user_content: str) -> int:
        lines = user_content.splitlines()
        choice_pattern = re.compile(r"^\s*[A-D]\s*[\.)\:\-]\s*.+$")
        return sum(bool(choice_pattern.match(line)) for line in lines)

    df["num_choice_lines"] = df["user_content"].apply(count_choice_lines)
    bad_choice_count = (df["num_choice_lines"] != 4).sum()

    print(f"=== Choice line count != 4 rows: {bad_choice_count} ===")
    if bad_choice_count > 0:
        print(df.loc[df["num_choice_lines"] != 4, ["sample_id", "user_content"]].head(10).to_string())
    print()

    # ===========================================================================
    # 8. Sample spot-check
    # ===========================================================================

    print("=== [8] Random sample (5 records from hllj) ===")
    hllj_records = [r for r in records if r["metadata"]["source_dataset"] == "hllj/vi_grade_school_math_mcq"]
    random.seed(42)
    sample = random.sample(hllj_records, min(5, len(hllj_records)))
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
