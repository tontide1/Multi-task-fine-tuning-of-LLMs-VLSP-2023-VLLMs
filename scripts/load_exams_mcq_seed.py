"""
load_exams_mcq_seed.py
======================
Load, normalize và export exams_mcq seed dataset từ 2 nguồn HF:
  1. roshansk23/Vietnam_HighSchool_Exam_Dataset
  2. hllj/vi_grade_school_math_mcq

Output: seed_exports/exams_mcq_seed.jsonl  (relative to CWD)
"""

import ast
import json
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
load_dotenv()

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)


# ===========================================================================
# Helper functions
# ===========================================================================

def ensure_list(x):
    """Parse string-encoded list hoặc trả về list nếu đã là list."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return x


def clean_choice_text(x):
    """Loại bỏ prefix 'A. ', 'B) ', ... khỏi text đáp án."""
    if pd.isna(x):
        return None
    text = str(x).strip()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^\s*[A-Da-d]\s*[\.)\:\-]\s*", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_choices(choice_list):
    """Trả về pd.Series([A, B, C, D]) từ list 4 lựa chọn."""
    if not isinstance(choice_list, list) or len(choice_list) != 4:
        return pd.Series([None, None, None, None])
    cleaned = [clean_choice_text(c) for c in choice_list]
    return pd.Series(cleaned)


def extract_answer_letter(explanation):
    """Parse đáp án từ chuỗi explanation theo nhiều format regex."""
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


def nonempty(x):
    return pd.notna(x) and str(x).strip() != ""


def build_user_content(row):
    return (
        f"Câu hỏi: {str(row['question']).strip()}\n"
        f"A. {str(row['choice_A']).strip()}\n"
        f"B. {str(row['choice_B']).strip()}\n"
        f"C. {str(row['choice_C']).strip()}\n"
        f"D. {str(row['choice_D']).strip()}"
    )


def build_record(row):
    return {
        "messages": [
            {"role": "user", "content": build_user_content(row)},
            {"role": "assistant", "content": f"Đáp án: {row['answer_letter']}"},
        ],
        "metadata": {
            "task": "exams_mcq",
            "source": "public",
            "source_dataset": row["source_dataset"],
            "sample_id": str(row["sample_id"]),
            "difficulty": "medium",
            "split": "train",
            "language": "vi",
            "subject": row["subject"] if pd.notna(row["subject"]) else None,
            "grade": str(row["grade"]) if pd.notna(row["grade"]) else None,
        },
    }



def main():
    # ===========================================================================
    # 1. Load & normalize roshansk23/Vietnam_HighSchool_Exam_Dataset
    # ===========================================================================

    print("[1/5] Loading roshansk23/Vietnam_HighSchool_Exam_Dataset ...")
    df_ro = pd.read_json("hf://datasets/roshansk23/Vietnam_HighSchool_Exam_Dataset/cleaned_VNHSGE.json")

    # Parse options thành list
    df_ro["options"] = df_ro["options"].apply(ensure_list)

    # Chỉ giữ mẫu có đúng 4 lựa chọn
    df_ro = df_ro[df_ro["options"].apply(lambda x: isinstance(x, list) and len(x) == 4)].copy()

    # Map answer 1-4 -> A-D
    answer_map_1_based = {1: "A", 2: "B", 3: "C", 4: "D", "1": "A", "2": "B", "3": "C", "4": "D"}
    df_ro["answer_letter"] = df_ro["answer"].map(answer_map_1_based)

    # Tách 4 lựa chọn
    df_ro["choice_A"] = df_ro["options"].apply(lambda x: str(x[0]).strip())
    df_ro["choice_B"] = df_ro["options"].apply(lambda x: str(x[1]).strip())
    df_ro["choice_C"] = df_ro["options"].apply(lambda x: str(x[2]).strip())
    df_ro["choice_D"] = df_ro["options"].apply(lambda x: str(x[3]).strip())

    df_ro["subject"] = df_ro["category_original_lang"].fillna(df_ro["category_en"])
    df_ro["grade"] = df_ro["level"]
    df_ro["source_dataset"] = "roshansk23/Vietnam_HighSchool_Exam_Dataset"
    df_ro["sample_id"] = "roshansk_" + df_ro.index.astype(str)

    print(f"   roshansk rows after filter: {len(df_ro)}")


    # ===========================================================================
    # 2. Load & normalize hllj/vi_grade_school_math_mcq
    # ===========================================================================

    print("[2/5] Loading hllj/vi_grade_school_math_mcq ...")
    df_hllj_raw = pd.read_json("hf://datasets/hllj/vi_grade_school_math_mcq/vietjack.json")

    # Explode problems list
    df_hllj_flat = df_hllj_raw.copy()
    df_hllj_flat = df_hllj_flat.explode("problems", ignore_index=True)
    problem_cols = pd.json_normalize(df_hllj_flat["problems"])
    df_hllj = pd.concat(
        [df_hllj_flat.drop(columns=["problems"]).reset_index(drop=True), problem_cols.reset_index(drop=True)],
        axis=1,
    )

    # Parse choices thành A/B/C/D, bỏ prefix
    # (clean_choice_text được áp dụng lại sau concat tại step 3 — không cần gọi riêng ở đây)
    df_hllj[["choice_A", "choice_B", "choice_C", "choice_D"]] = df_hllj["choices"].apply(parse_choices)

    # Extract answer_letter từ explanation
    df_hllj["answer_letter"] = df_hllj["explanation"].apply(extract_answer_letter)

    df_hllj["subject"] = "Toán"
    df_hllj["source_dataset"] = "hllj/vi_grade_school_math_mcq"
    df_hllj["sample_id"] = df_hllj["id"].astype(str) + "_" + df_hllj.index.astype(str)
    # grade đã có từ raw data

    print(f"   hllj flat rows: {len(df_hllj)}")


    # ===========================================================================
    # 3. Gộp hai nguồn về cùng schema
    # ===========================================================================

    print("[3/5] Merging datasets ...")
    common_cols = [
        "question", "choice_A", "choice_B", "choice_C", "choice_D",
        "answer_letter", "subject", "grade", "source_dataset", "sample_id",
    ]

    df_exams_seed = pd.concat(
        [df_ro[common_cols].copy(), df_hllj[common_cols].copy()],
        ignore_index=True,
    )

    choice_cols = ["choice_A", "choice_B", "choice_C", "choice_D"]
    for col in choice_cols:
        df_exams_seed[col] = df_exams_seed[col].apply(clean_choice_text)
    print(f"   Combined rows: {len(df_exams_seed)}")


    # ===========================================================================
    # 4. Filter & dedup
    # ===========================================================================

    print("[4/5] Filtering and deduplicating ...")
    valid_answers = {"A", "B", "C", "D"}

    mask_valid = (
        df_exams_seed["question"].apply(nonempty)
        & df_exams_seed["choice_A"].apply(nonempty)
        & df_exams_seed["choice_B"].apply(nonempty)
        & df_exams_seed["choice_C"].apply(nonempty)
        & df_exams_seed["choice_D"].apply(nonempty)
        & df_exams_seed["answer_letter"].isin(valid_answers)
    )

    df_exams_seed = df_exams_seed[mask_valid].copy()
    df_exams_seed = df_exams_seed.drop_duplicates(
        subset=["question", "choice_A", "choice_B", "choice_C", "choice_D", "answer_letter"]
    ).reset_index(drop=True)

    print(f"   Rows after filter/dedup: {len(df_exams_seed)}")

    print("   Source distribution:")
    print(df_exams_seed["source_dataset"].value_counts().to_string())
    print("   Subject distribution:")
    print(df_exams_seed["subject"].value_counts(dropna=False).head(20).to_string())


    # ===========================================================================
    # 5. Export JSONL
    # ===========================================================================

    print("[5/5] Writing JSONL ...")
    records = [build_record(row) for _, row in df_exams_seed.iterrows()]

    out_path = Path(__file__).parent.parent / "seed_exports" / "exams_mcq_seed.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {out_path} | records={len(records)}")


if __name__ == "__main__":
    main()
