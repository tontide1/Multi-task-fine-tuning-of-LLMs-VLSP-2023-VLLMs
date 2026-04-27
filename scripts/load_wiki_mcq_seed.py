"""
load_wiki_mcq_seed.py
======================
Load, normalize và export wiki_mcq seed dataset từ HuggingFace:
  - lighteval/okapi_mmlu (config: vi)

Pipeline:
  1. Load tất cả splits (test, validation, dev) từ HF.
  2. Gộp thành một DataFrame duy nhất.
  3. Normalize: clean text, parse choices (ndarray → list), normalize answer.
  4. Convert từng row thành chat format {"messages": [...], "metadata": {...}}.
  5. Dedup theo MD5 hash của (question, choices).
  6. Export JSONL + rejects JSONL + report JSON.

Output (relative to repo root / seed_exports/):
  - wiki_mcq_seed.jsonl
  - wiki_mcq_seed_rejects.jsonl
  - wiki_mcq_seed_report.json
"""

import ast
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
load_dotenv()

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ID = "lighteval/okapi_mmlu"
CONFIG_NAME = "vi"

TASK_NAME = "wiki_mcq"
SOURCE_NAME = "lighteval/okapi_mmlu:vi"
DIFFICULTY = "medium"

ANSWER_LETTERS = ["A", "B", "C", "D"]


# ===========================================================================
# Helper functions
# ===========================================================================

def clean_text(x):
    """Clean whitespace và NBSP."""
    if x is None:
        return ""
    x = str(x)
    x = x.replace("\u00a0", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_answer(ans):
    """
    Normalize answer về A/B/C/D.
    Chấp nhận: 'A'/'B'/'C'/'D', 0/1/2/3, '0'/'1'/'2'/'3'.
    """
    if ans is None:
        return None

    if isinstance(ans, int):
        if 0 <= ans <= 3:
            return ANSWER_LETTERS[ans]
        return None

    s = str(ans).strip().upper()

    if s in ANSWER_LETTERS:
        return s

    if s in ["0", "1", "2", "3"]:
        return ANSWER_LETTERS[int(s)]

    m = re.match(r"^([ABCD])[\.\)]?$", s)
    if m:
        return m.group(1)

    return None


def normalize_choices(choices):
    """
    Normalize choices về list[str] 4 phần tử.
    Chấp nhận: list, tuple, numpy.ndarray, pandas.Series, string-encoded list.
    """
    if choices is None:
        return None

    if isinstance(choices, np.ndarray):
        choices = choices.tolist()
    elif isinstance(choices, pd.Series):
        choices = choices.tolist()
    elif isinstance(choices, tuple):
        choices = list(choices)
    elif isinstance(choices, str):
        s = choices.strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                choices = list(parsed)
            else:
                return None
        except Exception:
            return None

    if not isinstance(choices, list):
        return None

    if len(choices) != 4:
        return None

    cleaned = [clean_text(c) for c in choices]

    if any(not c for c in cleaned):
        return None

    return cleaned


def make_user_content(question, choices):
    return "\n".join([
        f"Câu hỏi: {question}",
        f"A. {choices[0]}",
        f"B. {choices[1]}",
        f"C. {choices[2]}",
        f"D. {choices[3]}",
    ])


def make_hash(question, choices):
    key = json.dumps(
        {"question": question, "choices": choices},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def convert_row(row):
    """
    Convert một row dict thành chat record.
    Trả về (item, reject_reasons).
    item là None nếu row bị reject.
    """
    question = clean_text(row.get("question"))
    choices = normalize_choices(row.get("choices"))
    answer = normalize_answer(row.get("answer"))

    reject_reasons = []

    if not question:
        reject_reasons.append("missing_question")

    if choices is None:
        reject_reasons.append("invalid_choices")

    if answer is None:
        reject_reasons.append("invalid_answer")

    if reject_reasons:
        return None, reject_reasons

    user_content = make_user_content(question, choices)

    item = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"Đáp án: {answer}"},
        ],
        "metadata": {
            "task": TASK_NAME,
            "source": SOURCE_NAME,
            "difficulty": DIFFICULTY,
            "source_dataset": DATASET_ID,
            "source_config": CONFIG_NAME,
            "source_split": row.get("original_split"),
            "source_id": row.get("id"),
            "subject": row.get("subject"),
            "answer": answer,
        },
    }

    item["metadata"]["dedup_hash"] = make_hash(question, choices)
    return item, []


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    from datasets import load_dataset

    # Determine output directory relative to repo root
    out_dir = Path(__file__).parent.parent / "seed_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "wiki_mcq_seed.jsonl"
    out_rejects_jsonl = out_dir / "wiki_mcq_seed_rejects.jsonl"
    out_report_json = out_dir / "wiki_mcq_seed_report.json"

    # ===========================================================================
    # 1. Load dataset từ HF
    # ===========================================================================

    print(f"[1/5] Loading {DATASET_ID} (config={CONFIG_NAME}) ...")
    try:
        ds = load_dataset(DATASET_ID, CONFIG_NAME)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    for split_name, split_ds in ds.items():
        print(f"   {split_name}: {len(split_ds)} rows | cols: {split_ds.column_names}")

    # ===========================================================================
    # 2. Gộp tất cả splits về một DataFrame
    # ===========================================================================

    print("[2/5] Merging all splits into one DataFrame ...")
    frames = []
    for split_name, split_ds in ds.items():
        df_split = split_ds.to_pandas()
        df_split["original_split"] = split_name
        frames.append(df_split)

    df_raw = pd.concat(frames, ignore_index=True)
    print(f"   Total rows: {len(df_raw)}")

    # Thống kê nhanh
    print("   Rows by split:")
    print(df_raw["original_split"].value_counts().to_string())
    print("   Answer distribution:")
    print(df_raw["answer"].value_counts(dropna=False).to_string())

    # ===========================================================================
    # 3. Convert rows → chat format
    # ===========================================================================

    print("[3/5] Converting rows ...")
    converted = []
    rejects = []

    for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
        item, reasons = convert_row(row.to_dict())

        if item is None:
            reject = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in row.to_dict().items()}
            reject["_row_index"] = int(idx)
            reject["_reject_reasons"] = reasons
            rejects.append(reject)
        else:
            converted.append(item)

    print(f"   Rows before filter : {len(df_raw)}")
    print(f"   Rows after filter  : {len(converted)}")
    print(f"   Rejected           : {len(rejects)}")

    # ===========================================================================
    # 4. Deduplication theo dedup_hash
    # ===========================================================================

    print("[4/5] Deduplicating ...")
    seen = set()
    deduped = []
    duplicates = []

    for item in converted:
        h = item["metadata"]["dedup_hash"]
        if h in seen:
            duplicates.append(item)
            continue
        seen.add(h)
        deduped.append(item)

    print(f"   Rows before dedup  : {len(converted)}")
    print(f"   Rows after dedup   : {len(deduped)}")
    print(f"   Duplicates removed : {len(duplicates)}")

    # Quick QA: đếm phân bố answer / split / subject
    answer_counter: Counter = Counter()
    split_counter: Counter = Counter()
    subject_counter: Counter = Counter()
    bad_assistant_format = 0
    bad_user_format = 0

    for item in deduped:
        content = item["messages"][1]["content"]
        m = re.match(r"^Đáp án:\s*([ABCD])$", content)
        if m is None:
            bad_assistant_format += 1
        else:
            answer_counter[m.group(1)] += 1

        user_content = item["messages"][0]["content"]
        for prefix in ["Câu hỏi:", "A.", "B.", "C.", "D."]:
            if prefix not in user_content:
                bad_user_format += 1
                break

        subject_counter[item["metadata"].get("subject")] += 1
        split_counter[item["metadata"].get("source_split")] += 1

    print(f"   Bad assistant format: {bad_assistant_format}")
    print(f"   Bad user format    : {bad_user_format}")

    # ===========================================================================
    # 5. Export JSONL + report
    # ===========================================================================

    print("[5/5] Writing output files ...")
    write_jsonl(out_jsonl, deduped)
    write_jsonl(out_rejects_jsonl, rejects)

    top_subjects = pd.Series(subject_counter).sort_values(ascending=False).head(50)

    report = {
        "dataset_id": DATASET_ID,
        "config": CONFIG_NAME,
        "task": TASK_NAME,
        "source": SOURCE_NAME,
        "rows_before_filter": int(len(df_raw)),
        "rows_after_basic_filter": int(len(converted)),
        "rows_after_dedup": int(len(deduped)),
        "rejected": int(len(rejects)),
        "duplicates_removed": int(len(duplicates)),
        "answer_distribution": {str(k): int(v) for k, v in sorted(answer_counter.items())},
        "split_distribution": {str(k): int(v) for k, v in split_counter.items()},
        "top_subjects": {str(k): int(v) for k, v in top_subjects.items()},
        "output_jsonl": str(out_jsonl),
        "rejects_jsonl": str(out_rejects_jsonl),
    }

    with open(out_report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] {out_jsonl} | records={len(deduped)}")
    print(f"[OK] {out_rejects_jsonl} | rejects={len(rejects)}")
    print(f"[OK] {out_report_json}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
