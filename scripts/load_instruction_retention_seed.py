"""
load_instruction_retention_seed.py
==================================
ETL for instruction_retention_seed.jsonl from vi-alpaca and alpaca_multiturns_dialogue_vi.
"""

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
import random

# Constants
VI_ALPACA_ID = "bkai-foundation-models/vi-alpaca"
MULTITURN_ID = "lamhieu/alpaca_multiturns_dialogue_vi"

TASK_NAME = "instruction_retention"
QC_VERSION = "instruction_retention_v1"
RANDOM_SEED = 42

MIN_CONTENT_CHARS = 10
MAX_USER_CHARS = 2000
MAX_ASSISTANT_CHARS = 4000

# Target sizes
CAP_VI_ALPACA = 12000
CAP_MULTITURN = 8000

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u00a0", " ")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()

def clean_markers(text: str) -> str:
    if text is None:
        return ""
    pattern = r"^\s*(?:#+\s*(?:Human|Assistant|Instruction|Response|Input|Output)\s*:?|(?:Human|Assistant|Instruction|Response|Input|Output)\s*:)\s*"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()

def is_mcq_like(user_content: str, assistant_content: str) -> bool:
    has_choices = bool(re.search(r"[A-D]\.\s+.*?\n[A-D]\.\s+.*?\n[A-D]\.\s+.*?\n[A-D]\.\s+", user_content))
    is_short_answer = bool(re.match(r"^\s*(Đáp án|Answer|Kết quả)\s*:?\s*[A-D]\s*$", assistant_content, re.IGNORECASE))
    
    if is_short_answer:
        return True
    if has_choices and len(assistant_content) < 20:
        if re.match(r"^\s*[A-D]\.?\s*$", assistant_content, re.IGNORECASE):
            return True
    return False

def extract_first_user_assistant_pair(messages):
    # messages can be list or numpy.ndarray
    if messages is None:
        return None, None
    try:
        n = len(messages)
    except TypeError:
        return None, None
        
    for i in range(n - 1):
        m1 = messages[i]
        m2 = messages[i+1]
        # Use .get() but handle case where m1/m2 might be accessed as dict
        try:
            role1 = m1.get("role")
            role2 = m2.get("role")
            if role1 == "user" and role2 == "assistant":
                return m1.get("content"), m2.get("content")
        except AttributeError:
            continue
    return None, None

def make_dedup_hash(user: str, assistant: str) -> str:
    u = normalize_text(user)
    a = normalize_text(assistant)
    key = f"{u}|{a}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def build_record(user, assistant, source_dataset, source_split, source_id, turn_type):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "task": TASK_NAME,
            "source": "public",
            "source_dataset": source_dataset,
            "source_split": source_split,
            "source_id": source_id,
            "language": "vi",
            "difficulty": "medium",
            "turn_type": turn_type,
            "dedup_hash": make_dedup_hash(user, assistant),
            "qc_version": QC_VERSION,
        }
    }

def build_reject_record(source_dataset, source_split, source_id, reason, user_preview, assistant_preview):
    return {
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_id": source_id,
        "reason": reason,
        "user_preview": (user_preview or "")[:200],
        "assistant_preview": (assistant_preview or "")[:200],
    }

def process_vi_alpaca_row(row, split_name):
    instruction = row.get("instruction", "")
    input_text = row.get("input", "")
    output = row.get("output", "")
    source_id = f"alpaca-{row.get('_row_index')}"

    user = instruction
    if input_text and input_text.strip():
        user = f"{instruction}\n\n{input_text}"
    
    assistant = output
    
    # 1. Normalize
    user = normalize_text(user)
    assistant = normalize_text(assistant)
    
    # 2. Clean markers
    user = clean_markers(user)
    assistant = clean_markers(assistant)
    
    # 3. Filters
    if not user or len(user) < MIN_CONTENT_CHARS:
        return None, "user_too_short"
    if len(user) > MAX_USER_CHARS:
        return None, "user_too_long"
    if not assistant or len(assistant) < MIN_CONTENT_CHARS:
        return None, "assistant_too_short"
    if len(assistant) > MAX_ASSISTANT_CHARS:
        return None, "assistant_too_long"
    
    if is_mcq_like(user, assistant):
        return None, "mcq_like_contamination"
    
    record = build_record(user, assistant, VI_ALPACA_ID, split_name, source_id, "single_turn")
    return record, None

def process_multiturn_row(row, split_name):
    messages = row.get("messages", [])
    source_id = f"multiturn-{row.get('_row_index')}"
    
    user_raw, assistant_raw = extract_first_user_assistant_pair(messages)
    
    if not user_raw or not assistant_raw:
        return None, "no_valid_user_assistant_pair"
    
    # 1. Normalize
    user = normalize_text(user_raw)
    assistant = normalize_text(assistant_raw)
    
    # 2. Clean markers
    user = clean_markers(user)
    assistant = clean_markers(assistant)
    
    # 3. Filters
    if not user or len(user) < MIN_CONTENT_CHARS:
        return None, "user_too_short"
    if len(user) > MAX_USER_CHARS:
        return None, "user_too_long"
    if not assistant or len(assistant) < MIN_CONTENT_CHARS:
        return None, "assistant_too_short"
    if len(assistant) > MAX_ASSISTANT_CHARS:
        return None, "assistant_too_long"
    
    if is_mcq_like(user, assistant):
        return None, "mcq_like_contamination"
    
    record = build_record(user, assistant, MULTITURN_ID, split_name, source_id, "first_turn_pair")
    return record, None

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    # Lazy imports for heavy dependencies
    import pandas as pd
    from datasets import load_dataset
    from tqdm.auto import tqdm

    repo_root = Path(__file__).parent.parent
    out_dir = repo_root / "seed_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_jsonl = out_dir / "instruction_retention_seed.jsonl"
    out_rejects = out_dir / "instruction_retention_seed_rejects.jsonl"
    out_report = out_dir / "instruction_retention_seed_report.json"
    
    print("[1/5] Loading datasets...")
    ds_alpaca = load_dataset(VI_ALPACA_ID)
    ds_multi = load_dataset(MULTITURN_ID)
    
    kept_alpaca = []
    kept_multi = []
    rejects = []
    reject_reasons = Counter()
    
    print("[2/5] Processing vi-alpaca...")
    for split_name, split_ds in ds_alpaca.items():
        df = split_ds.to_pandas()
        df["_row_index"] = df.index
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"vi-alpaca {split_name}"):
            record, reason = process_vi_alpaca_row(row.to_dict(), split_name)
            if reason:
                reject_reasons[reason] += 1
                rejects.append(build_reject_record(
                    VI_ALPACA_ID, 
                    split_name, 
                    f"alpaca-{row.get('_row_index')}", 
                    reason, 
                    row.get("instruction"), 
                    row.get("output")
                ))
            else:
                kept_alpaca.append(record)
                
    print("[3/5] Processing multiturn...")
    for split_name, split_ds in ds_multi.items():
        df = split_ds.to_pandas()
        df["_row_index"] = df.index
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"multiturn {split_name}"):
            record, reason = process_multiturn_row(row.to_dict(), split_name)
            if reason:
                reject_reasons[reason] += 1
                rejects.append(build_reject_record(
                    MULTITURN_ID, 
                    split_name, 
                    f"multiturn-{row.get('_row_index')}", 
                    reason, 
                    str(row.get("messages")), 
                    ""
                ))
            else:
                kept_multi.append(record)
    
    print("[4/5] Deduplicating and Sampling...")
    
    def dedup_and_log(records, dataset_id):
        seen = set()
        unique = []
        for r in records:
            h = r["metadata"]["dedup_hash"]
            if h in seen:
                reject_reasons["duplicate"] += 1
                rejects.append(build_reject_record(
                    dataset_id,
                    r["metadata"]["source_split"],
                    r["metadata"]["source_id"],
                    "duplicate",
                    r["messages"][0]["content"],
                    r["messages"][1]["content"]
                ))
            else:
                seen.add(h)
                unique.append(r)
        return unique
    
    unique_alpaca = dedup_and_log(kept_alpaca, VI_ALPACA_ID)
    unique_multi = dedup_and_log(kept_multi, MULTITURN_ID)
    
    # Random sampling
    random.seed(RANDOM_SEED)
    final_alpaca = random.sample(unique_alpaca, min(len(unique_alpaca), CAP_VI_ALPACA))
    final_multi = random.sample(unique_multi, min(len(unique_multi), CAP_MULTITURN))
    
    final_all = final_alpaca + final_multi
    random.shuffle(final_all)
    
    print("[5/5] Writing outputs...")
    write_jsonl(out_jsonl, final_all)
    write_jsonl(out_rejects, rejects)
    
    report = {
        "total_processed": len(kept_alpaca) + len(kept_multi) + len(rejects),
        "total_kept_before_sample": len(unique_alpaca) + len(unique_multi),
        "total_final": len(final_all),
        "by_source": {
            VI_ALPACA_ID: len(final_alpaca),
            MULTITURN_ID: len(final_multi)
        },
        "reject_reasons": dict(reject_reasons),
        "qc_version": QC_VERSION,
        "random_seed": RANDOM_SEED
    }
    
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        
    print(f"Done! Exported {len(final_all)} records to {out_jsonl}")

if __name__ == "__main__":
    main()
