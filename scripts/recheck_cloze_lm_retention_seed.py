
import json
import random
import re
from pathlib import Path
from collections import Counter

def validate_record_schema(record: dict):
    errors = []
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        errors.append("bad_messages_structure")
        return errors

    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        errors.append("bad_roles")

    user_content = messages[0].get("content", "")
    assistant_content = messages[1].get("content", "")

    if not user_content:
        errors.append("empty_user_content")
    if not assistant_content:
        errors.append("empty_assistant_content")

    metadata = record.get("metadata", {})
    required = ["task", "source", "source_dataset", "source_id", "dedup_hash", "domain"]
    for field in required:
        if field not in metadata:
            errors.append(f"missing_metadata_{field}")

    return errors

def check_mcq_contamination(user_content, assistant_content):
    has_choices = bool(re.search(r"[A-D]\.\s+.*?\n[A-D]\.\s+.*?\n[A-D]\.\s+.*?\n[A-D]\.\s+", user_content))
    is_short_answer = bool(re.match(r"^\s*(Đáp án|Answer|Kết quả)\s*:?\s*[A-D]\s*$", assistant_content, re.IGNORECASE))
    return has_choices or is_short_answer

def get_stats(lens):
    if not lens:
        return "N/A"
    lens = sorted(lens)
    n = len(lens)
    p50 = lens[int(n * 0.5)]
    p90 = lens[int(n * 0.9)]
    p99 = lens[int(n * 0.99)]
    return f"p50={p50}, p90={p90}, p99={p99}, max={max(lens)}"

def main():
    jsonl_path = Path("seed_exports/cloze_lm_retention_seed.jsonl")
    if not jsonl_path.exists():
        print(f"File not found: {jsonl_path}")
        return
    
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"=== [1] Total records: {len(records)} ===")
    
    domains = Counter()
    hashes = Counter()
    user_lens = []
    assistant_lens = []
    assistant_word_counts = []
    mcq_contaminations = 0
    invalid_records = []

    for i, rec in enumerate(records):
        errors = validate_record_schema(rec)
        if errors:
            invalid_records.append((i, errors))
            continue

        meta = rec["metadata"]
        domains[meta.get("domain", "unknown")] += 1
        hashes[meta["dedup_hash"]] += 1
        
        user_content = rec["messages"][0]["content"]
        assistant_content = rec["messages"][1]["content"]
        
        user_lens.append(len(user_content))
        assistant_lens.append(len(assistant_content))
        assistant_word_counts.append(len(assistant_content.split()))
        
        if check_mcq_contamination(user_content, assistant_content):
            mcq_contaminations += 1

    print("\n=== [2] Domain Distribution (Top 10) ===")
    for k, v in domains.most_common(10):
        print(f"  {k}: {v}")

    print("\n=== [3] Deduplication ===")
    dup_count = sum(1 for v in hashes.values() if v > 1)
    print(f"  Duplicate hashes: {dup_count}")

    print("\n=== [4] Length Statistics (Chars) ===")
    print(f"  User: {get_stats(user_lens)}")
    print(f"  Assistant: {get_stats(assistant_lens)}")

    print("\n=== [5] Assistant Word Counts ===")
    print(f"  {get_stats(assistant_word_counts)}")

    print("\n=== [6] QC Re-check ===")
    print(f"  MCQ-like contamination count: {mcq_contaminations}")
    print(f"  Invalid records (schema): {len(invalid_records)}")

    print("\n=== [7] Random Samples ===")
    random.seed(42)
    sample = random.sample(records, min(5, len(records)))
    for i, rec in enumerate(sample, 1):
        print(f"--- Sample {i} ---")
        user_txt = rec['messages'][0]['content'].replace('\n', ' ')
        print(f"User: {user_txt[:120]}...")
        print(f"Assistant: {rec['messages'][1]['content']}")

if __name__ == "__main__":
    main()
