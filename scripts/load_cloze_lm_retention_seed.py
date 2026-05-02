
"""
load_cloze_lm_retention_seed.py
==================================
ETL for cloze_lm_retention_seed.jsonl from VTSNLP/vietnamese_curated_dataset.
Target: ~10,000 high-quality samples for next-word prediction.
"""

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Constants
DATASET_ID = "VTSNLP/vietnamese_curated_dataset"
TASK_NAME = "cloze_lm_retention"
OUTPUT_FILE = "seed_exports/cloze_lm_retention_seed.jsonl"
REJECTS_FILE = "seed_exports/cloze_lm_retention_seed_rejects.jsonl"
REPORT_FILE = "seed_exports/cloze_lm_retention_seed_report.json"

CAP_CLOZE = 10000
MIN_WORDS = 20
MAX_WORDS = 100 # Tighten for quality as per brainstorm

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u00a0", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_cloze_pair(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS):
    """
    Extracts a cloze pair (prompt, target) from a text.
    Returns (user_prompt, assistant_response, reject_reason).
    """
    if not text:
        return None, None, "empty_text"
    
    text = normalize_text(text)
    words = text.split()
    
    if len(words) < min_words:
        return None, None, f"too_short_{len(words)}"
    if len(words) > max_words:
        return None, None, f"too_long_{len(words)}"

    target_word_raw = words[-1]
    # Clean target: remove trailing punctuation
    target_word = target_word_raw.rstrip(".,!?\"':;")
    
    if not target_word:
        return None, None, "empty_target_after_cleaning"
    
    # Check if target is alphabetic (including Vietnamese characters)
    if not target_word.isalpha():
         return None, None, "invalid_target_word"

    # Context is everything but the last word
    context = " ".join(words[:-1])
    
    user_prompt = f"Điền từ tiếp theo vào chỗ trống:\n\n{context}..."
    assistant_response = target_word
    
    return user_prompt, assistant_response, None

def make_dedup_hash(user: str, assistant: str) -> str:
    key = f"{user}|{assistant}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def main():
    print(f"Loading {DATASET_ID} in streaming mode...")
    try:
        # Load only the first shard to avoid timeouts and excessive downloads
        # Total rows ~12M, 132 shards -> ~92k rows per shard. More than enough for CAP_CLOZE=10000.
        ds = load_dataset(DATASET_ID, split="train", streaming=True, data_files="train-00000-of-00132.parquet")
    except Exception as e:
        print(f"Error loading dataset shard: {e}")
        try:
            print("Retrying with full dataset...")
            ds = load_dataset(DATASET_ID, split="train", streaming=True)
        except Exception as e2:
            print(f"Error loading full dataset: {e2}")
            sys.exit(1)

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    seen_hashes = set()
    records = []
    rejects = []
    
    stats = {
        "total_processed": 0,
        "accepted": 0,
        "rejected": 0,
        "reject_reasons": {}
    }

    pbar = tqdm(total=CAP_CLOZE, desc="Processing")
    
    try:
        for item in ds:
            stats["total_processed"] += 1
            text = item.get("text", "")
            source_id = item.get("id", "unknown")
            domain = item.get("domain", "unknown")
            
            user_prompt, assistant_response, reject_reason = extract_cloze_pair(text)
            
            if reject_reason:
                stats["rejected"] += 1
                stats["reject_reasons"][reject_reason] = stats["reject_reasons"].get(reject_reason, 0) + 1
                if len(rejects) < 1000: # Cap rejects log
                    rejects.append({
                        "source_id": source_id,
                        "reject_reason": reject_reason,
                        "text_preview": text[:100]
                    })
                continue
                
            dedup_hash = make_dedup_hash(user_prompt, assistant_response)
            if dedup_hash in seen_hashes:
                stats["rejected"] += 1
                stats["reject_reasons"]["duplicate"] = stats["reject_reasons"].get("duplicate", 0) + 1
                continue
                
            seen_hashes.add(dedup_hash)
            
            record = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ],
                "metadata": {
                    "task": TASK_NAME,
                    "source": "public",
                    "source_dataset": DATASET_ID,
                    "source_id": str(source_id),
                    "domain": domain,
                    "language": "vi",
                    "difficulty": "medium",
                    "dedup_hash": dedup_hash
                }
            }
            records.append(record)
            stats["accepted"] += 1
            pbar.update(1)
            
            if stats["accepted"] >= CAP_CLOZE:
                break
    except Exception as e:
        print(f"Error during iteration: {e}")

    pbar.close()
    
    print(f"Saving {len(records)} records to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Saving {len(rejects)} sample rejects to {REJECTS_FILE}...")
    with open(REJECTS_FILE, "w", encoding="utf-8") as f:
        for r in rejects:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Saving report to {REPORT_FILE}...")
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
