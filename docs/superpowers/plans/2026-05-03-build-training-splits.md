# Build Training Splits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a monolithic script to load, cross-deduplicate, and deterministically split 5 seed datasets into `train_all`, `val_all`, `shadow_eval`, and probe files using fixed percentage ratios.

**Architecture:** A single Python script `scripts/build_training_split.py` handling loading, SHA256 deduplication based on prompt/response, deterministic shuffling (seed=42), percentage-based slicing, and JSONL output generation. A corresponding test file `scripts/test_build_training_split.py` will verify core logic.

**Tech Stack:** Python 3.11, `json`, `random`, `pathlib`, `hashlib`, `unittest`.

---

### Task 1: Create Test Structure and Helper Functions

**Files:**
- Create: `scripts/test_build_training_split.py`
- Create: `scripts/build_training_split.py`

- [ ] **Step 1: Write the failing tests for deduplication and bucketing**

Create `scripts/test_build_training_split.py`:

```python
import unittest
from scripts.build_training_split import compute_hash, deduplicate_records, rebucket_records

class TestBuildTrainingSplit(unittest.TestCase):
    def test_compute_hash(self):
        record = {
            "messages": [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"}
            ]
        }
        hash_val = compute_hash(record)
        self.assertTrue(isinstance(hash_val, str))
        self.assertTrue(len(hash_val) > 0)
        
        # Changing content should change hash
        record2 = {"messages": [{"role": "user", "content": "Question 2"}, {"role": "assistant", "content": "Answer 1"}]}
        self.assertNotEqual(hash_val, compute_hash(record2))

    def test_deduplicate_records(self):
        records = [
            {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}], "id": 1},
            {"messages": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}], "id": 2},
            {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}], "id": 3}, # Duplicate of 1
        ]
        deduped = deduplicate_records(records)
        self.assertEqual(len(deduped), 2)
        self.assertEqual([r["id"] for r in deduped], [1, 2])

    def test_rebucket_records(self):
        records = [
            {"metadata": {"task": "task_a"}, "id": 1},
            {"metadata": {"task": "task_b"}, "id": 2},
            {"metadata": {"task": "task_a"}, "id": 3},
        ]
        buckets = rebucket_records(records)
        self.assertEqual(list(buckets.keys()), ["task_a", "task_b"])
        self.assertEqual(len(buckets["task_a"]), 2)
        self.assertEqual(len(buckets["task_b"]), 1)

if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest scripts.test_build_training_split -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write minimal implementation for helpers**

Create `scripts/build_training_split.py`:

```python
import hashlib
from collections import defaultdict

def compute_hash(record: dict) -> str:
    """Compute a SHA256 hash based on user and assistant messages."""
    content_str = ""
    for msg in record.get("messages", []):
        content_str += f"[{msg.get('role', '')}]:{msg.get('content', '')}|"
    return hashlib.sha256(content_str.encode('utf-8')).hexdigest()

def deduplicate_records(records: list[dict]) -> list[dict]:
    """Remove records with duplicate prompt/response hashes. Keep first seen."""
    seen_hashes = set()
    deduped = []
    for record in records:
        h = compute_hash(record)
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(record)
    return deduped

def rebucket_records(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by their metadata.task value."""
    buckets = defaultdict(list)
    for record in records:
        task = record.get("metadata", {}).get("task", "unknown")
        buckets[task].append(record)
    return buckets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest scripts.test_build_training_split -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/build_training_split.py scripts/test_build_training_split.py
git commit -m "test: add deduplication and bucketing logic for training split"
```

### Task 2: Implement Deterministic Splitting Logic

**Files:**
- Modify: `scripts/test_build_training_split.py`
- Modify: `scripts/build_training_split.py`

- [ ] **Step 1: Write the failing tests for splitting**

Append to `scripts/test_build_training_split.py` inside `TestBuildTrainingSplit`:

```python
    def test_split_group_a(self):
        from scripts.build_training_split import split_group_a
        bucket = [{"id": i} for i in range(100)]
        train, val, shadow = split_group_a(bucket, train_pct=0.90, val_pct=0.05, shadow_pct=0.05)
        self.assertEqual(len(train), 90)
        self.assertEqual(len(val), 5)
        self.assertEqual(len(shadow), 5)
        self.assertEqual(len(train) + len(val) + len(shadow), 100)

    def test_split_group_b(self):
        from scripts.build_training_split import split_group_b
        bucket = [{"id": i} for i in range(100)]
        train, probe = split_group_b(bucket, train_pct=0.95, probe_pct=0.05)
        self.assertEqual(len(train), 95)
        self.assertEqual(len(probe), 5)
        self.assertEqual(len(train) + len(probe), 100)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest scripts.test_build_training_split -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation for splitting**

Append to `scripts/build_training_split.py`:

```python
import random

def split_group_a(bucket: list[dict], train_pct: float, val_pct: float, shadow_pct: float) -> tuple[list[dict], list[dict], list[dict]]:
    """Split benchmark proxy tasks into train, val, and shadow."""
    total = len(bucket)
    train_idx = int(total * train_pct)
    val_idx = train_idx + int(total * val_pct)
    
    train_slice = bucket[:train_idx]
    val_slice = bucket[train_idx:val_idx]
    shadow_slice = bucket[val_idx:]
    
    return train_slice, val_slice, shadow_slice

def split_group_b(bucket: list[dict], train_pct: float, probe_pct: float) -> tuple[list[dict], list[dict]]:
    """Split retention tasks into train and probe."""
    total = len(bucket)
    train_idx = int(total * train_pct)
    
    train_slice = bucket[:train_idx]
    probe_slice = bucket[train_idx:]
    
    return train_slice, probe_slice
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest scripts.test_build_training_split -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/build_training_split.py scripts/test_build_training_split.py
git commit -m "feat: implement deterministic splitting logic"
```

### Task 3: Implement Main Pipeline and File I/O

**Files:**
- Modify: `scripts/build_training_split.py`

- [ ] **Step 1: Implement main pipeline function**

Append to `scripts/build_training_split.py`:

```python
import json
import argparse
from pathlib import Path

def load_jsonl(filepath: Path) -> list[dict]:
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def write_jsonl(records: list[dict], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    parser = argparse.add_argument_group("Build Training Splits")
    # For simplicity in this script, we'll hardcode paths, but argparse can be used later if needed.
    
    seed_dir = Path(__file__).resolve().parent.parent / "seed_exports"
    output_dir = seed_dir / "splits"
    
    input_files = [
        "comprehension_short_answer_seed.jsonl",
        "exams_mcq_seed.jsonl",
        "wiki_mcq_seed_final.jsonl",
        "instruction_retention_seed.jsonl",
        "cloze_lm_retention_seed.jsonl"
    ]
    
    print("Loading datasets...")
    all_records = []
    for filename in input_files:
        filepath = seed_dir / filename
        if filepath.exists():
            print(f"  Loaded {filename}")
            all_records.extend(load_jsonl(filepath))
        else:
            print(f"  WARNING: File not found: {filepath}")

    total_loaded = len(all_records)
    print(f"\nTotal records loaded: {total_loaded}")
    
    print("Deduplicating...")
    deduped_records = deduplicate_records(all_records)
    print(f"Total after deduplication: {len(deduped_records)} (Removed {total_loaded - len(deduped_records)})")
    
    buckets = rebucket_records(deduped_records)
    
    # Set seed for deterministic shuffle
    random.seed(42)
    
    # Initialize output containers
    train_all = []
    val_all = []
    shadow_eval = []
    instruction_probe = []
    cloze_probe = []
    
    report = {"total_loaded": total_loaded, "total_deduped": len(deduped_records), "tasks": {}}
    
    print("\nSplitting buckets...")
    for task, bucket in buckets.items():
        random.shuffle(bucket)
        report["tasks"][task] = {"total_clean": len(bucket)}
        
        if task in ["exams_mcq", "wiki_mcq", "comprehension_short_answer"]:
            train_slice, val_slice, shadow_slice = split_group_a(bucket, 0.90, 0.05, 0.05)
            train_all.extend(train_slice)
            val_all.extend(val_slice)
            shadow_eval.extend(shadow_slice)
            report["tasks"][task].update({
                "train": len(train_slice), "val": len(val_slice), "shadow": len(shadow_slice)
            })
        elif task == "instruction_retention":
            train_slice, probe_slice = split_group_b(bucket, 0.95, 0.05)
            train_all.extend(train_slice)
            instruction_probe.extend(probe_slice)
            report["tasks"][task].update({"train": len(train_slice), "probe": len(probe_slice)})
        elif task == "cloze_lm_retention":
            train_slice, probe_slice = split_group_b(bucket, 0.95, 0.05)
            train_all.extend(train_slice)
            cloze_probe.extend(probe_slice)
            report["tasks"][task].update({"train": len(train_slice), "probe": len(probe_slice)})
        else:
            print(f"WARNING: Unknown task '{task}'. Skipping.")
            report["tasks"][task].update({"status": "skipped"})
            
    print("\nWriting splits...")
    write_jsonl(train_all, output_dir / "train_all.jsonl")
    write_jsonl(val_all, output_dir / "val_all.jsonl")
    write_jsonl(shadow_eval, output_dir / "shadow_eval.jsonl")
    write_jsonl(instruction_probe, output_dir / "instruction_probe.jsonl")
    write_jsonl(cloze_probe, output_dir / "cloze_probe.jsonl")
    
    with open(output_dir / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print("Done! Report saved to split_report.json.")
    print(f"Final sizes - train_all: {len(train_all)}, val_all: {len(val_all)}, shadow_eval: {len(shadow_eval)}, instruction_probe: {len(instruction_probe)}, cloze_probe: {len(cloze_probe)}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify no regressions**

Run: `python -m unittest scripts.test_build_training_split -v`
Expected: PASS

- [ ] **Step 3: Execute the script**

Run: `python scripts/build_training_split.py`
Expected: Script successfully loads available files, prints deduplication stats, and generates output files in `seed_exports/splits/`.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_training_split.py
git commit -m "feat: complete end-to-step training split pipeline"
```
