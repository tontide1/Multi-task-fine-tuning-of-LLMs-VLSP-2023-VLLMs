# Comprehension Seed Raw Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ETL + QC scripts to export `seed_exports/comprehension_seed_raw.jsonl` with source-specific span checks, rejects/report, and a recheck validator. Downstream train đọc hiểu trong repo dùng **`comprehension_short_answer`** (`scripts/filter_comprehension_raw_uit.py`, `scripts/load_comprehension_short_answer_seed.py`), không dùng MCQ synthesis từ raw pool trong đường mặc định hiện tại vì chưa có API LLM/distractor generator đáng tin cậy. Better clean extractive QA than noisy generated MCQ.

**Architecture:** A single loader script (`scripts/load_comprehension_seed_raw.py`) handles dataset loading, QC, dedup, and export. A recheck script (`scripts/recheck_comprehension_seed_raw.py`) validates the exported JSONL. Unit tests cover the loader’s pure functions and the recheck validator.

**Tech Stack:** Python 3.11 (conda `nlp`), `datasets`, `pandas`, `tqdm`, `python-dotenv`, `unittest`.

---

## File Structure

- Create: `scripts/load_comprehension_seed_raw.py`
- Create: `scripts/recheck_comprehension_seed_raw.py`
- Create: `scripts/test_comprehension_seed_raw.py`

---

### Task 1: Normalize + Hash Utilities (TDD)

**Files:**
- Create: `scripts/test_comprehension_seed_raw.py`
- Create: `scripts/load_comprehension_seed_raw.py`

- [ ] **Step 1: Write failing tests for normalization and hash**

Create `scripts/test_comprehension_seed_raw.py` with:

```python
import unittest

from scripts.load_comprehension_seed_raw import (
    build_user_content,
    make_dedup_hash,
    normalize_for_hash,
)


class TestComprehensionSeedRawUtils(unittest.TestCase):
    def test_normalize_for_hash(self):
        raw = "  Xin\u00a0Chao   Ban "
        self.assertEqual(normalize_for_hash(raw), "xin chao ban")

    def test_make_dedup_hash_collapses_whitespace(self):
        h1 = make_dedup_hash("A  B", "C", "D")
        h2 = make_dedup_hash("A B", "C", "D")
        self.assertEqual(h1, h2)

    def test_build_user_content(self):
        self.assertEqual(
            build_user_content("CTX", "Q"),
            "Đoạn văn: CTX\n\nCâu hỏi: Q",
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: FAIL (ImportError or missing functions in `scripts/load_comprehension_seed_raw.py`).

- [ ] **Step 3: Implement minimal utilities**

Create `scripts/load_comprehension_seed_raw.py` with:

```python
"""
load_comprehension_seed_raw.py
===============================
ETL for comprehension_seed_raw.jsonl from UIT-ViQuAD2.0 and ShynBui datasets.
"""

import hashlib
import re


def normalize_for_hash(text: str) -> str:
    if text is None:
        return ""
    s = str(text).replace("\u00a0", " ")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def make_dedup_hash(context: str, question: str, answer_text: str) -> str:
    key = "\n".join([
        normalize_for_hash(context),
        normalize_for_hash(question),
        normalize_for_hash(answer_text),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def build_user_content(context: str, question: str) -> str:
    return f"Đoạn văn: {context}\n\nCâu hỏi: {question}"
```

- [ ] **Step 4: Re-run tests**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/load_comprehension_seed_raw.py scripts/test_comprehension_seed_raw.py
git commit -m "feat: add hash utilities for comprehension seed"
```

---

### Task 2: UIT Answer Variant Validation (TDD)

**Files:**
- Modify: `scripts/test_comprehension_seed_raw.py`
- Modify: `scripts/load_comprehension_seed_raw.py`

- [ ] **Step 1: Add failing tests for UIT answer variants**

Append to `scripts/test_comprehension_seed_raw.py`:

```python
from scripts.load_comprehension_seed_raw import (
    extract_valid_answer_variants,
    select_primary_variant,
    strict_span_match,
)


class TestUITAnswerVariants(unittest.TestCase):
    def test_strict_span_match(self):
        context = "Ha Noi la thu do"
        self.assertTrue(strict_span_match(context, "Ha Noi", 0))
        self.assertFalse(strict_span_match(context, "Sai", 0))

    def test_extract_valid_answer_variants(self):
        context = "Ha Noi la thu do"
        answers_text = ["Ha Noi", "thu do", "Sai"]
        answers_start = [0, 10, 99]
        variants = extract_valid_answer_variants(context, answers_text, answers_start)
        self.assertEqual(
            variants,
            [
                {"text": "Ha Noi", "answer_start": 0},
                {"text": "thu do", "answer_start": 10},
            ],
        )

    def test_select_primary_variant(self):
        variants = [
            {"text": "A", "answer_start": 0},
            {"text": "B", "answer_start": 2},
        ]
        self.assertEqual(select_primary_variant(variants), variants[0])
```

- [ ] **Step 2: Run tests to verify failure**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: FAIL (missing functions).

- [ ] **Step 3: Implement UIT answer utilities**

Update `scripts/load_comprehension_seed_raw.py`:

```python
def strict_span_match(context: str, answer_text: str, answer_start: int) -> bool:
    if context is None or answer_text is None:
        return False
    if not isinstance(answer_start, int):
        return False
    if answer_start < 0:
        return False
    end = answer_start + len(answer_text)
    if end > len(context):
        return False
    return context[answer_start:end] == answer_text


def extract_valid_answer_variants(context, answers_text, answers_start):
    variants = []
    if not isinstance(answers_text, (list, tuple)):
        return variants
    if not isinstance(answers_start, (list, tuple)):
        return variants

    for text, start in zip(answers_text, answers_start):
        if text is None:
            continue
        try:
            start_int = int(start)
        except (TypeError, ValueError):
            continue
        t = str(text)
        if strict_span_match(context, t, start_int):
            variants.append({"text": t, "answer_start": start_int})
    return variants


def select_primary_variant(variants):
    return variants[0] if variants else None
```

- [ ] **Step 4: Re-run tests**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/load_comprehension_seed_raw.py scripts/test_comprehension_seed_raw.py
git commit -m "feat: validate UIT answer variants"
```

---

### Task 3: ShynBui Normalization + Offset Recompute (TDD)

**Files:**
- Modify: `scripts/test_comprehension_seed_raw.py`
- Modify: `scripts/load_comprehension_seed_raw.py`

- [ ] **Step 1: Add failing tests for ShynBui normalization**

Append to `scripts/test_comprehension_seed_raw.py`:

```python
from scripts.load_comprehension_seed_raw import (
    normalize_shynbui_text,
    recompute_answer_start,
    strict_span_match,
)


class TestShynBuiNormalization(unittest.TestCase):
    def test_normalize_shynbui_text(self):
        self.assertEqual(normalize_shynbui_text("Ha_Noi"), "Ha Noi")

    def test_recompute_answer_start(self):
        context = "Ha Noi la thu do"
        answer = "thu do"
        self.assertEqual(recompute_answer_start(context, answer), 10)

    def test_strict_span_match_raw_with_underscore(self):
        context = "Ha_Noi la thu_do"
        answer = "thu_do"
        self.assertTrue(strict_span_match(context, answer, 10))
```

- [ ] **Step 2: Run tests to verify failure**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: FAIL (missing functions).

- [ ] **Step 3: Implement ShynBui normalization helpers**

Update `scripts/load_comprehension_seed_raw.py`:

```python
def normalize_shynbui_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).replace("\u00a0", " ")
    s = s.replace("_", " ")
    return s.strip()


def recompute_answer_start(context: str, answer_text: str) -> int:
    if context is None or answer_text is None:
        return -1
    return context.find(answer_text)
```

- [ ] **Step 4: Re-run tests**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/load_comprehension_seed_raw.py scripts/test_comprehension_seed_raw.py
git commit -m "feat: add ShynBui normalization helpers"
```

---

### Task 4: Length Filters + Reject Reasons (TDD)

**Files:**
- Modify: `scripts/test_comprehension_seed_raw.py`
- Modify: `scripts/load_comprehension_seed_raw.py`

- [ ] **Step 1: Add failing tests for length filters**

Append to `scripts/test_comprehension_seed_raw.py`:

```python
from scripts.load_comprehension_seed_raw import (
    apply_length_filters,
    MAX_CONTEXT_CHARS,
    MIN_CONTEXT_CHARS,
    MAX_ANSWER_CHARS,
)


class TestLengthFilters(unittest.TestCase):
    def test_context_too_short(self):
        reason = apply_length_filters("x" * (MIN_CONTEXT_CHARS - 1), "Hop le?", "a")
        self.assertEqual(reason, "context_too_short")

    def test_context_too_long(self):
        reason = apply_length_filters("x" * (MAX_CONTEXT_CHARS + 1), "Hop le?", "a")
        self.assertEqual(reason, "context_too_long")

    def test_answer_too_long(self):
        reason = apply_length_filters("x" * MIN_CONTEXT_CHARS, "Hop le?", "a" * (MAX_ANSWER_CHARS + 1))
        self.assertEqual(reason, "answer_too_long")
```

- [ ] **Step 2: Run tests to verify failure**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: FAIL (missing constants/functions).

- [ ] **Step 3: Implement length filters**

Update `scripts/load_comprehension_seed_raw.py`:

```python
MIN_CONTEXT_CHARS = 80
MAX_CONTEXT_CHARS = 8000
MIN_QUESTION_CHARS = 5
MAX_QUESTION_CHARS = 400
MAX_ANSWER_CHARS = 300


def apply_length_filters(context: str, question: str, answer_text: str):
    if context is None or len(context) < MIN_CONTEXT_CHARS:
        return "context_too_short"
    if len(context) > MAX_CONTEXT_CHARS:
        return "context_too_long"
    if question is None or len(question) < MIN_QUESTION_CHARS:
        return "question_too_short"
    if len(question) > MAX_QUESTION_CHARS:
        return "question_too_long"
    if answer_text is None:
        return "missing_answer"
    if len(answer_text) > MAX_ANSWER_CHARS:
        return "answer_too_long"
    return None
```

- [ ] **Step 4: Re-run tests**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/load_comprehension_seed_raw.py scripts/test_comprehension_seed_raw.py
git commit -m "feat: add length filters for comprehension seed"
```

---

### Task 5: Record + Reject Builders (TDD)

**Files:**
- Modify: `scripts/test_comprehension_seed_raw.py`
- Modify: `scripts/load_comprehension_seed_raw.py`

- [ ] **Step 1: Add failing tests for record builders**

Append to `scripts/test_comprehension_seed_raw.py`:

```python
from scripts.load_comprehension_seed_raw import (
    build_record,
    build_reject_record,
    QC_VERSION,
)


class TestRecordBuilders(unittest.TestCase):
    def test_build_record_fields(self):
        record = build_record(
            context="CTX",
            question="Q",
            answer_text="A",
            answer_start=0,
            source_dataset="ds",
            source_split="train",
            source_id="id",
            title=None,
            answer_variants=[],
            span_check_mode="strict_exact",
            original_answer_start=0,
        )
        self.assertEqual(record["context"], "CTX")
        self.assertEqual(record["question"], "Q")
        self.assertEqual(record["answer_text"], "A")
        self.assertEqual(record["messages"][0]["content"], "Đoạn văn: CTX\n\nCâu hỏi: Q")
        self.assertEqual(record["messages"][1]["content"], "A")
        self.assertEqual(record["metadata"]["answer_text"], "A")
        self.assertEqual(record["metadata"]["qc_version"], QC_VERSION)

    def test_build_reject_record(self):
        reject = build_reject_record(
            source_dataset="ds",
            source_split="train",
            source_id="id",
            reason="span_mismatch",
            context="CTX",
            question="Q",
            answer="A",
            answer_start=1,
        )
        self.assertEqual(reject["reason"], "span_mismatch")
        self.assertEqual(reject["context_preview"], "CTX")
```

- [ ] **Step 2: Run tests to verify failure**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: FAIL (missing builders/constants).

- [ ] **Step 3: Implement record builders**

Update `scripts/load_comprehension_seed_raw.py`:

```python
TASK_NAME = "comprehension_raw"
QC_VERSION = "comprehension_raw_source_specific_span_v1"


def build_record(
    *,
    context: str,
    question: str,
    answer_text: str,
    answer_start: int,
    source_dataset: str,
    source_split: str,
    source_id: str,
    title,
    answer_variants,
    span_check_mode: str,
    original_answer_start=None,
):
    return {
        "messages": [
            {"role": "user", "content": build_user_content(context, question)},
            {"role": "assistant", "content": answer_text},
        ],
        "context": context,
        "question": question,
        "answer_text": answer_text,
        "metadata": {
            "task": TASK_NAME,
            "source": "public",
            "source_dataset": source_dataset,
            "source_split": source_split,
            "source_id": source_id,
            "answer_start": answer_start,
            "original_answer_start": original_answer_start,
            "answer_text": answer_text,
            "answer_variants": answer_variants,
            "title": title,
            "language": "vi",
            "difficulty": "medium",
            "span_check_mode": span_check_mode,
            "dedup_hash": make_dedup_hash(context, question, answer_text),
            "qc_version": QC_VERSION,
        },
    }


def build_reject_record(
    *,
    source_dataset: str,
    source_split: str,
    source_id: str,
    reason: str,
    context: str,
    question: str,
    answer: str,
    answer_start,
):
    preview = (context or "")[:200]
    return {
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_id": source_id,
        "reason": reason,
        "context_preview": preview,
        "question": question,
        "answer": answer,
        "answer_start": answer_start,
    }
```

- [ ] **Step 4: Re-run tests**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/load_comprehension_seed_raw.py scripts/test_comprehension_seed_raw.py
git commit -m "feat: add record builders for comprehension seed"
```

---

### Task 6: Dedup Helper (TDD)

**Files:**
- Modify: `scripts/test_comprehension_seed_raw.py`
- Modify: `scripts/load_comprehension_seed_raw.py`

- [ ] **Step 1: Add failing tests for dedup**

Append to `scripts/test_comprehension_seed_raw.py`:

```python
from scripts.load_comprehension_seed_raw import dedup_records


class TestDedup(unittest.TestCase):
    def test_dedup_records(self):
        rec1 = build_record(
            context="CTX",
            question="Q",
            answer_text="A",
            answer_start=0,
            source_dataset="ds",
            source_split="train",
            source_id="id1",
            title=None,
            answer_variants=[],
            span_check_mode="strict_exact",
            original_answer_start=0,
        )
        rec2 = build_record(
            context="CTX",
            question="Q",
            answer_text="A",
            answer_start=0,
            source_dataset="ds",
            source_split="train",
            source_id="id2",
            title=None,
            answer_variants=[],
            span_check_mode="strict_exact",
            original_answer_start=0,
        )
        deduped, duplicates = dedup_records([rec1, rec2])
        self.assertEqual(len(deduped), 1)
        self.assertEqual(len(duplicates), 1)
```

- [ ] **Step 2: Run tests to verify failure**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: FAIL (missing function).

- [ ] **Step 3: Implement dedup helper**

Update `scripts/load_comprehension_seed_raw.py`:

```python
def dedup_records(records):
    seen = set()
    deduped = []
    duplicates = []
    for rec in records:
        h = rec["metadata"].get("dedup_hash")
        if h in seen:
            duplicates.append(rec)
            continue
        seen.add(h)
        deduped.append(rec)
    return deduped, duplicates
```

- [ ] **Step 4: Re-run tests**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/load_comprehension_seed_raw.py scripts/test_comprehension_seed_raw.py
git commit -m "feat: add dedup helper for comprehension seed"
```

---

### Task 7: Loader Pipeline (Implementation + Manual Check)

**Files:**
- Modify: `scripts/load_comprehension_seed_raw.py`

- [ ] **Step 1: Implement loader pipeline**

Replace `scripts/load_comprehension_seed_raw.py` content with:

```python
"""
load_comprehension_seed_raw.py
===============================
ETL for comprehension_seed_raw.jsonl from UIT-ViQuAD2.0 and ShynBui datasets.
"""

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm


load_dotenv()

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)


UIT_DATASET_ID = "taidng/UIT-ViQuAD2.0"
SHYNBUI_DATASET_ID = "ShynBui/Vietnamese_Reading_Comprehension_Dataset"

TASK_NAME = "comprehension_raw"
QC_VERSION = "comprehension_raw_source_specific_span_v1"

MIN_CONTEXT_CHARS = 80
MAX_CONTEXT_CHARS = 8000
MIN_QUESTION_CHARS = 5
MAX_QUESTION_CHARS = 400
MAX_ANSWER_CHARS = 300


def normalize_for_hash(text: str) -> str:
    if text is None:
        return ""
    s = str(text).replace("\u00a0", " ")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def make_dedup_hash(context: str, question: str, answer_text: str) -> str:
    key = "\n".join([
        normalize_for_hash(context),
        normalize_for_hash(question),
        normalize_for_hash(answer_text),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def build_user_content(context: str, question: str) -> str:
    return f"Đoạn văn: {context}\n\nCâu hỏi: {question}"


def strict_span_match(context: str, answer_text: str, answer_start: int) -> bool:
    if context is None or answer_text is None:
        return False
    if not isinstance(answer_start, int):
        return False
    if answer_start < 0:
        return False
    end = answer_start + len(answer_text)
    if end > len(context):
        return False
    return context[answer_start:end] == answer_text


def extract_valid_answer_variants(context, answers_text, answers_start):
    variants = []
    if not isinstance(answers_text, (list, tuple)):
        return variants
    if not isinstance(answers_start, (list, tuple)):
        return variants

    for text, start in zip(answers_text, answers_start):
        if text is None:
            continue
        try:
            start_int = int(start)
        except (TypeError, ValueError):
            continue
        t = str(text)
        if strict_span_match(context, t, start_int):
            variants.append({"text": t, "answer_start": start_int})
    return variants


def select_primary_variant(variants):
    return variants[0] if variants else None


def normalize_shynbui_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).replace("\u00a0", " ")
    s = s.replace("_", " ")
    return s.strip()


def recompute_answer_start(context: str, answer_text: str) -> int:
    if context is None or answer_text is None:
        return -1
    return context.find(answer_text)


def apply_length_filters(context: str, question: str, answer_text: str):
    if context is None or len(context) < MIN_CONTEXT_CHARS:
        return "context_too_short"
    if len(context) > MAX_CONTEXT_CHARS:
        return "context_too_long"
    if question is None or len(question) < MIN_QUESTION_CHARS:
        return "question_too_short"
    if len(question) > MAX_QUESTION_CHARS:
        return "question_too_long"
    if answer_text is None:
        return "missing_answer"
    if len(answer_text) > MAX_ANSWER_CHARS:
        return "answer_too_long"
    return None


def build_record(
    *,
    context: str,
    question: str,
    answer_text: str,
    answer_start: int,
    source_dataset: str,
    source_split: str,
    source_id: str,
    title,
    answer_variants,
    span_check_mode: str,
    original_answer_start=None,
):
    return {
        "messages": [
            {"role": "user", "content": build_user_content(context, question)},
            {"role": "assistant", "content": answer_text},
        ],
        "context": context,
        "question": question,
        "answer_text": answer_text,
        "metadata": {
            "task": TASK_NAME,
            "source": "public",
            "source_dataset": source_dataset,
            "source_split": source_split,
            "source_id": source_id,
            "answer_start": answer_start,
            "original_answer_start": original_answer_start,
            "answer_text": answer_text,
            "answer_variants": answer_variants,
            "title": title,
            "language": "vi",
            "difficulty": "medium",
            "span_check_mode": span_check_mode,
            "dedup_hash": make_dedup_hash(context, question, answer_text),
            "qc_version": QC_VERSION,
        },
    }


def build_reject_record(
    *,
    source_dataset: str,
    source_split: str,
    source_id: str,
    reason: str,
    context: str,
    question: str,
    answer: str,
    answer_start,
):
    preview = (context or "")[:200]
    return {
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_id": source_id,
        "reason": reason,
        "context_preview": preview,
        "question": question,
        "answer": answer,
        "answer_start": answer_start,
    }


def dedup_records(records):
    seen = set()
    deduped = []
    duplicates = []
    for rec in records:
        h = rec["metadata"].get("dedup_hash")
        if h in seen:
            duplicates.append(rec)
            continue
        seen.add(h)
        deduped.append(rec)
    return deduped, duplicates


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_str(x):
    return "" if x is None else str(x)


def process_uit_row(row: dict, split_name: str):
    context = safe_str(row.get("context"))
    question = safe_str(row.get("question"))
    is_impossible = bool(row.get("is_impossible"))
    answers = row.get("answers") or {}
    answers_text = answers.get("text") if isinstance(answers, dict) else None
    answers_start = answers.get("answer_start") if isinstance(answers, dict) else None
    source_id = row.get("id") or row.get("uit_id") or f"{split_name}-{row.get('_row_index')}"
    title = row.get("title")

    if is_impossible:
        return None, "is_impossible"

    if not context:
        return None, "missing_context"
    if not question:
        return None, "missing_question"

    length_reason = apply_length_filters(context, question, safe_str(answers_text[0] if isinstance(answers_text, list) and answers_text else ""))
    if length_reason:
        return None, length_reason

    variants = extract_valid_answer_variants(context, answers_text, answers_start)
    if not variants:
        return None, "no_valid_answer_variant"

    primary = select_primary_variant(variants)
    record = build_record(
        context=context,
        question=question,
        answer_text=primary["text"],
        answer_start=primary["answer_start"],
        source_dataset=UIT_DATASET_ID,
        source_split=split_name,
        source_id=str(source_id),
        title=title,
        answer_variants=variants,
        span_check_mode="strict_exact",
        original_answer_start=primary["answer_start"],
    )
    return record, None


def process_shynbui_row(row: dict, split_name: str):
    context_raw = safe_str(row.get("context"))
    question_raw = safe_str(row.get("question"))
    answer_raw = safe_str(row.get("answer"))
    source_id = row.get("id") or f"{split_name}-{row.get('_row_index')}"
    answer_start_raw = row.get("answer_start")

    if not context_raw:
        return None, "missing_context"
    if not question_raw:
        return None, "missing_question"
    if not answer_raw:
        return None, "missing_answer"

    length_reason = apply_length_filters(context_raw, question_raw, answer_raw)
    if length_reason:
        return None, length_reason

    try:
        answer_start_raw = int(answer_start_raw)
    except (TypeError, ValueError):
        return None, "invalid_answer_start"

    if answer_start_raw < 0 or answer_start_raw + len(answer_raw) > len(context_raw):
        return None, "answer_start_out_of_range"

    if not strict_span_match(context_raw, answer_raw, answer_start_raw):
        return None, "span_mismatch"

    context = normalize_shynbui_text(context_raw)
    question = normalize_shynbui_text(question_raw)
    answer_text = normalize_shynbui_text(answer_raw)
    answer_start = recompute_answer_start(context, answer_text)
    if answer_start < 0:
        return None, "answer_not_found_after_normalize"

    record = build_record(
        context=context,
        question=question,
        answer_text=answer_text,
        answer_start=answer_start,
        source_dataset=SHYNBUI_DATASET_ID,
        source_split=split_name,
        source_id=str(source_id),
        title=None,
        answer_variants=[],
        span_check_mode="raw_exact_then_normalized_find",
        original_answer_start=answer_start_raw,
    )
    return record, None


def main():
    repo_root = Path(__file__).parent.parent
    out_dir = repo_root / "seed_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "comprehension_seed_raw.jsonl"
    out_rejects_jsonl = out_dir / "comprehension_seed_raw_rejects.jsonl"
    out_report_json = out_dir / "comprehension_seed_raw_report.json"

    print("[1/5] Loading UIT-ViQuAD2.0 ...")
    try:
        ds_uit = load_dataset(UIT_DATASET_ID)
    except Exception as e:
        print(f"[ERROR] Failed to load {UIT_DATASET_ID}: {e}")
        sys.exit(1)

    print("[2/5] Loading ShynBui dataset ...")
    try:
        ds_shyn = load_dataset(SHYNBUI_DATASET_ID)
    except Exception as e:
        print(f"[ERROR] Failed to load {SHYNBUI_DATASET_ID}: {e}")
        sys.exit(1)

    kept = []
    rejects = []
    reject_reasons = Counter()
    kept_by_dataset = Counter()
    kept_by_split = Counter()

    def handle_reject(reason, row, split_name, dataset_id):
        reject_reasons[reason] += 1
        rejects.append(build_reject_record(
            source_dataset=dataset_id,
            source_split=split_name,
            source_id=str(row.get("id") or row.get("uit_id") or f"{split_name}-{row.get('_row_index')}"),
            reason=reason,
            context=safe_str(row.get("context")),
            question=safe_str(row.get("question")),
            answer=safe_str(row.get("answer") or (row.get("answers") or {}).get("text", "")),
            answer_start=row.get("answer_start") or (row.get("answers") or {}).get("answer_start"),
        ))

    print("[3/5] Processing UIT-ViQuAD2.0 ...")
    for split_name, split_ds in ds_uit.items():
        df = split_ds.to_pandas()
        df["_row_index"] = df.index
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"UIT {split_name}"):
            record, reason = process_uit_row(row.to_dict(), split_name)
            if reason:
                handle_reject(reason, row, split_name, UIT_DATASET_ID)
            else:
                kept.append(record)
                kept_by_dataset[UIT_DATASET_ID] += 1
                kept_by_split[split_name] += 1

    print("[4/5] Processing ShynBui ...")
    for split_name, split_ds in ds_shyn.items():
        df = split_ds.to_pandas()
        df["_row_index"] = df.index
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"ShynBui {split_name}"):
            record, reason = process_shynbui_row(row.to_dict(), split_name)
            if reason:
                handle_reject(reason, row, split_name, SHYNBUI_DATASET_ID)
            else:
                kept.append(record)
                kept_by_dataset[SHYNBUI_DATASET_ID] += 1
                kept_by_split[split_name] += 1

    print("[5/5] Deduplicating and writing outputs ...")
    deduped, duplicates = dedup_records(kept)
    reject_reasons["duplicate"] += len(duplicates)
    for dup in duplicates:
        rejects.append(build_reject_record(
            source_dataset=dup["metadata"]["source_dataset"],
            source_split=dup["metadata"]["source_split"],
            source_id=dup["metadata"]["source_id"],
            reason="duplicate",
            context=dup["context"],
            question=dup["question"],
            answer=dup["answer_text"],
            answer_start=dup["metadata"]["answer_start"],
        ))

    write_jsonl(out_jsonl, deduped)
    write_jsonl(out_rejects_jsonl, rejects)

    report = {
        "total_loaded": len(kept) + len(rejects),
        "total_kept": len(deduped),
        "total_rejected": len(rejects),
        "kept_by_dataset": {str(k): int(v) for k, v in kept_by_dataset.items()},
        "kept_by_split": {str(k): int(v) for k, v in kept_by_split.items()},
        "reject_reasons": {str(k): int(v) for k, v in reject_reasons.items()},
        "duplicate_count": len(duplicates),
        "split_policy": "raw_pool_all_source_splits_loaded; downstream split required",
        "qc_version": QC_VERSION,
    }

    with out_report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] {out_jsonl} | records={len(deduped)}")
    print(f"[OK] {out_rejects_jsonl} | rejects={len(rejects)}")
    print(f"[OK] {out_report_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run loader manually**

Run:
```
python scripts/load_comprehension_seed_raw.py
```

Expected: `[OK]` lines for JSONL and report files (counts may vary).

- [ ] **Step 3: Commit**

```bash
git add scripts/load_comprehension_seed_raw.py
git commit -m "feat: add comprehension raw seed loader"
```

---

### Task 8: Recheck Script (TDD + Manual Run)

**Files:**
- Modify: `scripts/test_comprehension_seed_raw.py`
- Create: `scripts/recheck_comprehension_seed_raw.py`

- [ ] **Step 1: Add failing tests for recheck validator**

Append to `scripts/test_comprehension_seed_raw.py`:

```python
from scripts.recheck_comprehension_seed_raw import validate_record_schema


class TestRecheckValidator(unittest.TestCase):
    def test_validate_record_schema_valid(self):
        record = build_record(
            context="CTX",
            question="Q",
            answer_text="A",
            answer_start=0,
            source_dataset="ds",
            source_split="train",
            source_id="id",
            title=None,
            answer_variants=[],
            span_check_mode="strict_exact",
            original_answer_start=0,
        )
        errors = validate_record_schema(record)
        self.assertEqual(errors, [])

    def test_validate_record_schema_missing_fields(self):
        errors = validate_record_schema({"messages": []})
        self.assertTrue(len(errors) > 0)
```

- [ ] **Step 2: Run tests to verify failure**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: FAIL (missing recheck module).

- [ ] **Step 3: Implement recheck script**

Create `scripts/recheck_comprehension_seed_raw.py`:

```python
"""
recheck_comprehension_seed_raw.py
=================================
Validate exported comprehension_seed_raw.jsonl.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


REQUIRED_METADATA_FIELDS = [
    "task",
    "source",
    "source_dataset",
    "source_split",
    "source_id",
    "answer_start",
    "answer_text",
    "span_check_mode",
    "dedup_hash",
    "qc_version",
]


def format_user_content(context: str, question: str) -> str:
    return f"Đoạn văn: {context}\n\nCâu hỏi: {question}"


def validate_record_schema(record: dict):
    errors = []
    messages = record.get("messages")
    if not isinstance(messages, list):
        errors.append("messages_not_list")
        return errors
    if len(messages) != 2:
        errors.append("messages_length_not_2")
    else:
        if messages[0].get("role") != "user":
            errors.append("first_role_not_user")
        if messages[1].get("role") != "assistant":
            errors.append("second_role_not_assistant")
        if not str(messages[0].get("content", "")).strip():
            errors.append("user_content_empty")
        if not str(messages[1].get("content", "")).strip():
            errors.append("assistant_content_empty")

    context = record.get("context")
    question = record.get("question")
    answer_text = record.get("answer_text")

    if not context:
        errors.append("missing_context")
    if not question:
        errors.append("missing_question")
    if not answer_text:
        errors.append("missing_answer")

    if context and question:
        expected_user = format_user_content(context, question)
        if messages and messages[0].get("content") != expected_user:
            errors.append("user_content_format_mismatch")

    if answer_text and messages and messages[1].get("content") != answer_text:
        errors.append("assistant_answer_mismatch")

    metadata = record.get("metadata", {})
    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            errors.append(f"missing_metadata_{field}")

    if answer_text and metadata.get("answer_text") != answer_text:
        errors.append("metadata_answer_text_mismatch")

    if answer_text and context and answer_text not in context:
        errors.append("answer_not_in_context")

    if metadata.get("span_check_mode") == "strict_exact":
        answer_start = metadata.get("answer_start")
        if isinstance(answer_start, int) and context and answer_text:
            end = answer_start + len(answer_text)
            if end > len(context) or context[answer_start:end] != answer_text:
                errors.append("strict_span_mismatch")
        else:
            errors.append("strict_span_invalid_start")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5)
    args = parser.parse_args()

    jsonl_path = Path(__file__).parent.parent / "seed_exports" / "comprehension_seed_raw.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} does not exist. Run loader first.")

    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"=== [1] Total records: {len(records)} ===\n")

    rows = []
    invalid = []
    for i, rec in enumerate(records):
        errors = validate_record_schema(rec)
        if errors:
            invalid.append({"index": i, "errors": errors})
        meta = rec.get("metadata", {})
        rows.append({
            "source_dataset": meta.get("source_dataset"),
            "source_split": meta.get("source_split"),
            "dedup_hash": meta.get("dedup_hash"),
        })

    df = pd.DataFrame(rows)
    print("=== [2] Source distribution ===")
    print(df["source_dataset"].value_counts(dropna=False).to_string())
    print()
    print("=== [3] Split distribution ===")
    print(df["source_split"].value_counts(dropna=False).to_string())
    print()

    print("=== [4] Schema validation ===")
    print(f"Invalid records: {len(invalid)}")
    if invalid:
        for inv in invalid[:5]:
            print(f"  - index={inv['index']}: {inv['errors']}")
    print()

    print("=== [5] Dedup hash uniqueness ===")
    hashes = df["dedup_hash"].dropna().tolist()
    unique_hashes = set(hashes)
    print(f"Total hashes  : {len(hashes)}")
    print(f"Unique hashes : {len(unique_hashes)}")
    print(f"Duplicates    : {len(hashes) - len(unique_hashes)}")
    print()

    print(f"=== [6] Random sample ({args.sample}) ===")
    random.seed(42)
    sample = random.sample(records, min(args.sample, len(records)))
    for i, rec in enumerate(sample, 1):
        print(f"--- sample {i} ---")
        print("[user]")
        print(rec["messages"][0]["content"])
        print("[assistant]")
        print(rec["messages"][1]["content"])
        print("[metadata]")
        print(json.dumps(rec.get("metadata", {}), ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Re-run tests**

Run:
```
python -m unittest scripts.test_comprehension_seed_raw -v
```

Expected: PASS.

- [ ] **Step 5: Run recheck manually**

Run:
```
python scripts/recheck_comprehension_seed_raw.py --sample 5
```

Expected: Printed counts, schema validation summary, and 5 samples.

- [ ] **Step 6: Commit**

```bash
git add scripts/recheck_comprehension_seed_raw.py scripts/test_comprehension_seed_raw.py
git commit -m "feat: add recheck for comprehension raw seed"
```

---

### Task 9: Update Seed Exports (Manual)

**Files:**
- Create (generated): `seed_exports/comprehension_seed_raw.jsonl`
- Create (generated): `seed_exports/comprehension_seed_raw_rejects.jsonl`
- Create (generated): `seed_exports/comprehension_seed_raw_report.json`

- [ ] **Step 1: Run loader**

Run:
```
python scripts/load_comprehension_seed_raw.py
```

- [ ] **Step 2: Run recheck**

Run:
```
python scripts/recheck_comprehension_seed_raw.py --sample 5
```

- [ ] **Step 3: Commit (optional, if you want outputs tracked)**

```bash
git add seed_exports/comprehension_seed_raw.jsonl seed_exports/comprehension_seed_raw_rejects.jsonl seed_exports/comprehension_seed_raw_report.json
git commit -m "data: export comprehension raw seed"
```

---

## Self-Review Checklist

- **Spec coverage:** loader + recheck + rejects/report + source-specific QC + length filters + split policy are all implemented.
- **Placeholder scan:** no TODOs or vague steps remain.
- **Type consistency:** function names used in tests match loader/recheck definitions.
