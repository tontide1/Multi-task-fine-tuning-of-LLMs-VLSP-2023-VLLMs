> **Trạng thái (2026-05): archive, không còn là đường train mặc định.** Train đọc hiểu hiện dùng **`comprehension_short_answer`**; xem `README.md` và `docs/plan/overview.md`. Hiện chưa có API LLM/distractor generator đáng tin cậy để chuyển short answer thành MCQ sạch, nên hướng thực dụng là **Better clean extractive QA than noisy generated MCQ**. Plan sau đây chỉ là kế hoạch implementation cho pipeline MCQ lịch sử.

# Comprehension MCQ Seed Final Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the archived offline JSONL pipeline that turns `seed_exports/comprehension_seed_raw.jsonl` into a final UIT-only MCQ seed file with rule QC, solver QC, leakage checks, reports, and rejects. This is not the current train path.

**Architecture:** Keep model work outside the repo and use deterministic scripts only. A small shared helper module holds text normalization, JSONL IO, MCQ formatting, and raw-response parsing. Each pipeline stage is a single-purpose script with its own reject/report files so the boundary between deterministic transformation and external model output stays explicit.

**Tech Stack:** Python 3.11, standard library (`argparse`, `json`, `hashlib`, `re`, `random`, `difflib`, `pathlib`, `collections`, `tempfile`, `unittest`).

---

## File Structure

- Create: `scripts/comprehension_mcq_seed_common.py`
- Create: `scripts/filter_comprehension_raw_uit.py`
- Create: `scripts/prepare_comprehension_mcq_generation.py`
- Create: `scripts/build_comprehension_mcq_candidates.py`
- Create: `scripts/qc_comprehension_mcq_seed.py`
- Create: `scripts/prepare_comprehension_mcq_solver.py`
- Create: `scripts/apply_comprehension_mcq_solver_qc.py`
- Create: `scripts/check_comprehension_mcq_leakage.py`
- Create: `scripts/finalize_comprehension_mcq_seed.py`
- Create: `scripts/recheck_comprehension_mcq_seed.py`
- Create: `scripts/test_comprehension_mcq_seed.py`

Responsibilities:
- `scripts/comprehension_mcq_seed_common.py`: shared pure helpers for hash/format/JSON parsing/MCQ normalization.
- `scripts/filter_comprehension_raw_uit.py`: keep only UIT raw records and write the UIT-only raw pool.
- `scripts/prepare_comprehension_mcq_generation.py`: export distractor-generation requests and prefilter rejects.
- `scripts/build_comprehension_mcq_candidates.py`: merge raw UIT rows with external distractor output and build MCQ candidates.
- `scripts/qc_comprehension_mcq_seed.py`: rule-based MCQ QC and reject/report export.
- `scripts/prepare_comprehension_mcq_solver.py`: export solver requests from rule-checked MCQs.
- `scripts/apply_comprehension_mcq_solver_qc.py`: parse solver outputs and keep only unambiguous correct rows.
- `scripts/check_comprehension_mcq_leakage.py`: lexical leakage check against benchmark data.
- `scripts/finalize_comprehension_mcq_seed.py`: final invariants gate and final JSONL/report/sample export.
- `scripts/recheck_comprehension_mcq_seed.py`: read-only validator/reporting for final or intermediate MCQ JSONL.
- `scripts/test_comprehension_mcq_seed.py`: unit/integration tests for all helpers and stage scripts.

---

### Task 1: Shared Helpers and Test Harness

**Files:**
- Create: `scripts/comprehension_mcq_seed_common.py`
- Create: `scripts/test_comprehension_mcq_seed.py`

- [ ] **Step 1: Write the failing tests for shared helpers**

Create `scripts/test_comprehension_mcq_seed.py` with tests that pin the common contracts used by every stage:

```python
import unittest

from scripts.comprehension_mcq_seed_common import (
    build_mcq_user_content,
    canonical_choice_text,
    compute_context_hash,
    extract_json_object,
    make_dedup_hash,
    make_mcq_dedup_hash,
    normalize_for_hash,
    split_mcq_user_content,
    stable_choice_labels,
)


class TestSharedHelpers(unittest.TestCase):
    def test_normalize_for_hash(self):
        self.assertEqual(normalize_for_hash("  Xin\u00a0Chao   Ban "), "xin chao ban")

    def test_build_mcq_user_content(self):
        self.assertEqual(
            build_mcq_user_content("CTX", "Q", ["A1", "A2", "A3", "A4"]),
            "Đọc đoạn văn sau và chọn đáp án đúng.\n\nĐoạn văn: CTX\n\nCâu hỏi: Q\nA. A1\nB. A2\nC. A3\nD. A4",
        )

    def test_split_mcq_user_content(self):
        context, question, choices = split_mcq_user_content(
            "Đọc đoạn văn sau và chọn đáp án đúng.\n\nĐoạn văn: CTX\n\nCâu hỏi: Q\nA. A1\nB. A2\nC. A3\nD. A4"
        )
        self.assertEqual(context, "CTX")
        self.assertEqual(question, "Q")
        self.assertEqual(choices, ["A1", "A2", "A3", "A4"])

    def test_canonical_choice_text(self):
        self.assertEqual(canonical_choice_text("  A. Hà_Nội  "), "hà nội")

    def test_make_dedup_hash_is_stable(self):
        h1 = make_dedup_hash("A  B", "C", "D")
        h2 = make_dedup_hash("A B", "C", "D")
        self.assertEqual(h1, h2)

    def test_make_mcq_dedup_hash_is_stable(self):
        h1 = make_mcq_dedup_hash("CTX", "Q", ["A", "B", "C", "D"])
        h2 = make_mcq_dedup_hash("CTX", "Q", ["A", "B", "C", "D"])
        self.assertEqual(h1, h2)

    def test_extract_json_object(self):
        raw = "preface```json\n{\"distractors\":[\"x\",\"y\",\"z\"]}\n```tail"
        parsed = extract_json_object(raw)
        self.assertEqual(parsed["distractors"], ["x", "y", "z"])

    def test_stable_choice_labels(self):
        self.assertEqual(stable_choice_labels("seed-1"), ["C", "A", "D", "B"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: FAIL with import errors or missing helper functions in `scripts/comprehension_mcq_seed_common.py`.

- [ ] **Step 3: Implement the minimal shared helper module**

Create `scripts/comprehension_mcq_seed_common.py` with pure helpers only:

```python
import hashlib
import json
import re
from difflib import SequenceMatcher


ANSWER_LABELS = ["A", "B", "C", "D"]
BANNED_OPTION_TEXTS = {
    "không có thông tin",
    "tất cả các đáp án trên",
    "cả a và b",
}


def normalize_for_hash(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).replace("\u00a0", " ").lower().strip())


def canonical_choice_text(text):
    if text is None:
        return ""
    value = re.sub(r"^\s*[A-Da-d]\s*[\.)\:\-]\s*", "", str(text).replace("\u00a0", " "))
    return re.sub(r"\s+", " ", value).strip().lower()


def build_mcq_user_content(context, question, choices):
    return (
        "Đọc đoạn văn sau và chọn đáp án đúng.\n\n"
        f"Đoạn văn: {context}\n\n"
        f"Câu hỏi: {question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}"
    )


def split_mcq_user_content(content):
    lines = [line.strip() for line in str(content).splitlines() if line.strip()]
    passage_line = next(line for line in lines if line.startswith("Đoạn văn: "))
    question_line = next(line for line in lines if line.startswith("Câu hỏi: "))
    choice_lines = [line for line in lines if re.match(r"^[A-D]\.\s+", line)]
    if len(choice_lines) != 4:
        raise ValueError("expected exactly 4 choice lines")
    context = passage_line[len("Đoạn văn: ") :]
    question = question_line[len("Câu hỏi: ") :]
    choices = [re.sub(r"^[A-D]\.\s+", "", line).strip() for line in choice_lines]
    return context, question, choices


def make_dedup_hash(context, question, answer_text):
    payload = "\n".join([normalize_for_hash(context), normalize_for_hash(question), normalize_for_hash(answer_text)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def make_mcq_dedup_hash(context, question, choices):
    payload = "\n".join(
        [
            normalize_for_hash(context),
            normalize_for_hash(question),
            *[canonical_choice_text(choice) for choice in choices],
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def compute_context_hash(context):
    return hashlib.sha1(normalize_for_hash(context).encode("utf-8")).hexdigest()


def extract_json_object(raw_text):
    text = "" if raw_text is None else str(raw_text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def stable_choice_labels(seed_text):
    order = list(ANSWER_LABELS)
    digest = hashlib.sha1(normalize_for_hash(seed_text).encode("utf-8")).digest()
    for i in range(len(order) - 1, 0, -1):
        j = digest[i % len(digest)] % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def is_near_duplicate_text(left, right):
    a = canonical_choice_text(left)
    b = canonical_choice_text(right)
    if not a or not b:
        return False
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.92
```

- [ ] **Step 4: Re-run the tests**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: PASS for the shared helper tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/comprehension_mcq_seed_common.py scripts/test_comprehension_mcq_seed.py
git commit -m "feat: add shared helpers for comprehension mcq seed"
```

---

### Task 2: UIT-Only Raw Filter

**Files:**
- Create: `scripts/filter_comprehension_raw_uit.py`
- Modify: `scripts/test_comprehension_mcq_seed.py`

- [ ] **Step 1: Add failing tests for UIT-only filtering**

Append tests that construct a small raw JSONL fixture and verify the filter keeps only UIT rows:

```python
import json
import tempfile
from pathlib import Path

from scripts.filter_comprehension_raw_uit import filter_uit_only, main as filter_main


class TestUitOnlyFilter(unittest.TestCase):
    def test_filter_keeps_only_uit_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_path = tmp / "raw.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({
                            "messages": [{"role": "user", "content": "Đoạn văn: A\n\nCâu hỏi: B"}, {"role": "assistant", "content": "X"}],
                            "context": "A",
                            "question": "B",
                            "answer_text": "X",
                            "metadata": {"source_dataset": "taidng/UIT-ViQuAD2.0", "span_check_mode": "strict_exact", "answer_variants": [{"text": "X", "answer_start": 0}], "dedup_hash": "h1"},
                        }),
                        json.dumps({
                            "messages": [{"role": "user", "content": "Đoạn văn: C\n\nCâu hỏi: D"}, {"role": "assistant", "content": "Y"}],
                            "context": "C",
                            "question": "D",
                            "answer_text": "Y",
                            "metadata": {"source_dataset": "ShynBui/Vietnamese_Reading_Comprehension_Dataset", "span_check_mode": "raw_exact_then_normalized_find", "answer_variants": [], "dedup_hash": "h2"},
                        }),
                    ]
                ),
                encoding="utf-8",
            )
            output_path = tmp / "uit.jsonl"
            report_path = tmp / "report.json"
            rejects_path = tmp / "rejects.jsonl"
            kept = filter_uit_only(input_path, output_path, rejects_path, report_path)
            self.assertEqual(kept, 1)
            self.assertTrue(output_path.exists())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: FAIL because `scripts/filter_comprehension_raw_uit.py` does not exist yet.

- [ ] **Step 3: Implement the minimal UIT filter script**

Create `scripts/filter_comprehension_raw_uit.py` with a small CLI and one pure function:

```python
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.comprehension_mcq_seed_common import normalize_for_hash


UIT_DATASET_ID = "taidng/UIT-ViQuAD2.0"


def filter_uit_only(input_path, output_path, rejects_path, report_path):
    kept = []
    rejects = []
    source_counts = Counter()
    span_counts = Counter()
    invalid_records = 0

    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {})
            source_dataset = metadata.get("source_dataset")
            span_check_mode = metadata.get("span_check_mode")
            answer_variants = metadata.get("answer_variants")

            if source_dataset != UIT_DATASET_ID:
                rejects.append({"reason": "non_uit_dataset", "source_id": metadata.get("source_id")})
                continue
            if span_check_mode != "strict_exact":
                rejects.append({"reason": "invalid_span_check_mode", "source_id": metadata.get("source_id")})
                continue
            if not isinstance(answer_variants, list) or not answer_variants:
                rejects.append({"reason": "empty_answer_variants", "source_id": metadata.get("source_id")})
                continue

            source_counts[source_dataset] += 1
            span_counts[span_check_mode] += 1
            kept.append(record)

    output_path = Path(output_path)
    rejects_path = Path(rejects_path)
    report_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with rejects_path.open("w", encoding="utf-8") as handle:
        for reject in rejects:
            handle.write(json.dumps(reject, ensure_ascii=False) + "\n")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump({"total_input": len(kept) + len(rejects), "total_kept": len(kept), "total_rejected": len(rejects), "source_datasets": dict(source_counts), "span_check_modes": dict(span_counts), "invalid_records": invalid_records}, handle, ensure_ascii=False, indent=2)
    return len(kept)
```

- [ ] **Step 4: Re-run the tests**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: PASS for the UIT-only filter test.

- [ ] **Step 5: Commit**

```bash
git add scripts/filter_comprehension_raw_uit.py scripts/test_comprehension_mcq_seed.py
git commit -m "feat: add uit-only filter for comprehension mcq seed"
```

---

### Task 3: Distractor Generation Request Export

**Files:**
- Create: `scripts/prepare_comprehension_mcq_generation.py`
- Modify: `scripts/test_comprehension_mcq_seed.py`

- [ ] **Step 1: Add failing tests for generation request filtering**

Write tests that cover the request filter and emitted schema:

```python
from scripts.comprehension_mcq_seed_common import compute_context_hash
from scripts.prepare_comprehension_mcq_generation import build_generation_request, should_keep_for_generation


class TestGenerationRequests(unittest.TestCase):
    def test_should_keep_for_generation(self):
        self.assertFalse(should_keep_for_generation({"answer_text": "a", "context": "x", "question": "y"}))
        self.assertTrue(should_keep_for_generation({"answer_text": "Hà Nội", "context": "a" * 100, "question": "Thủ đô là gì?"}))

    def test_build_generation_request(self):
        req = build_generation_request({
            "context": "CTX",
            "question": "Q",
            "answer_text": "A",
            "metadata": {"source_id": "sid", "source_split": "train", "title": "T", "dedup_hash": "h"},
        })
        self.assertEqual(req["request_id"], "cmcq-gen-sid")
        self.assertEqual(req["gold_answer_text"], "A")
        self.assertEqual(req["context_hash"], compute_context_hash("CTX"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: FAIL because `scripts/prepare_comprehension_mcq_generation.py` is not implemented yet.

- [ ] **Step 3: Implement the generation request exporter**

Create `scripts/prepare_comprehension_mcq_generation.py` with a deterministic filter and JSONL writer:

```python
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.comprehension_mcq_seed_common import compute_context_hash, make_dedup_hash, normalize_for_hash


MIN_ANSWER_CHARS = 2
MAX_ANSWER_CHARS = 220
MAX_CONTEXT_CHARS = 8000


def should_keep_for_generation(record):
    context = record.get("context", "")
    question = record.get("question", "")
    answer_text = record.get("answer_text", "")
    if not isinstance(context, str) or not context.strip():
        return False
    if not isinstance(question, str) or not question.strip():
        return False
    if not isinstance(answer_text, str) or not answer_text.strip():
        return False
    if len(answer_text) < MIN_ANSWER_CHARS or len(answer_text) > MAX_ANSWER_CHARS:
        return False
    if len(context) > MAX_CONTEXT_CHARS:
        return False
    return True


def build_generation_request(record):
    metadata = record["metadata"]
    return {
        "request_id": f"cmcq-gen-{metadata['source_id']}",
        "source_id": metadata["source_id"],
        "source_split": metadata["source_split"],
        "context": record["context"],
        "question": record["question"],
        "gold_answer_text": record["answer_text"],
        "answer_variants": [variant["text"] for variant in metadata.get("answer_variants", [])],
        "title": metadata.get("title"),
        "raw_dedup_hash": metadata["dedup_hash"],
        "context_hash": compute_context_hash(record["context"]),
        "generation_prompt_version": "comprehension_mcq_distractors_v1",
        "filter_version": "comprehension_mcq_generation_filter_v1",
    }
```

- [ ] **Step 4: Re-run the tests**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: PASS for generation request tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_comprehension_mcq_generation.py scripts/test_comprehension_mcq_seed.py
git commit -m "feat: export comprehension mcq generation requests"
```

---

### Task 4: Parse External Distractor Output and Build MCQ Candidates

**Files:**
- Create: `scripts/build_comprehension_mcq_candidates.py`
- Modify: `scripts/test_comprehension_mcq_seed.py`

- [ ] **Step 1: Add failing tests for JSON parsing and deterministic shuffle**

Add tests that cover fenced JSON parsing and stable answer-label placement:

```python
from scripts.build_comprehension_mcq_candidates import build_candidate_record, parse_generation_output


class TestCandidateBuilder(unittest.TestCase):
    def test_parse_generation_output_from_fenced_json(self):
        payload = parse_generation_output("```json\n{\"distractors\":[\"x\",\"y\",\"z\"]}\n```")
        self.assertEqual(payload["distractors"], ["x", "y", "z"])

    def test_build_candidate_record(self):
        record = build_candidate_record(
            raw_record={
                "context": "CTX",
                "question": "Q",
                "answer_text": "A",
                "metadata": {"source_id": "sid", "source_split": "train", "dedup_hash": "raw-hash", "title": "T"},
            },
            distractors=["B", "C", "D"],
        )
        self.assertEqual(record["metadata"]["task"], "comprehension_mcq")
        self.assertEqual(record["messages"][1]["content"], f"Đáp án: {record['metadata']['answer']}")
        self.assertEqual(sorted(record["metadata"]["choices"].keys()), ["A", "B", "C", "D"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: FAIL because the candidate builder does not exist yet.

- [ ] **Step 3: Implement the candidate builder**

Create `scripts/build_comprehension_mcq_candidates.py` using the shared helpers:

```python
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from scripts.comprehension_mcq_seed_common import (
    ANSWER_LABELS,
    build_mcq_user_content,
    canonical_choice_text,
    compute_context_hash,
    extract_json_object,
    is_near_duplicate_text,
    make_mcq_dedup_hash,
    split_mcq_user_content,
    stable_choice_labels,
)


def parse_generation_output(raw_text):
    parsed = extract_json_object(raw_text)
    distractors = parsed.get("distractors")
    if not isinstance(distractors, list):
        raise ValueError("distractors must be a list")
    if len(distractors) != 3:
        raise ValueError("expected exactly 3 distractors")
    return parsed


def build_candidate_record(raw_record, distractors):
    context = raw_record["context"]
    question = raw_record["question"]
    gold_answer = raw_record["answer_text"]
    labels = stable_choice_labels(raw_record["metadata"]["dedup_hash"])
    choices_in_label_order = [gold_answer, *distractors]
    rng = random.Random(raw_record["metadata"]["dedup_hash"])
    rng.shuffle(choices_in_label_order)
    answer_label = labels[choices_in_label_order.index(gold_answer)]
    choices = {label: value for label, value in zip(labels, choices_in_label_order)}
    return {
        "messages": [
            {"role": "user", "content": build_mcq_user_content(context, question, [choices["A"], choices["B"], choices["C"], choices["D"]])},
            {"role": "assistant", "content": f"Đáp án: {answer_label}"},
        ],
        "metadata": {
            "task": "comprehension_mcq",
            "source": "synthetic",
            "source_dataset": "taidng/UIT-ViQuAD2.0",
            "source_split": raw_record["metadata"]["source_split"],
            "source_id": raw_record["metadata"]["source_id"],
            "title": raw_record["metadata"].get("title"),
            "context_hash": compute_context_hash(context),
            "raw_dedup_hash": raw_record["metadata"]["dedup_hash"],
            "mcq_dedup_hash": make_mcq_dedup_hash(context, question, [choices["A"], choices["B"], choices["C"], choices["D"]]),
            "gold_answer_text": gold_answer,
            "answer": answer_label,
            "choices": choices,
            "generation_method": "llm_distractor_generation_v1",
            "generation_prompt_version": "comprehension_mcq_distractors_v1",
            "qc_version": "comprehension_mcq_uit_rule_qc_v1",
            "language": "vi",
            "difficulty": "medium",
        },
    }
```

- [ ] **Step 4: Re-run the tests**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: PASS for parser and candidate-builder tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_comprehension_mcq_candidates.py scripts/test_comprehension_mcq_seed.py
git commit -m "feat: build comprehension mcq candidates"
```

---

### Task 5: Rule-Based QC and Recheck

**Files:**
- Create: `scripts/qc_comprehension_mcq_seed.py`
- Create: `scripts/recheck_comprehension_mcq_seed.py`
- Modify: `scripts/test_comprehension_mcq_seed.py`

- [ ] **Step 1: Add failing tests for rule QC and recheck validation**

Add tests that validate hard rejects and the read-only validator:

```python
from scripts.qc_comprehension_mcq_seed import rule_check_record
from scripts.recheck_comprehension_mcq_seed import validate_record_schema


class TestRuleQc(unittest.TestCase):
    def test_rule_check_rejects_duplicate_choices(self):
        record = {
            "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "Đáp án: A"}],
            "metadata": {"task": "comprehension_mcq", "source_dataset": "taidng/UIT-ViQuAD2.0", "answer": "A", "choices": {"A": "x", "B": "x", "C": "y", "D": "z"}, "mcq_dedup_hash": "h1"},
        }
        accepted, reason = rule_check_record(record)
        self.assertFalse(accepted)
        self.assertEqual(reason, "duplicate_choices_normalized")

    def test_validate_final_schema(self):
        record = {
            "messages": [{"role": "user", "content": "Đọc đoạn văn sau và chọn đáp án đúng.\n\nĐoạn văn: x\n\nCâu hỏi: y\nA. a\nB. b\nC. c\nD. d"}, {"role": "assistant", "content": "Đáp án: A"}],
            "metadata": {"task": "comprehension_mcq", "source": "synthetic", "source_dataset": "taidng/UIT-ViQuAD2.0", "gold_answer_text": "a", "answer": "A", "choices": {"A": "a", "B": "b", "C": "c", "D": "d"}, "mcq_dedup_hash": "h1"},
        }
        self.assertEqual(validate_record_schema(record), [])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: FAIL because rule QC and recheck logic are not wired yet.

- [ ] **Step 3: Implement rule QC as a transformer**

Create `scripts/qc_comprehension_mcq_seed.py` with a single `rule_check_record` function plus a CLI that reads candidates and writes checked/reject JSONL:

```python
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.comprehension_mcq_seed_common import ANSWER_LABELS, BANNED_OPTION_TEXTS, canonical_choice_text, is_near_duplicate_text


def rule_check_record(record):
    messages = record.get("messages")
    metadata = record.get("metadata", {})
    if not isinstance(messages, list) or len(messages) != 2:
        return False, "invalid_messages"
    if messages[1].get("content") != f"Đáp án: {metadata.get('answer')}":
        return False, "invalid_assistant_answer_format"
    if metadata.get("task") != "comprehension_mcq":
        return False, "invalid_task"
    if metadata.get("source_dataset") != "taidng/UIT-ViQuAD2.0":
        return False, "invalid_source_dataset"

    choices = metadata.get("choices")
    if not isinstance(choices, dict) or sorted(choices.keys()) != ["A", "B", "C", "D"]:
        return False, "missing_choices"

    normalized = [canonical_choice_text(choices[label]) for label in ANSWER_LABELS]
    if any(not value for value in normalized):
        return False, "empty_choice"
    if len(set(normalized)) != 4:
        return False, "duplicate_choices_normalized"
    if any(value in BANNED_OPTION_TEXTS for value in normalized):
        return False, "banned_option_text"

    gold = metadata.get("gold_answer_text", "")
    answer_label = metadata.get("answer")
    if answer_label not in ANSWER_LABELS:
        return False, "invalid_answer_label"
    if canonical_choice_text(choices[answer_label]) != canonical_choice_text(gold):
        return False, "gold_answer_label_mismatch"
    return True, None
```

- [ ] **Step 4: Create the read-only recheck script**

Create `scripts/recheck_comprehension_mcq_seed.py` so it accepts `--input` and only validates/reporting happens there:

```python
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=Path(__file__).resolve().parent.parent / "seed_exports" / "comprehension_mcq_seed_final.jsonl")
    parser.add_argument("--sample", type=int, default=5)
    args = parser.parse_args(argv)
    records = load_records(Path(args.input))
    invalid_records = []
    for rec in records:
        errors = validate_record_schema(rec)
        if errors:
            invalid_records.append(errors)
```

- [ ] **Step 5: Re-run the tests**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: PASS for rule QC and recheck tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/qc_comprehension_mcq_seed.py scripts/recheck_comprehension_mcq_seed.py scripts/test_comprehension_mcq_seed.py
git commit -m "feat: add rule qc for comprehension mcq seed"
```

---

### Task 6: Solver Request Export and Solver QC

**Files:**
- Create: `scripts/prepare_comprehension_mcq_solver.py`
- Create: `scripts/apply_comprehension_mcq_solver_qc.py`
- Modify: `scripts/test_comprehension_mcq_seed.py`

- [ ] **Step 1: Add failing tests for solver request and solver-output parsing**

Write tests that pin the request shape and solver rejection logic:

```python
from scripts.prepare_comprehension_mcq_solver import build_solver_request
from scripts.apply_comprehension_mcq_solver_qc import parse_solver_output, solver_keep_decision


class TestSolverQc(unittest.TestCase):
    def test_build_solver_request(self):
        req = build_solver_request({
            "metadata": {"mcq_dedup_hash": "h1", "source_id": "sid", "answer": "C", "choices": {"A": "a", "B": "b", "C": "c", "D": "d"}},
            "messages": [{"role": "user", "content": "Đọc đoạn văn sau và chọn đáp án đúng.\n\nĐoạn văn: x\n\nCâu hỏi: y\nA. a\nB. b\nC. c\nD. d"}, {"role": "assistant", "content": "Đáp án: C"}],
        })
        self.assertEqual(req["mcq_dedup_hash"], "h1")
        self.assertEqual(req["gold_answer"], "C")

    def test_parse_solver_output(self):
        parsed = parse_solver_output('```json\n{"predicted_answer":"C","is_unambiguous":true,"bad_reason":null}\n```')
        self.assertEqual(parsed["predicted_answer"], "C")

    def test_solver_keep_decision(self):
        record = {"metadata": {"answer": "C"}}
        parsed = {"predicted_answer": "C", "is_unambiguous": True, "bad_reason": None}
        self.assertTrue(solver_keep_decision(record, parsed))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: FAIL because solver request and solver QC scripts do not exist yet.

- [ ] **Step 3: Implement solver request export**

Create `scripts/prepare_comprehension_mcq_solver.py`:

```python
import argparse
import json
from pathlib import Path

from scripts.comprehension_mcq_seed_common import split_mcq_user_content


def build_solver_request(record):
    context, question, _choices = split_mcq_user_content(record["messages"][0]["content"])
    metadata = record["metadata"]
    return {
        "mcq_dedup_hash": metadata["mcq_dedup_hash"],
        "source_id": metadata["source_id"],
        "context": context,
        "question": question,
        "choices": metadata["choices"],
        "gold_answer": metadata["answer"],
        "solver_prompt_version": "comprehension_mcq_solver_v1",
    }
```

- [ ] **Step 4: Implement solver QC**

Create `scripts/apply_comprehension_mcq_solver_qc.py`:

```python
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.comprehension_mcq_seed_common import extract_json_object


def parse_solver_output(raw_text):
    parsed = extract_json_object(raw_text)
    if "predicted_answer" not in parsed:
        raise ValueError("missing predicted_answer")
    return parsed


def solver_keep_decision(record, parsed):
    return (
        parsed.get("predicted_answer") == record["metadata"]["answer"]
        and parsed.get("is_unambiguous") is True
        and not parsed.get("bad_reason")
    )
```

- [ ] **Step 5: Re-run the tests**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: PASS for solver request and solver QC tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_comprehension_mcq_solver.py scripts/apply_comprehension_mcq_solver_qc.py scripts/test_comprehension_mcq_seed.py
git commit -m "feat: add solver boundary for comprehension mcq seed"
```

---

### Task 7: Lexical Leakage Checker

**Files:**
- Create: `scripts/check_comprehension_mcq_leakage.py`
- Modify: `scripts/test_comprehension_mcq_seed.py`

- [ ] **Step 1: Add failing tests for leakage helpers**

Add tests for exact/fuzzy question overlap and same-choice-set detection:

```python
from scripts.check_comprehension_mcq_leakage import (
    has_exact_question_match,
    has_high_context_overlap,
    has_same_choices_pattern,
    is_near_duplicate_question,
)


class TestLeakageCheck(unittest.TestCase):
    def test_exact_question_match(self):
        self.assertTrue(has_exact_question_match("Câu hỏi gì?", "  câu hỏi gì? "))

    def test_near_duplicate_question(self):
        self.assertTrue(is_near_duplicate_question("Ai là tác giả?", "Tác giả là ai?"))

    def test_same_choices_pattern(self):
        self.assertTrue(has_same_choices_pattern({"A": "a", "B": "b", "C": "c", "D": "d"}, {"A": "a", "B": "b", "C": "c", "D": "d"}))

    def test_high_context_overlap(self):
        self.assertTrue(has_high_context_overlap("abc def ghi", "abc def ghi xyz"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: FAIL because leakage helpers are not implemented yet.

- [ ] **Step 3: Implement the lexical leakage checker**

Create `scripts/check_comprehension_mcq_leakage.py` with path-driven benchmark loading:

```python
import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

from scripts.comprehension_mcq_seed_common import canonical_choice_text


def has_exact_question_match(left, right):
    return canonical_choice_text(left) == canonical_choice_text(right)


def is_near_duplicate_question(left, right):
    a = canonical_choice_text(left)
    b = canonical_choice_text(right)
    return a and b and SequenceMatcher(None, a, b).ratio() >= 0.92


def has_same_choices_pattern(left_choices, right_choices):
    left = [canonical_choice_text(left_choices[label]) for label in ["A", "B", "C", "D"]]
    right = [canonical_choice_text(right_choices[label]) for label in ["A", "B", "C", "D"]]
    return left == right


def has_high_context_overlap(left, right):
    a = canonical_choice_text(left)
    b = canonical_choice_text(right)
    return a and b and (a in b or b in a or SequenceMatcher(None, a, b).ratio() >= 0.85)
```

- [ ] **Step 4: Re-run the tests**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: PASS for leakage helper tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/check_comprehension_mcq_leakage.py scripts/test_comprehension_mcq_seed.py
git commit -m "feat: add leakage checks for comprehension mcq seed"
```

---

### Task 8: Finalizer and Recheck Integration

**Files:**
- Create: `scripts/finalize_comprehension_mcq_seed.py`
- Create: `scripts/recheck_comprehension_mcq_seed.py`
- Modify: `scripts/test_comprehension_mcq_seed.py`

- [ ] **Step 1: Add failing tests for the final gate**

Add tests that verify the finalizer keeps only fully valid rows and emits the final report:

```python
from scripts.finalize_comprehension_mcq_seed import finalize_records


class TestFinalizer(unittest.TestCase):
    def test_finalize_records(self):
        records = [{
            "messages": [{"role": "user", "content": "Đọc đoạn văn sau và chọn đáp án đúng.\n\nĐoạn văn: x\n\nCâu hỏi: y\nA. a\nB. b\nC. c\nD. d"}, {"role": "assistant", "content": "Đáp án: A"}],
            "metadata": {"task": "comprehension_mcq", "source": "synthetic", "source_dataset": "taidng/UIT-ViQuAD2.0", "gold_answer_text": "a", "answer": "A", "choices": {"A": "a", "B": "b", "C": "c", "D": "d"}, "mcq_dedup_hash": "h1", "context_hash": "ctx"},
        }]
        final_records, report = finalize_records(records)
        self.assertEqual(len(final_records), 1)
        self.assertEqual(report["total_kept"], 1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: FAIL because the finalizer does not exist yet.

- [ ] **Step 3: Implement the finalizer**

Create `scripts/finalize_comprehension_mcq_seed.py`:

```python
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.recheck_comprehension_mcq_seed import validate_record_schema


def finalize_records(records):
    kept = []
    rejects = []
    for record in records:
        errors = validate_record_schema(record)
        if errors:
            rejects.append({"source_id": record.get("metadata", {}).get("source_id"), "errors": errors})
            continue
        kept.append(record)
    report = {"total_kept": len(kept), "total_rejected": len(rejects)}
    return kept, report
```

- [ ] **Step 4: Implement the read-only recheck script**

Create `scripts/recheck_comprehension_mcq_seed.py` with `validate_record_schema()` as the reusable validator and a CLI that defaults to the final file while accepting `--input`:

```python
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=Path(__file__).resolve().parent.parent / "seed_exports" / "comprehension_mcq_seed_final.jsonl")
    parser.add_argument("--sample", type=int, default=5)
    args = parser.parse_args(argv)
    records = _load_records(Path(args.input))
    invalid_records = []
    for rec in records:
        errors = validate_record_schema(rec)
        if errors:
            invalid_records.append(errors)
```

- [ ] **Step 5: Re-run the tests**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: PASS for finalizer and recheck tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/finalize_comprehension_mcq_seed.py scripts/recheck_comprehension_mcq_seed.py scripts/test_comprehension_mcq_seed.py
git commit -m "feat: finalize comprehension mcq seed"
```

---

### Task 9: Integration Verification and README Command Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a small README block for the new pipeline**

Add a short runnable section that names the new scripts and the offline-boundary files:

```md
# comprehension_mcq_seed_final: UIT-only MCQ pipeline
python scripts/filter_comprehension_raw_uit.py
python scripts/prepare_comprehension_mcq_generation.py
# external generation writes seed_exports/comprehension_mcq_generation_outputs_raw.jsonl
python scripts/build_comprehension_mcq_candidates.py
python scripts/qc_comprehension_mcq_seed.py
python scripts/prepare_comprehension_mcq_solver.py
# external solver writes seed_exports/comprehension_mcq_solver_outputs_raw.jsonl
python scripts/apply_comprehension_mcq_solver_qc.py
python scripts/check_comprehension_mcq_leakage.py
python scripts/finalize_comprehension_mcq_seed.py
python scripts/recheck_comprehension_mcq_seed.py --sample 5
```

- [ ] **Step 2: Run the full unit test suite for the new plan**

Run:

```bash
python -m unittest scripts.test_comprehension_mcq_seed -v
```

Expected: PASS.

- [ ] **Step 3: Run help checks for every script**

Run:

```bash
python scripts/filter_comprehension_raw_uit.py --help
python scripts/prepare_comprehension_mcq_generation.py --help
python scripts/build_comprehension_mcq_candidates.py --help
python scripts/qc_comprehension_mcq_seed.py --help
python scripts/prepare_comprehension_mcq_solver.py --help
python scripts/apply_comprehension_mcq_solver_qc.py --help
python scripts/check_comprehension_mcq_leakage.py --help
python scripts/finalize_comprehension_mcq_seed.py --help
python scripts/recheck_comprehension_mcq_seed.py --help
```

Expected: each command prints the script usage and exits `0`.

- [ ] **Step 4: Run a fixture-only end-to-end smoke test**

Create a tiny temp fixture flow inside `scripts/test_comprehension_mcq_seed.py` or a dedicated integration test:

```python
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # write raw UIT fixture -> filter -> write generation request fixture -> build candidate -> qc -> solver qc -> leakage -> finalize
    # assert each stage writes the expected JSONL/report files and the final count matches the fixture size
```

Expected: a small end-to-end run completes without touching live model providers or benchmark sources.

- [ ] **Step 5: Commit**

```bash
git add README.md scripts/test_comprehension_mcq_seed.py
git commit -m "docs: add comprehension mcq seed workflow"
```
