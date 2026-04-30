import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.load_comprehension_seed_raw import (
    build_user_content,
    build_record,
    build_reject_record,
    apply_length_filters,
    dedup_records,
    extract_valid_answer_variants,
    MISSING_ANSWER,
    MISSING_CONTEXT,
    MISSING_QUESTION,
    MAX_ANSWER_CHARS,
    MAX_CONTEXT_CHARS,
    MAX_QUESTION_CHARS,
    MIN_CONTEXT_CHARS,
    MIN_QUESTION_CHARS,
    QC_VERSION,
    make_dedup_hash,
    _dedup_records_with_rejects,
    run_loader_pipeline,
    process_shynbui_row,
    process_uit_row,
    normalize_for_hash,
    normalize_shynbui_text,
    recompute_answer_start,
    select_primary_variant,
    strict_span_match,
)


class TestComprehensionSeedRawUtilities(unittest.TestCase):
    def test_normalize_for_hash(self) -> None:
        self.assertEqual(normalize_for_hash(None), "")
        self.assertEqual(normalize_for_hash("  Xin\u00a0Chào   Việt Nam  "), "xin chào việt nam")

    def test_normalize_shynbui_text(self) -> None:
        self.assertEqual(normalize_shynbui_text("  Xin\u00a0_bui_  "), "Xin  bui")
        with self.assertRaises(TypeError):
            normalize_shynbui_text(None)
        with self.assertRaises(TypeError):
            normalize_shynbui_text(123)
        with self.assertRaises(ValueError):
            normalize_shynbui_text("   ")

    def test_recompute_answer_start(self) -> None:
        self.assertEqual(recompute_answer_start("Một hai ba", "hai"), 4)
        self.assertEqual(recompute_answer_start("Một hai ba", "bốn"), -1)
        with self.assertRaises(TypeError):
            recompute_answer_start(None, "hai")
        with self.assertRaises(TypeError):
            recompute_answer_start("Một hai ba", None)
        with self.assertRaises(TypeError):
            recompute_answer_start(123, "hai")
        with self.assertRaises(TypeError):
            recompute_answer_start("Một hai ba", 123)

    def test_shynbui_normalize_then_recompute_flow(self) -> None:
        raw_record = {
            "context": "Xin_chao_The_gioi",
            "question": "The_gioi_la_gi?",
            "answer": "The_gioi",
        }
        normalized_context = normalize_shynbui_text(raw_record["context"])
        normalized_question = normalize_shynbui_text(raw_record["question"])
        normalized_answer = normalize_shynbui_text(raw_record["answer"])

        self.assertEqual(normalized_context, "Xin chao The gioi")
        self.assertEqual(normalized_question, "The gioi la gi?")
        self.assertEqual(normalized_answer, "The gioi")
        self.assertEqual(recompute_answer_start(normalized_context, normalized_answer), 9)

    def test_shynbui_malformed_inputs_reject(self) -> None:
        with self.assertRaises(TypeError):
            normalize_shynbui_text(None)
        with self.assertRaises(ValueError):
            normalize_shynbui_text("___")

    def test_shynbui_malformed_flow_rejects(self) -> None:
        raw_record = {
            "context": "Xin_chao_The_gioi",
            "question": "The_gioi_la_gi?",
            "answer": None,
        }
        with self.assertRaises(TypeError):
            normalized_context = normalize_shynbui_text(raw_record["context"])
            normalized_answer = normalize_shynbui_text(raw_record["answer"])
            recompute_answer_start(normalized_context, normalized_answer)

    def test_make_dedup_hash(self) -> None:
        expected_payload = "context one\nquestion one\nanswer one".encode("utf-8")
        expected = hashlib.sha1(expected_payload).hexdigest()
        self.assertEqual(
            make_dedup_hash("Context\u00a0One", "Question One", "Answer One"),
            expected,
        )

    def test_build_user_content(self) -> None:
        self.assertEqual(
            build_user_content("Đây là đoạn văn.", "Câu hỏi là gì?"),
            "Đoạn văn: Đây là đoạn văn.\n\nCâu hỏi: Câu hỏi là gì?",
        )

    def test_strict_span_match(self) -> None:
        self.assertTrue(strict_span_match("Xin chào", "chào", 4))
        self.assertTrue(strict_span_match("A_B", "A_B", 0))
        self.assertFalse(strict_span_match("Xin chào", "chao", 4))
        self.assertFalse(strict_span_match("Xin chào", "Xin chào thế giới", 0))
        with self.assertRaises(ValueError):
            strict_span_match("Xin chào", "Xin", -1)
        with self.assertRaises(TypeError):
            strict_span_match(123, "123", 0)
        with self.assertRaises(TypeError):
            strict_span_match("123", 123, 0)
        with self.assertRaises(TypeError):
            strict_span_match("Xin chào", "chào", "4")

    def test_extract_valid_answer_variants(self) -> None:
        variants = extract_valid_answer_variants(
            "Một hai ba bốn",
            ["hai", "bốn", "không có"],
            [4, 11, 100],
        )

        self.assertEqual(
            variants,
            [
                {"text": "hai", "answer_start": 4},
                {"text": "bốn", "answer_start": 11},
            ],
        )

    def test_extract_valid_answer_variants_no_matches(self) -> None:
        self.assertEqual(
            extract_valid_answer_variants("Một hai ba bốn", ["năm"], [0]),
            [],
        )

    def test_extract_valid_answer_variants_handles_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            extract_valid_answer_variants("Một hai ba bốn", None, None)
        with self.assertRaises(ValueError):
            extract_valid_answer_variants("Một hai ba bốn", "scalar", "scalar")
        with self.assertRaises(ValueError):
            extract_valid_answer_variants("Một hai ba bốn", ["hai", "bốn"], [4])
        with self.assertRaises(ValueError):
            extract_valid_answer_variants("123 456", [123, "456"], [0, 4])
        with self.assertRaises(ValueError):
            extract_valid_answer_variants("Một hai ba bốn", ["hai"], [4.0])
        with self.assertRaises(ValueError):
            extract_valid_answer_variants("Một hai ba bốn", ["hai"], [True])

    def test_select_primary_variant(self) -> None:
        variants = [
            {"text": "hai", "answer_start": 4},
            {"text": "bốn", "answer_start": 11},
        ]
        self.assertEqual(select_primary_variant(variants), variants[0])
        self.assertIsNone(select_primary_variant([]))

    def test_length_filter_constants(self) -> None:
        self.assertEqual(MIN_CONTEXT_CHARS, 80)
        self.assertEqual(MAX_CONTEXT_CHARS, 8000)
        self.assertEqual(MIN_QUESTION_CHARS, 5)
        self.assertEqual(MAX_QUESTION_CHARS, 400)
        self.assertEqual(MAX_ANSWER_CHARS, 300)
        self.assertEqual(MISSING_CONTEXT, "missing_context")
        self.assertEqual(MISSING_QUESTION, "missing_question")
        self.assertEqual(MISSING_ANSWER, "missing_answer")

    def test_apply_length_filters(self) -> None:
        self.assertEqual(apply_length_filters(None, "hello", "answer"), "missing_context")
        self.assertEqual(apply_length_filters(123, "hello", "answer"), "missing_context")
        self.assertEqual(apply_length_filters("a" * 80, None, "answer"), "missing_question")
        self.assertEqual(apply_length_filters("a" * 80, 123, "answer"), "missing_question")
        self.assertEqual(apply_length_filters("a" * 80, "hello", 123), "missing_answer")
        self.assertEqual(apply_length_filters("a" * 79, "hello", "answer"), "context_too_short")
        self.assertEqual(apply_length_filters("a" * 8001, "hello", "answer"), "context_too_long")
        self.assertEqual(apply_length_filters("a" * 80, "abcd", "answer"), "question_too_short")
        self.assertEqual(apply_length_filters("a" * 80, "a" * 401, "answer"), "question_too_long")
        self.assertEqual(apply_length_filters("a" * 80, "hello", None), "missing_answer")
        self.assertEqual(apply_length_filters("a" * 80, "hello", "a" * 301), "answer_too_long")
        self.assertIsNone(apply_length_filters("a" * 80, "hello", "a" * 300))

    def test_build_record(self) -> None:
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

    def test_build_record_defaults_original_answer_start(self) -> None:
        record = build_record(
            context="CTX",
            question="Q",
            answer_text="A",
            answer_start=5,
            source_dataset="ds",
            source_split="train",
            source_id="id",
            title=None,
            answer_variants=[],
            span_check_mode="raw_exact_then_normalized_find",
        )
        self.assertEqual(record["metadata"]["original_answer_start"], 5)

    def test_build_reject_record(self) -> None:
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

    def test_build_reject_record_stringifies_context_preview(self) -> None:
        reject = build_reject_record(
            source_dataset="ds",
            source_split="train",
            source_id="id",
            reason="span_mismatch",
            context=12345,
            question="Q",
            answer="A",
            answer_start=1,
        )
        self.assertEqual(reject["context_preview"], "12345")

    def test_dedup_records_keeps_first_hash_occurrence(self) -> None:
        records = [
            {"metadata": {"dedup_hash": "h1"}, "id": 1},
            {"metadata": {"dedup_hash": "h2"}, "id": 2},
            {"metadata": {"dedup_hash": "h1"}, "id": 3},
            {"metadata": {"dedup_hash": "h3"}, "id": 4},
            {"metadata": {"dedup_hash": "h2"}, "id": 5},
        ]
        original = list(records)

        deduped, duplicates = dedup_records(records)

        self.assertEqual(deduped, [records[0], records[1], records[3]])
        self.assertEqual(duplicates, [records[2], records[4]])
        self.assertEqual(records, original)

    def test_dedup_records_handles_empty_input(self) -> None:
        deduped, duplicates = dedup_records([])

        self.assertEqual(deduped, [])
        self.assertEqual(duplicates, [])

    def test_dedup_records_rejects_malformed_records(self) -> None:
        with self.assertRaises(ValueError):
            dedup_records([{"id": 1}])
        with self.assertRaises(ValueError):
            dedup_records([{"metadata": {"source_id": "id1"}, "id": 1}])
        with self.assertRaises(ValueError):
            dedup_records([{"metadata": {"source_id": "id2", "dedup_hash": None}, "id": 2}])
        with self.assertRaises(ValueError):
            dedup_records([{"metadata": {"source_id": "id3", "dedup_hash": []}, "id": 3}])
        with self.assertRaises(ValueError):
            dedup_records([{"metadata": {"source_id": "id4", "dedup_hash": ""}, "id": 4}])


class TestComprehensionSeedRawRowProcessors(unittest.TestCase):
    def test_process_uit_row_keeps_first_valid_variant(self) -> None:
        row = {
            "id": "uit-1",
            "title": "Bai doc",
            "context": "Ha Noi la thu do. " + "noi dung " * 12,
            "question": "Thu do cua Viet Nam la gi?",
            "is_impossible": False,
            "answers": {
                "text": ["Ha Noi", "thu do", "khong hop le"],
                "answer_start": [0, 10, 999],
            },
        }

        record, reason = process_uit_row(row, "train")

        self.assertIsNone(reason)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["context"], row["context"])
        self.assertEqual(record["question"], row["question"])
        self.assertEqual(record["answer_text"], "Ha Noi")
        self.assertEqual(record["metadata"]["answer_variants"], [
            {"text": "Ha Noi", "answer_start": 0},
            {"text": "thu do", "answer_start": 10},
        ])
        self.assertEqual(record["metadata"]["span_check_mode"], "strict_exact")
        self.assertEqual(record["metadata"]["original_answer_start"], 0)

    def test_process_uit_row_skips_invalid_leading_variant(self) -> None:
        row = {
            "id": "uit-1b",
            "title": "Bai doc",
            "context": "Ha Noi la thu do. " + "noi dung " * 12,
            "question": "Thu do cua Viet Nam la gi?",
            "is_impossible": False,
            "answers": {
                "text": ["", "Ha Noi"],
                "answer_start": [0, 0],
            },
        }

        record, reason = process_uit_row(row, "train")

        self.assertIsNone(reason)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["answer_text"], "Ha Noi")
        self.assertEqual(record["metadata"]["answer_variants"], [{"text": "Ha Noi", "answer_start": 0}])

    def test_process_uit_row_skips_length_invalid_leading_variant(self) -> None:
        long_answer = "a" * 301
        row = {
            "id": "uit-1c",
            "title": "Bai doc",
            "context": long_answer + " Ha Noi " + "noi dung " * 10,
            "question": "Thu do cua Viet Nam la gi?",
            "is_impossible": False,
            "answers": {
                "text": [long_answer, "Ha Noi"],
                "answer_start": [0, len(long_answer) + 1],
            },
        }

        record, reason = process_uit_row(row, "train")

        self.assertIsNone(reason)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["answer_text"], "Ha Noi")

    def test_process_uit_row_rejects_impossible_and_bad_schema(self) -> None:
        impossible_row = {
            "id": "uit-2",
            "context": "Ha Noi la thu do. " + "noi dung " * 12,
            "question": "Thu do cua Viet Nam la gi?",
            "is_impossible": True,
            "answers": {"text": ["Ha Noi"], "answer_start": [0]},
        }
        record, reason = process_uit_row(impossible_row, "train")
        self.assertIsNone(record)
        self.assertEqual(reason, "is_impossible")

        bad_schema_row = {
            "id": "uit-3",
            "context": "Ha Noi la thu do. " + "noi dung " * 12,
            "question": "Thu do cua Viet Nam la gi?",
            "is_impossible": False,
            "answers": {"text": "Ha Noi", "answer_start": 0},
        }
        record, reason = process_uit_row(bad_schema_row, "train")
        self.assertIsNone(record)
        self.assertEqual(reason, "schema_error")

    def test_process_shynbui_row_normalizes_and_recomputes_start(self) -> None:
        row = {
            "id": "shyn-1",
            "context": "Ha_Noi la thu_do. " + "noi_dung " * 12,
            "question": "Thu_do_cua_Viet_Nam_la_gi?",
            "answer": "thu_do",
            "answer_start": 10,
        }

        record, reason = process_shynbui_row(row, "train")

        self.assertIsNone(reason)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["context"], ("Ha Noi la thu do. " + "noi dung " * 12).strip())
        self.assertEqual(record["question"], "Thu do cua Viet Nam la gi?")
        self.assertEqual(record["answer_text"], "thu do")
        self.assertEqual(record["metadata"]["answer_start"], 10)
        self.assertEqual(record["metadata"]["original_answer_start"], 10)
        self.assertEqual(record["metadata"]["span_check_mode"], "raw_exact_then_normalized_find")

    def test_process_shynbui_row_preserves_validated_offset_with_duplicates(self) -> None:
        row = {
            "id": "shyn-1b",
            "context": "abc_def " * 10 + "abc_def",
            "question": "Cau hoi?",
            "answer": "abc_def",
            "answer_start": 80,
        }

        record, reason = process_shynbui_row(row, "train")

        self.assertIsNone(reason)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["context"], ("abc def " * 10 + "abc def"))
        self.assertEqual(record["metadata"]["answer_start"], 80)
        self.assertEqual(record["metadata"]["original_answer_start"], 80)

    def test_process_shynbui_row_rejects_bad_rows(self) -> None:
        row = {
            "id": "shyn-2",
            "context": "Ha_Noi la thu_do. " + "noi_dung " * 12,
            "question": "Thu_do_cua_Viet_Nam_la_gi?",
            "answer": None,
            "answer_start": 10,
        }

        record, reason = process_shynbui_row(row, "train")

        self.assertIsNone(record)
        self.assertEqual(reason, "missing_answer")


class TestComprehensionSeedRawPipelineBehavior(unittest.TestCase):
    def test_pipeline_dedups_and_reports_duplicates(self) -> None:
        source_datasets = {
            "taidng/UIT-ViQuAD2.0": {
                "train": [
                    {
                        "id": "uit-a",
                        "title": "Bai doc",
                        "context": "Ha Noi la thu do. " + "noi dung " * 12,
                        "question": "Thu do cua Viet Nam la gi?",
                        "is_impossible": False,
                        "answers": {"text": ["Ha Noi"], "answer_start": [0]},
                    }
                ]
            },
            "ShynBui/Vietnamese_Reading_Comprehension_Dataset": {
                "train": [
                    {
                        "id": "shyn-a",
                        "context": "Ha Noi la thu do. " + "noi dung " * 12,
                        "question": "Thu do cua Viet Nam la gi?",
                        "answer": "Ha Noi",
                        "answer_start": 0,
                    }
                ]
            },
        }

        outputs = run_loader_pipeline(source_datasets)

        self.assertEqual(outputs["report"]["total_loaded"], 2)
        self.assertEqual(outputs["report"]["total_kept"], 1)
        self.assertEqual(outputs["report"]["total_rejected"], 1)
        self.assertEqual(outputs["report"]["duplicate_count"], 1)
        self.assertEqual(outputs["report"]["reject_reasons"].get("duplicate"), 1)
        self.assertEqual(outputs["report"]["kept_by_dataset"], {"taidng/UIT-ViQuAD2.0": 1})
        self.assertEqual(outputs["report"]["kept_by_split"], {"train": 1})
        self.assertEqual(len(outputs["rejects"]), 1)
        self.assertEqual(outputs["rejects"][0]["reason"], "duplicate")

    def test_dedup_fallback_rejects_malformed_hash(self) -> None:
        record = {
            "context": "Ha Noi la thu do. " + "noi dung " * 12,
            "question": "Thu do cua Viet Nam la gi?",
            "answer_text": "Ha Noi",
            "metadata": {
                "source_dataset": "taidng/UIT-ViQuAD2.0",
                "source_split": "train",
                "source_id": "uit-a",
                "answer_start": 0,
                "dedup_hash": None,
            },
        }

        deduped, duplicates, rejects = _dedup_records_with_rejects([record])

        self.assertEqual(deduped, [])
        self.assertEqual(duplicates, [])
        self.assertEqual(len(rejects), 1)
        self.assertEqual(rejects[0]["reason"], "schema_error")


class TestComprehensionSeedRawRecheckUtilities(unittest.TestCase):
    def test_format_user_content(self) -> None:
        from scripts.recheck_comprehension_seed_raw import format_user_content

        self.assertEqual(
            format_user_content("Đây là đoạn văn.", "Câu hỏi là gì?"),
            "Đoạn văn: Đây là đoạn văn.\n\nCâu hỏi: Câu hỏi là gì?",
        )

    def test_validate_record_schema_reports_expected_errors(self) -> None:
        from scripts.recheck_comprehension_seed_raw import validate_record_schema

        record = {
            "messages": [
                {"role": "user", "content": "Đoạn văn: abc\n\nCâu hỏi: q"},
                {"role": "assistant", "content": "wrong"},
            ],
            "context": "abc answer",
            "question": "q",
            "answer_text": "answer",
            "metadata": {
                "task": "comprehension_raw",
                "source": "public",
                "source_dataset": "ds",
                "source_split": "train",
                "source_id": "id-1",
                "answer_start": 4,
                "original_answer_start": 4,
                "answer_text": "answer",
                "answer_variants": [],
                "title": None,
                "language": "vi",
                "difficulty": "medium",
                "span_check_mode": "strict_exact",
                "dedup_hash": "hash",
                "qc_version": "v1",
            },
        }

        errors = validate_record_schema(record)

        self.assertIn("assistant content does not match answer_text", errors)

    def test_validate_record_schema_requires_non_empty_fields(self) -> None:
        from scripts.recheck_comprehension_seed_raw import format_user_content, validate_record_schema

        record = {
            "messages": [
                {"role": "user", "content": format_user_content("", "")},
                {"role": "assistant", "content": ""},
            ],
            "context": "",
            "question": "",
            "answer_text": "",
            "metadata": {
                "task": "comprehension_raw",
                "source": "public",
                "source_dataset": "ds",
                "source_split": "train",
                "source_id": "id-2",
                "answer_start": 0,
                "original_answer_start": 0,
                "answer_text": "",
                "answer_variants": [],
                "title": None,
                "language": "vi",
                "difficulty": "medium",
                "span_check_mode": "strict_exact",
                "dedup_hash": "hash-2",
                "qc_version": "v1",
            },
        }

        errors = validate_record_schema(record)

        self.assertIn("context is not a non-empty string", errors)
        self.assertIn("question is not a non-empty string", errors)
        self.assertIn("answer_text is not a non-empty string", errors)

    def test_validate_record_schema_requires_matching_metadata_answer_text(self) -> None:
        from scripts.recheck_comprehension_seed_raw import format_user_content, validate_record_schema

        record = {
            "messages": [
                {"role": "user", "content": format_user_content("abc answer", "q")},
                {"role": "assistant", "content": "answer"},
            ],
            "context": "abc answer",
            "question": "q",
            "answer_text": "answer",
            "metadata": {
                "task": "comprehension_raw",
                "source": "public",
                "source_dataset": "ds",
                "source_split": "train",
                "source_id": "id-3",
                "answer_start": 4,
                "original_answer_start": 4,
                "answer_text": "different",
                "answer_variants": [],
                "title": None,
                "language": "vi",
                "difficulty": "medium",
                "span_check_mode": "strict_exact",
                "dedup_hash": "hash-3",
                "qc_version": "v1",
            },
        }

        errors = validate_record_schema(record)

        self.assertIn("metadata.answer_text does not match top-level answer_text", errors)

    def test_validate_record_schema_requires_non_empty_dedup_hash(self) -> None:
        from scripts.recheck_comprehension_seed_raw import format_user_content, validate_record_schema

        record = {
            "messages": [
                {"role": "user", "content": format_user_content("abc answer", "q")},
                {"role": "assistant", "content": "answer"},
            ],
            "context": "abc answer",
            "question": "q",
            "answer_text": "answer",
            "metadata": {
                "task": "comprehension_raw",
                "source": "public",
                "source_dataset": "ds",
                "source_split": "train",
                "source_id": "id-4",
                "answer_start": 4,
                "original_answer_start": 4,
                "answer_text": "answer",
                "answer_variants": [],
                "title": None,
                "language": "vi",
                "difficulty": "medium",
                "span_check_mode": "strict_exact",
                "dedup_hash": None,
                "qc_version": "v1",
            },
        }

        errors = validate_record_schema(record)

        self.assertIn("metadata.dedup_hash is not a non-empty string", errors)

    def test_main_reports_duplicate_dedup_hash(self) -> None:
        from scripts import recheck_comprehension_seed_raw as recheck

        records = [
            {
                "messages": [
                    {"role": "user", "content": recheck.format_user_content("abc answer", "q")},
                    {"role": "assistant", "content": "answer"},
                ],
                "context": "abc answer",
                "question": "q",
                "answer_text": "answer",
                "metadata": {
                    "task": "comprehension_raw",
                    "source": "public",
                    "source_dataset": "ds",
                    "source_split": "train",
                    "source_id": "id-4",
                    "answer_start": 4,
                    "original_answer_start": 4,
                    "answer_text": "answer",
                    "answer_variants": [],
                    "title": None,
                    "language": "vi",
                    "difficulty": "medium",
                    "span_check_mode": "strict_exact",
                    "dedup_hash": "hash-dup",
                    "qc_version": "v1",
                },
            },
            {
                "messages": [
                    {"role": "user", "content": recheck.format_user_content("def answer", "q")},
                    {"role": "assistant", "content": "answer"},
                ],
                "context": "def answer",
                "question": "q",
                "answer_text": "answer",
                "metadata": {
                    "task": "comprehension_raw",
                    "source": "public",
                    "source_dataset": "ds",
                    "source_split": "train",
                    "source_id": "id-5",
                    "answer_start": 4,
                    "original_answer_start": 4,
                    "answer_text": "answer",
                    "answer_variants": [],
                    "title": None,
                    "language": "vi",
                    "difficulty": "medium",
                    "span_check_mode": "strict_exact",
                    "dedup_hash": "hash-dup",
                    "qc_version": "v1",
                },
            },
        ]

        fake_path = SimpleNamespace(exists=lambda: True)

        with patch.object(recheck, "JSONL_PATH", fake_path), patch.object(recheck, "_load_records", return_value=records):
            self.assertEqual(recheck.main([]), 1)

    def test_main_handles_invalid_sample_records_without_crashing(self) -> None:
        from scripts import recheck_comprehension_seed_raw as recheck

        records = [
            {
                "messages": [{"role": "user", "content": "Đoạn văn: abc\n\nCâu hỏi: q"}],
                "context": "abc answer",
                "question": "q",
                "answer_text": "answer",
                "metadata": {
                    "task": "comprehension_raw",
                    "source": "public",
                    "source_dataset": "ds",
                    "source_split": "train",
                    "source_id": "id-6",
                    "answer_start": 4,
                    "original_answer_start": 4,
                    "answer_text": "answer",
                    "answer_variants": [],
                    "title": None,
                    "language": "vi",
                    "difficulty": "medium",
                    "span_check_mode": "strict_exact",
                    "dedup_hash": "hash-6",
                    "qc_version": "v1",
                },
            }
        ]

        fake_path = SimpleNamespace(exists=lambda: True)

        with patch.object(recheck, "JSONL_PATH", fake_path), patch.object(recheck, "_load_records", return_value=records):
            self.assertEqual(recheck.main(["--sample", "1"]), 1)

    def test_main_returns_non_zero_when_input_missing(self) -> None:
        from scripts import recheck_comprehension_seed_raw as recheck

        with patch.object(recheck, "JSONL_PATH", Path("/tmp/does-not-exist.jsonl")):
            self.assertEqual(recheck.main([]), 1)


if __name__ == "__main__":
    unittest.main()
