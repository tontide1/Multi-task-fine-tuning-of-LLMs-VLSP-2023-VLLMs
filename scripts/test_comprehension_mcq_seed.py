import hashlib
import json
import subprocess
import tempfile
import unittest
import sys
from pathlib import Path

from scripts.comprehension_mcq_seed_common import (
    build_mcq_user_content,
    canonical_choice_text,
    compute_context_hash,
    extract_json_object,
    is_near_duplicate_text,
    make_dedup_hash,
    make_mcq_dedup_hash,
    normalize_for_hash,
    split_mcq_user_content,
    stable_choice_labels,
)
from scripts.build_comprehension_mcq_candidates import build_candidate_record, parse_generation_output
from scripts.qc_comprehension_mcq_seed import rule_check_record
from scripts.prepare_comprehension_mcq_generation import (
    build_generation_request,
    filter_generation_requests,
    should_keep_for_generation,
)
from scripts.apply_comprehension_mcq_solver_qc import parse_solver_output, solver_keep_decision
from scripts import finalize_comprehension_mcq_seed
from scripts.finalize_comprehension_mcq_seed import finalize_records
from scripts.prepare_comprehension_mcq_solver import build_solver_request
from scripts import recheck_comprehension_mcq_seed
from scripts.recheck_comprehension_mcq_seed import validate_record_schema


class TestComprehensionMcqSeedCommon(unittest.TestCase):
    def test_normalize_for_hash(self) -> None:
        self.assertEqual(normalize_for_hash(None), "")
        self.assertEqual(normalize_for_hash("  Xin\u00a0Chào   Việt Nam  "), "xin chào việt nam")

    def test_canonical_choice_text(self) -> None:
        self.assertEqual(canonical_choice_text("  A. Hà_Nội  "), "hà nội")
        self.assertEqual(canonical_choice_text("B)   Đà Nẵng"), "đà nẵng")
        self.assertEqual(canonical_choice_text(None), "")

    def test_build_mcq_user_content_exact_format(self) -> None:
        content = build_mcq_user_content("Đây là đoạn văn.", "Câu hỏi là gì?", ["A1", "B2", "C3", "D4"])
        self.assertEqual(
            content,
            "Đọc đoạn văn sau và chọn đáp án đúng.\n\n"
            "Đoạn văn: Đây là đoạn văn.\n\n"
            "Câu hỏi: Câu hỏi là gì?\n"
            "A. A1\n"
            "B. B2\n"
            "C. C3\n"
            "D. D4",
        )

    def test_split_mcq_user_content_roundtrip(self) -> None:
        content = build_mcq_user_content("CTX", "Q?", ["A1", "B2", "C3", "D4"])
        context, question, choices = split_mcq_user_content(content)
        self.assertEqual(context, "CTX")
        self.assertEqual(question, "Q?")
        self.assertEqual(choices, ["A1", "B2", "C3", "D4"])

    def test_split_mcq_user_content_roundtrip_multiline(self) -> None:
        content = build_mcq_user_content("Dòng 1\nDòng 2", "Câu hỏi?\nChi tiết?", ["A1", "B2", "C3", "D4"])
        context, question, choices = split_mcq_user_content(content)
        self.assertEqual(context, "Dòng 1\nDòng 2")
        self.assertEqual(question, "Câu hỏi?\nChi tiết?")
        self.assertEqual(choices, ["A1", "B2", "C3", "D4"])

    def test_make_dedup_hash(self) -> None:
        expected = hashlib.sha1("ctx\nq\na".encode("utf-8")).hexdigest()
        self.assertEqual(make_dedup_hash("CTX", "Q", "A"), expected)

    def test_make_mcq_dedup_hash_is_stable(self) -> None:
        choices = ["A", "B", "C", "D"]
        expected = hashlib.sha1("ctx\nq\na\nb\nc\nd".encode("utf-8")).hexdigest()
        self.assertEqual(make_mcq_dedup_hash("CTX", "Q", choices), expected)
        self.assertEqual(make_mcq_dedup_hash("CTX", "Q", choices), expected)

    def test_compute_context_hash(self) -> None:
        expected = hashlib.sha1("ctx".encode("utf-8")).hexdigest()
        self.assertEqual(compute_context_hash("CTX"), expected)

    def test_extract_json_object(self) -> None:
        self.assertEqual(extract_json_object('{"a": 1, "b": [2, 3]}'), {"a": 1, "b": [2, 3]})
        self.assertEqual(extract_json_object("```json\n{\"x\": 1}\n```"), {"x": 1})

    def test_stable_choice_labels_is_deterministic_permutation(self) -> None:
        labels1 = stable_choice_labels("seed-1")
        labels2 = stable_choice_labels("seed-1")
        labels3 = stable_choice_labels("seed-2")

        self.assertEqual(labels1, labels2)
        self.assertCountEqual(labels1, ["A", "B", "C", "D"])
        self.assertCountEqual(labels3, ["A", "B", "C", "D"])
        self.assertNotEqual(labels1, labels3)

    def test_is_near_duplicate_text(self) -> None:
        self.assertTrue(is_near_duplicate_text("  A. Hà Nội  ", "hà nội"))
        self.assertTrue(is_near_duplicate_text("Xin chào", "xin chào"))
        self.assertFalse(is_near_duplicate_text("Hà Nội", "TP Hồ Chí Minh"))

    def test_extract_json_object_rejects_invalid_json(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object("not json")
        with self.assertRaises(json.JSONDecodeError):
            extract_json_object("{bad json}")


class TestComprehensionMcqLeakageChecker(unittest.TestCase):
    def test_exact_question_match(self) -> None:
        from scripts.check_comprehension_mcq_leakage import has_exact_question_match

        self.assertTrue(has_exact_question_match("Câu hỏi gì?", "  câu hỏi gì? "))

    def test_near_duplicate_question(self) -> None:
        from scripts.check_comprehension_mcq_leakage import is_near_duplicate_question

        self.assertTrue(is_near_duplicate_question("Ai là tác giả?", "Tác giả là ai?"))

    def test_same_choices_pattern(self) -> None:
        from scripts.check_comprehension_mcq_leakage import has_same_choices_pattern

        left_choices = {"A": "a", "B": "b", "C": "c", "D": "d"}
        right_choices = {"A": "a", "B": "b", "C": "c", "D": "d"}

        self.assertTrue(has_same_choices_pattern(left_choices, right_choices))

    def test_high_context_overlap(self) -> None:
        from scripts.check_comprehension_mcq_leakage import has_high_context_overlap

        self.assertTrue(has_high_context_overlap("abc def ghi", "abc def ghi xyz"))

    def test_cli_runs_on_jsonl_file_and_directory_benchmark(self) -> None:
        from scripts.check_comprehension_mcq_leakage import main

        record = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Đọc đoạn văn sau và chọn đáp án đúng.\n\n"
                        "Đoạn văn: Cùng một đoạn văn mẫu.\n\n"
                        "Câu hỏi: Ai là tác giả?\n"
                        "A. A1\n"
                        "B. B2\n"
                        "C. C3\n"
                        "D. D4"
                    ),
                },
                {"role": "assistant", "content": "Đáp án: B"},
            ],
            "metadata": {
                "source_id": "candidate-1",
                "answer": "B",
                "choices": {"A": "A1", "B": "B2", "C": "C3", "D": "D4"},
            },
        }
        benchmark_record = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Đọc đoạn văn sau và chọn đáp án đúng.\n\n"
                        "Đoạn văn: Cùng một đoạn văn mẫu khác.\n\n"
                        "Câu hỏi: Tác giả là ai?\n"
                        "A. A1\n"
                        "B. B2\n"
                        "C. C3\n"
                        "D. D4"
                    ),
                },
                {"role": "assistant", "content": "Đáp án: B"},
            ],
            "metadata": {
                "source_id": "benchmark-1",
                "answer": "B",
                "choices": {"A": "A1", "B": "B2", "C": "C3", "D": "D4"},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.jsonl"
            benchmark_dir = tmp_path / "benchmark"
            benchmark_file = benchmark_dir / "part-1.jsonl"
            output_path = tmp_path / "output.jsonl"
            rejects_path = tmp_path / "rejects.jsonl"
            report_path = tmp_path / "report.json"
            benchmark_dir.mkdir()

            input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            benchmark_file.write_text(json.dumps(benchmark_record, ensure_ascii=False) + "\n", encoding="utf-8")

            self.assertEqual(
                main(
                    [
                        "--input",
                        str(input_path),
                        "--benchmark",
                        str(benchmark_dir),
                        "--output-jsonl",
                        str(output_path),
                        "--rejects-jsonl",
                        str(rejects_path),
                        "--report-json",
                        str(report_path),
                    ]
                ),
                0,
            )

            kept_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            reject_rows = [
                json.loads(line)
                for line in rejects_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(kept_rows, [])
            self.assertEqual(len(reject_rows), 1)
            self.assertEqual(reject_rows[0]["reason"], "near_duplicate_question")
            self.assertEqual(report["benchmark_jsonl_inputs"], [str(benchmark_file)])
            self.assertEqual(report["total_loaded"], 1)
            self.assertEqual(report["total_rejected"], 1)

    def test_same_answer_fact_pattern(self) -> None:
        from scripts.check_comprehension_mcq_leakage import check_leakage

        candidate = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Đọc đoạn văn sau và chọn đáp án đúng.\n\n"
                        "Đoạn văn: Câu chuyện hoàn toàn khác.\n\n"
                        "Câu hỏi: Câu nào đúng?\n"
                        "A. Alpha\n"
                        "B. Beta\n"
                        "C. Gamma\n"
                        "D. Delta"
                    ),
                },
                {"role": "assistant", "content": "Đáp án: C"},
            ],
            "metadata": {"source_id": "candidate-2", "answer": "C"},
        }
        benchmark = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Đọc đoạn văn sau và chọn đáp án đúng.\n\n"
                        "Đoạn văn: Một dữ kiện khác.\n\n"
                        "Câu hỏi: Câu hỏi riêng biệt?\n"
                        "A. Một\n"
                        "B. Hai\n"
                        "C. Ba\n"
                        "D. Bốn"
                    ),
                },
                {"role": "assistant", "content": "Đáp án: C"},
            ],
            "metadata": {
                "source_id": "benchmark-2",
                "answer": "C",
                "gold_answer_text": "Ba",
            },
        }

        kept, rejects, report = check_leakage([candidate], [benchmark])

        self.assertEqual(kept, [])
        self.assertEqual(len(rejects), 1)
        self.assertEqual(rejects[0]["reason"], "same_answer_fact_pattern")
        self.assertEqual(report["reject_reasons"], {"same_answer_fact_pattern": 1})


class TestComprehensionRawUitFilter(unittest.TestCase):
    def test_filter_uit_only_and_main_write_matching_report(self) -> None:
        from scripts import filter_comprehension_raw_uit as uit_filter

        keep_record = {
            "context": "Một đoạn văn đủ điều kiện để giữ lại.",
            "question": "Câu hỏi nào được giữ?",
            "answer_text": "được giữ",
            "metadata": {
                "source_dataset": "taidng/UIT-ViQuAD2.0",
                "span_check_mode": "strict_exact",
                "answer_variants": [{"text": "được giữ", "answer_start": 24}],
                "source_id": "keep-1",
                "source_split": "train",
            },
        }
        reject_dataset_record = {
            "context": "Một đoạn văn khác.",
            "question": "Câu hỏi bị loại?",
            "answer_text": "bị loại",
            "metadata": {
                "source_dataset": "other/dataset",
                "span_check_mode": "strict_exact",
                "answer_variants": [{"text": "bị loại", "answer_start": 15}],
                "source_id": "reject-1",
                "source_split": "train",
            },
        }
        reject_span_mode_record = {
            "context": "Một đoạn văn khác nữa.",
            "question": "Câu hỏi bị loại do span?",
            "answer_text": "bị loại",
            "metadata": {
                "source_dataset": "taidng/UIT-ViQuAD2.0",
                "span_check_mode": "loose",
                "answer_variants": [{"text": "bị loại", "answer_start": 18}],
                "source_id": "reject-2",
                "source_split": "validation",
            },
        }
        reject_variants_record = {
            "context": "Một đoạn văn khác nữa.",
            "question": "Câu hỏi bị loại do variants?",
            "answer_text": "bị loại",
            "metadata": {
                "source_dataset": "taidng/UIT-ViQuAD2.0",
                "span_check_mode": "strict_exact",
                "answer_variants": [],
                "source_id": "reject-3",
                "source_split": "test",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "comprehension_seed_raw.jsonl"
            output_path = tmp_path / "comprehension_seed_raw_uit_only.jsonl"
            rejects_path = tmp_path / "comprehension_seed_raw_uit_only_rejects.jsonl"
            report_path = tmp_path / "comprehension_seed_raw_uit_only_report.json"

            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(keep_record, ensure_ascii=False),
                        json.dumps(reject_dataset_record, ensure_ascii=False),
                        json.dumps(reject_span_mode_record, ensure_ascii=False),
                        json.dumps(reject_variants_record, ensure_ascii=False),
                        "{not json}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            kept_records, reject_rows, report = uit_filter.filter_uit_only(
                [
                    keep_record,
                    reject_dataset_record,
                    reject_span_mode_record,
                    reject_variants_record,
                ]
            )

            self.assertEqual(kept_records, [keep_record])
            self.assertEqual(len(reject_rows), 3)
            self.assertEqual(report["total_loaded"], 4)
            self.assertEqual(report["total_kept"], 1)
            self.assertEqual(report["total_rejected"], 3)
            self.assertNotIn("kept_by_dataset", report)
            self.assertEqual(report["kept_by_split"], {"train": 1})
            self.assertEqual(report["source_datasets"], {"taidng/UIT-ViQuAD2.0": 1})
            self.assertEqual(report["span_check_modes"], {"strict_exact": 1})
            self.assertEqual(report["empty_answer_variants"], 1)
            self.assertEqual(report["invalid_records"], 0)
            self.assertEqual(report["reject_reasons"]["source_dataset"], 1)
            self.assertEqual(report["reject_reasons"]["span_check_mode"], 1)
            self.assertEqual(report["reject_reasons"]["answer_variants_empty"], 1)

            self.assertEqual(
                uit_filter.main(
                    [
                        "--input-jsonl",
                        str(input_path),
                        "--output-jsonl",
                        str(output_path),
                        "--rejects-jsonl",
                        str(rejects_path),
                        "--report-json",
                        str(report_path),
                    ]
                ),
                0,
            )

            kept_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            reject_rows = [
                json.loads(line)
                for line in rejects_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(kept_rows, [keep_record])
            self.assertEqual(len(reject_rows), 4)
            self.assertEqual(report["total_loaded"], 5)
            self.assertEqual(report["total_kept"], 1)
            self.assertEqual(report["total_rejected"], 4)
            self.assertEqual(report["kept_by_split"], {"train": 1})
            self.assertEqual(report["source_datasets"], {"taidng/UIT-ViQuAD2.0": 1})
            self.assertEqual(report["span_check_modes"], {"strict_exact": 1})
            self.assertEqual(report["empty_answer_variants"], 1)
            self.assertEqual(report["invalid_records"], 1)
            self.assertEqual(report["reject_reasons"]["source_dataset"], 1)
            self.assertEqual(report["reject_reasons"]["span_check_mode"], 1)
            self.assertEqual(report["reject_reasons"]["answer_variants_empty"], 1)
            self.assertEqual(report["reject_reasons"]["invalid_json"], 1)
            self.assertEqual(report["output_jsonl"], str(output_path))
            self.assertEqual(report["rejects_jsonl"], str(rejects_path))
            self.assertEqual(report["report_json"], str(report_path))


class TestComprehensionMcqCandidateBuilder(unittest.TestCase):
    def test_parse_generation_output_accepts_raw_and_fenced_json(self) -> None:
        raw_payload = parse_generation_output('{"distractors": ["x", "y", "z"]}')
        fenced_payload = parse_generation_output("```json\n{\"distractors\": [\"x\", \"y\", \"z\"]}\n```")

        self.assertEqual(raw_payload["distractors"], ["x", "y", "z"])
        self.assertEqual(fenced_payload["distractors"], ["x", "y", "z"])

    def test_parse_generation_output_rejects_wrong_distractor_count(self) -> None:
        with self.assertRaises(ValueError):
            parse_generation_output('{"distractors": ["x", "y"]}')

    def test_build_candidate_record_exports_expected_schema(self) -> None:
        raw_record = {
            "context": "CTX",
            "question": "Q",
            "answer_text": "A",
            "metadata": {
                "source_id": "sid",
                "source_split": "train",
                "dedup_hash": "raw-hash",
                "title": "T",
            },
        }

        record = build_candidate_record(raw_record, ["B", "C", "D"])

        self.assertEqual(len(record["messages"]), 2)
        self.assertEqual(record["messages"][0]["role"], "user")
        self.assertEqual(record["messages"][1]["role"], "assistant")
        self.assertEqual(record["messages"][1]["content"], f"Đáp án: {record['metadata']['answer']}")
        self.assertEqual(record["metadata"]["task"], "comprehension_mcq")
        self.assertEqual(record["metadata"]["source"], "synthetic")
        self.assertEqual(record["metadata"]["source_dataset"], "taidng/UIT-ViQuAD2.0")
        self.assertEqual(record["metadata"]["source_split"], "train")
        self.assertEqual(record["metadata"]["source_id"], "sid")
        self.assertEqual(record["metadata"]["title"], "T")
        self.assertEqual(record["metadata"]["context_hash"], compute_context_hash("CTX"))
        self.assertEqual(record["metadata"]["raw_dedup_hash"], "raw-hash")
        self.assertEqual(record["metadata"]["gold_answer_text"], "A")
        self.assertEqual(record["metadata"]["generation_method"], "llm_distractor_generation_v1")
        self.assertEqual(record["metadata"]["generation_prompt_version"], "comprehension_mcq_distractors_v1")
        self.assertEqual(record["metadata"]["qc_version"], "comprehension_mcq_uit_rule_qc_v1")
        self.assertEqual(record["metadata"]["language"], "vi")
        self.assertEqual(record["metadata"]["difficulty"], "medium")
        self.assertEqual(sorted(record["metadata"]["choices"].keys()), ["A", "B", "C", "D"])
        self.assertEqual(
            record["messages"][0]["content"],
            build_mcq_user_content(
                "CTX",
                "Q",
                [
                    record["metadata"]["choices"]["A"],
                    record["metadata"]["choices"]["B"],
                    record["metadata"]["choices"]["C"],
                    record["metadata"]["choices"]["D"],
                ],
            ),
        )
        self.assertEqual(
            record["metadata"]["mcq_dedup_hash"],
            make_mcq_dedup_hash(
                "CTX",
                "Q",
                [
                    record["metadata"]["choices"]["A"],
                    record["metadata"]["choices"]["B"],
                    record["metadata"]["choices"]["C"],
                    record["metadata"]["choices"]["D"],
                ],
            ),
        )


class TestComprehensionMcqGenerationRequestExport(unittest.TestCase):
    def test_should_keep_for_generation_applies_deterministic_bounds(self) -> None:
        self.assertFalse(should_keep_for_generation({"context": "", "question": "Q", "answer_text": "AB"}))
        self.assertFalse(should_keep_for_generation({"context": "CTX", "question": None, "answer_text": "AB"}))
        self.assertFalse(should_keep_for_generation({"context": "CTX", "question": "Q", "answer_text": "A"}))
        self.assertFalse(should_keep_for_generation({"context": "CTX", "question": "Q", "answer_text": "A" * 221}))
        self.assertFalse(should_keep_for_generation({"context": "A" * 8001, "question": "Q", "answer_text": "AB"}))
        self.assertFalse(should_keep_for_generation({"context": 123, "question": "Q", "answer_text": "AB"}))
        self.assertTrue(
            should_keep_for_generation(
                {"context": "A" * 8000, "question": "Câu hỏi hợp lệ?", "answer_text": "AB"}
            )
        )
        self.assertTrue(
            should_keep_for_generation(
                {"context": "CTX", "question": "Câu hỏi hợp lệ?", "answer_text": "A" * 220}
            )
        )

    def test_build_generation_request_exports_expected_schema(self) -> None:
        record = {
            "context": "CTX",
            "question": "Q",
            "answer_text": "Đáp án",
            "metadata": {
                "source_id": "sid",
                "source_split": "train",
                "title": "T",
                "dedup_hash": "raw-hash",
                "answer_variants": [{"text": "Đáp án", "answer_start": 0}, {"text": "Phương án", "answer_start": 10}],
            },
        }

        request = build_generation_request(record)

        self.assertEqual(request["request_id"], "cmcq-gen-sid")
        self.assertEqual(request["source_id"], "sid")
        self.assertEqual(request["source_split"], "train")
        self.assertEqual(request["context"], "CTX")
        self.assertEqual(request["question"], "Q")
        self.assertEqual(request["gold_answer_text"], "Đáp án")
        self.assertEqual(request["answer_variants"], ["Đáp án", "Phương án"])
        self.assertEqual(request["title"], "T")
        self.assertEqual(request["raw_dedup_hash"], "raw-hash")
        self.assertEqual(request["context_hash"], compute_context_hash("CTX"))
        self.assertEqual(request["generation_prompt_version"], "comprehension_mcq_distractors_v1")
        self.assertEqual(request["filter_version"], "comprehension_mcq_generation_filter_v1")

    def test_filter_generation_requests_reports_short_question_and_generic_answer(self) -> None:
        kept_record = {
            "context": "CTX đủ dài để vượt qua bộ lọc.",
            "question": "Câu hỏi nào hợp lệ?",
            "answer_text": "Đáp án",
            "metadata": {"source_id": "keep", "source_split": "train", "dedup_hash": "h1"},
        }
        short_question_record = {
            "context": "CTX đủ dài để vượt qua bộ lọc.",
            "question": "Ai?",
            "answer_text": "Đáp án",
            "metadata": {"source_id": "short-q", "source_split": "train", "dedup_hash": "h2"},
        }
        generic_answer_record = {
            "context": "CTX đủ dài để vượt qua bộ lọc.",
            "question": "Câu hỏi nào hợp lệ?",
            "answer_text": "Không có thông tin",
            "metadata": {"source_id": "generic-a", "source_split": "train", "dedup_hash": "h3"},
        }

        kept, rejects, report = filter_generation_requests(
            [kept_record, short_question_record, generic_answer_record]
        )

        self.assertEqual(kept, [kept_record])
        self.assertEqual(len(rejects), 2)
        self.assertEqual(report["reject_reasons"]["question_too_short"], 1)
        self.assertEqual(report["reject_reasons"]["generic_answer_text"], 1)

    def test_prepare_generation_script_help_runs_directly(self) -> None:
        script_path = Path(__file__).resolve().parent / "prepare_comprehension_mcq_generation.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout)


class TestComprehensionMcqRuleQc(unittest.TestCase):
    def _base_record(self) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Đọc đoạn văn sau và chọn đáp án đúng.\n\n"
                        "Đoạn văn: CTX\n\n"
                        "Câu hỏi: Q\n"
                        "A. a\n"
                        "B. b\n"
                        "C. c\n"
                        "D. d"
                    ),
                },
                {"role": "assistant", "content": "Đáp án: A"},
            ],
            "metadata": {
                "task": "comprehension_mcq",
                "source": "synthetic",
                "source_dataset": "taidng/UIT-ViQuAD2.0",
                "source_split": "train",
                "source_id": "sid",
                "gold_answer_text": "a",
                "answer": "A",
                "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "mcq_dedup_hash": "hash-1",
                "context_hash": "ctx-hash",
            },
        }

    def test_rule_check_record_rejects_expected_hard_cases(self) -> None:
        base_hash = self._base_record()["metadata"]["mcq_dedup_hash"]
        cases = [
            ("invalid_messages", {"messages": [], "metadata": self._base_record()["metadata"]}),
            (
                "invalid_assistant_answer_format",
                {
                    **self._base_record(),
                    "messages": [
                        self._base_record()["messages"][0],
                        {"role": "assistant", "content": "A"},
                    ],
                },
            ),
            (
                "invalid_answer_label",
                {
                    **self._base_record(),
                    "metadata": {**self._base_record()["metadata"], "answer": "E"},
                },
            ),
            (
                "missing_choices",
                {
                    **self._base_record(),
                    "metadata": {k: v for k, v in self._base_record()["metadata"].items() if k != "choices"},
                },
            ),
            (
                "duplicate_choices_normalized",
                {
                    **self._base_record(),
                    "metadata": {
                        **self._base_record()["metadata"],
                        "choices": {"A": "a", "B": "A. a", "C": "c", "D": "d"},
                    },
                },
            ),
            (
                "gold_answer_missing_from_choices",
                {
                    **self._base_record(),
                    "metadata": {
                        **self._base_record()["metadata"],
                        "choices": {"A": "x", "B": "b", "C": "c", "D": "d"},
                    },
                },
            ),
            (
                "gold_answer_label_mismatch",
                {
                    **self._base_record(),
                    "metadata": {
                        **self._base_record()["metadata"],
                        "choices": {"A": "x", "B": "a", "C": "c", "D": "d"},
                    },
                },
            ),
            (
                "distractor_matches_gold",
                {
                    **self._base_record(),
                    "metadata": {
                        **self._base_record()["metadata"],
                        "gold_answer_text": "hà nội đẹp",
                        "choices": {
                            "A": "hà nội đẹp",
                            "B": "đẹp hà nội",
                            "C": "đà nẵng",
                            "D": "huế",
                        },
                    },
                },
            ),
            (
                "distractor_contains_gold",
                {
                    **self._base_record(),
                    "metadata": {
                        **self._base_record()["metadata"],
                        "gold_answer_text": "hà nội",
                        "choices": {
                            "A": "hà nội",
                            "B": "thành phố hà nội cổ",
                            "C": "đà nẵng",
                            "D": "huế",
                        },
                    },
                },
            ),
            (
                "banned_option_text",
                {
                    **self._base_record(),
                    "metadata": {
                        **self._base_record()["metadata"],
                        "choices": {"A": "a", "B": "b", "C": "Không có thông tin", "D": "d"},
                    },
                },
            ),
            (
                "missing_context_or_question",
                {
                    **self._base_record(),
                    "messages": [
                        {
                            "role": "user",
                            "content": "Đọc đoạn văn sau và chọn đáp án đúng.\n\nĐoạn văn: \n\nCâu hỏi: \nA. a\nB. b\nC. c\nD. d",
                        },
                        {"role": "assistant", "content": "Đáp án: A"},
                    ],
                },
            ),
            (
                "invalid_task",
                {
                    **self._base_record(),
                    "metadata": {**self._base_record()["metadata"], "task": "other"},
                },
            ),
            (
                "invalid_source_dataset",
                {
                    **self._base_record(),
                    "metadata": {**self._base_record()["metadata"], "source_dataset": "other/dataset"},
                },
            ),
            (
                "choice_too_long",
                {
                    **self._base_record(),
                    "metadata": {
                        **self._base_record()["metadata"],
                        "choices": {
                            "A": "a",
                            "B": "b",
                            "C": "c",
                            "D": "d" * 301,
                        },
                    },
                },
            ),
            (
                "choice_length_imbalance",
                {
                    **self._base_record(),
                    "metadata": {
                        **self._base_record()["metadata"],
                        "choices": {
                            "A": "a",
                            "B": "b",
                            "C": "x" * 60,
                            "D": "c",
                        },
                    },
                },
            ),
            (
                "duplicate_mcq_dedup_hash",
                {
                    **self._base_record(),
                    "metadata": {**self._base_record()["metadata"], "mcq_dedup_hash": base_hash},
                },
                {"seen_mcq_dedup_hashes": {base_hash}},
            ),
        ]

        for case in cases:
            if len(case) == 2:
                expected_reason, record = case
                kwargs = {}
            else:
                expected_reason, record, kwargs = case
            with self.subTest(reason=expected_reason):
                accepted, reason = rule_check_record(record, **kwargs)
                self.assertFalse(accepted)
                self.assertEqual(reason, expected_reason)


class TestComprehensionMcqRecheck(unittest.TestCase):
    def _valid_record(self) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Đọc đoạn văn sau và chọn đáp án đúng.\n\n"
                        "Đoạn văn: CTX\n\n"
                        "Câu hỏi: Q\n"
                        "A. a\n"
                        "B. b\n"
                        "C. c\n"
                        "D. d"
                    ),
                },
                {"role": "assistant", "content": "Đáp án: A"},
            ],
            "metadata": {
                "task": "comprehension_mcq",
                "source": "synthetic",
                "source_dataset": "taidng/UIT-ViQuAD2.0",
                "source_split": "train",
                "source_id": "sid",
                "gold_answer_text": "a",
                "answer": "A",
                "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "mcq_dedup_hash": "hash-1",
                "context_hash": "ctx-hash",
            },
        }

    def test_validate_record_schema_accepts_valid_final_style_record(self) -> None:
        self.assertEqual(validate_record_schema(self._valid_record()), [])

    def test_validate_record_schema_reports_schema_errors(self) -> None:
        record = {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "Đáp án: A"},
            ],
            "metadata": {"task": "comprehension_mcq"},
        }

        errors = validate_record_schema(record)

        self.assertIn("empty_user_content", errors)
        self.assertIn("missing_metadata_source", errors)
        self.assertIn("missing_metadata_source_dataset", errors)
        self.assertIn("missing_metadata_source_split", errors)
        self.assertIn("missing_metadata_source_id", errors)
        self.assertIn("missing_metadata_gold_answer_text", errors)
        self.assertIn("missing_metadata_answer", errors)
        self.assertIn("missing_metadata_choices", errors)
        self.assertIn("missing_metadata_mcq_dedup_hash", errors)

    def test_main_runs_on_valid_final_style_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "final.jsonl"
            input_path.write_text(json.dumps(self._valid_record(), ensure_ascii=False) + "\n", encoding="utf-8")

            result = recheck_comprehension_mcq_seed.main(["--input", str(input_path), "--sample", "1"])

            self.assertEqual(result, 0)


class TestComprehensionMcqFinalizer(unittest.TestCase):
    def _valid_record(self, source_id: str = "sid", dedup_hash: str = "hash-1") -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Đọc đoạn văn sau và chọn đáp án đúng.\n\n"
                        "Đoạn văn: CTX\n\n"
                        "Câu hỏi: Q\n"
                        "A. a\n"
                        "B. b\n"
                        "C. c\n"
                        "D. d"
                    ),
                },
                {"role": "assistant", "content": "Đáp án: A"},
            ],
            "metadata": {
                "task": "comprehension_mcq",
                "source": "synthetic",
                "source_dataset": "taidng/UIT-ViQuAD2.0",
                "source_split": "train",
                "source_id": source_id,
                "gold_answer_text": "a",
                "answer": "A",
                "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "mcq_dedup_hash": dedup_hash,
                "context_hash": "ctx-hash",
            },
        }

    def test_finalize_records_keeps_valid_rows_and_builds_report(self) -> None:
        records = [self._valid_record(source_id="sid-1", dedup_hash="hash-1"), self._valid_record(source_id="sid-2", dedup_hash="hash-2")]

        final_records, report, samples = finalize_records(records, sample_size=1)

        self.assertEqual(len(final_records), 2)
        self.assertEqual(report["total_loaded"], 2)
        self.assertEqual(report["total_kept"], 2)
        self.assertEqual(report["total_rejected"], 0)
        self.assertEqual(report["duplicate_mcq_dedup_hashes"], [])
        self.assertEqual(report["output_jsonl"], str(finalize_comprehension_mcq_seed.DEFAULT_OUTPUT_JSONL))
        self.assertEqual(report["report_json"], str(finalize_comprehension_mcq_seed.DEFAULT_REPORT_JSON))
        self.assertEqual(report["samples_jsonl"], str(finalize_comprehension_mcq_seed.DEFAULT_SAMPLES_JSONL))
        self.assertEqual(samples, [records[0]])

    def test_finalize_records_rejects_invalid_schema(self) -> None:
        record = self._valid_record()
        record["metadata"]["choices"] = {"A": "a", "B": "b", "C": "b", "D": "d"}

        with self.assertRaises(ValueError) as ctx:
            finalize_records([record])

        self.assertIn("duplicate_choices_normalized", str(ctx.exception))

    def test_finalize_records_rejects_duplicate_dedup_hash(self) -> None:
        records = [self._valid_record(source_id="sid-1", dedup_hash="hash-1"), self._valid_record(source_id="sid-2", dedup_hash="hash-1")]

        with self.assertRaises(ValueError) as ctx:
            finalize_records(records)

        self.assertIn("duplicate mcq_dedup_hash", str(ctx.exception))

    def test_main_writes_final_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "no_leak.jsonl"
            output_path = tmp_path / "final.jsonl"
            report_path = tmp_path / "final_report.json"
            samples_path = tmp_path / "final_samples.jsonl"
            input_path.write_text(json.dumps(self._valid_record(), ensure_ascii=False) + "\n", encoding="utf-8")

            result = finalize_comprehension_mcq_seed.main(
                [
                    "--input",
                    str(input_path),
                    "--output-jsonl",
                    str(output_path),
                    "--report-json",
                    str(report_path),
                    "--samples-jsonl",
                    str(samples_path),
                    "--sample-size",
                    "1",
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(
                [
                    json.loads(line)
                    for line in output_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ],
                [self._valid_record()],
            )
            self.assertEqual(
                [
                    json.loads(line)
                    for line in samples_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ],
                [self._valid_record()],
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["input_jsonl"], str(input_path))
            self.assertEqual(report["output_jsonl"], str(output_path))
            self.assertEqual(report["report_json"], str(report_path))
            self.assertEqual(report["samples_jsonl"], str(samples_path))


class TestComprehensionMcqSolverBoundary(unittest.TestCase):
    def _rule_checked_record(self) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Đọc đoạn văn sau và chọn đáp án đúng.\n\n"
                        "Đoạn văn: CTX\n\n"
                        "Câu hỏi: Q\n"
                        "A. a\n"
                        "B. b\n"
                        "C. c\n"
                        "D. d"
                    ),
                },
                {"role": "assistant", "content": "Đáp án: A"},
            ],
            "metadata": {
                "task": "comprehension_mcq",
                "source": "synthetic",
                "source_dataset": "taidng/UIT-ViQuAD2.0",
                "source_split": "train",
                "source_id": "sid",
                "gold_answer_text": "a",
                "answer": "A",
                "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "mcq_dedup_hash": "hash-1",
                "context_hash": "ctx-hash",
            },
        }

    def test_build_solver_request_exports_expected_schema(self) -> None:
        request = build_solver_request(self._rule_checked_record())

        self.assertEqual(request["mcq_dedup_hash"], "hash-1")
        self.assertEqual(request["source_id"], "sid")
        self.assertEqual(request["gold_answer"], "A")
        self.assertEqual(request["solver_prompt_version"], "comprehension_mcq_solver_v1")
        self.assertEqual(request["choices"], {"A": "a", "B": "b", "C": "c", "D": "d"})
        self.assertEqual(request["context"], "CTX")
        self.assertEqual(request["question"], "Q")

    def test_parse_solver_output_accepts_raw_and_fenced_json(self) -> None:
        raw = parse_solver_output('{"predicted_answer":"A","is_unambiguous":true,"bad_reason":null}')
        fenced = parse_solver_output("```json\n{\"predicted_answer\":\"A\",\"is_unambiguous\":true,\"bad_reason\":null}\n```")

        self.assertEqual(raw["predicted_answer"], "A")
        self.assertEqual(fenced["predicted_answer"], "A")

    def test_solver_keep_decision_requires_match_and_unambiguous(self) -> None:
        record = self._rule_checked_record()
        self.assertTrue(
            solver_keep_decision(
                record,
                {"predicted_answer": "A", "is_unambiguous": True, "bad_reason": None},
            )
        )
        self.assertFalse(
            solver_keep_decision(
                record,
                {"predicted_answer": "B", "is_unambiguous": True, "bad_reason": None},
            )
        )


if __name__ == "__main__":
    unittest.main()
