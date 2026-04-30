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
from scripts.prepare_comprehension_mcq_generation import (
    build_generation_request,
    should_keep_for_generation,
)


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
                {"context": "A" * 8000, "question": "Q", "answer_text": "AB"}
            )
        )
        self.assertTrue(
            should_keep_for_generation(
                {"context": "CTX", "question": "Q", "answer_text": "A" * 220}
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


if __name__ == "__main__":
    unittest.main()
