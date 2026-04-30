import hashlib
import json
import unittest

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


if __name__ == "__main__":
    unittest.main()
