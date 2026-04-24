import unittest

from scripts.baseline_parsers import (
    MCQ_INVALID_EMPTY,
    MCQ_INVALID_MULTI_LABEL,
    MCQ_INVALID_NO_LABEL,
    MCQ_INVALID_OUT_OF_SET,
    MCQ_OK_FALLBACK,
    MCQ_OK_STRICT,
    is_lambada_prediction_correct,
    is_mcq_prediction_correct,
    normalize_lambada_text,
    normalize_mcq_answer_key,
    parse_mcq_prediction,
)


class TestMCQParser(unittest.TestCase):
    def test_parse_strict_format(self) -> None:
        result = parse_mcq_prediction("Đáp án: C")
        self.assertEqual(result.label, "C")
        self.assertEqual(result.status, MCQ_OK_STRICT)

    def test_parse_fallback_single_label(self) -> None:
        result = parse_mcq_prediction("C.")
        self.assertEqual(result.label, "C")
        self.assertEqual(result.status, MCQ_OK_FALLBACK)

    def test_parse_fallback_inline_label(self) -> None:
        result = parse_mcq_prediction("Em chọn B vì đáp án này phù hợp.")
        self.assertEqual(result.label, "B")
        self.assertEqual(result.status, MCQ_OK_FALLBACK)

    def test_parse_invalid_multi_label(self) -> None:
        result = parse_mcq_prediction("Đáp án: A\nChọn C")
        self.assertIsNone(result.label)
        self.assertEqual(result.status, MCQ_INVALID_MULTI_LABEL)

    def test_parse_invalid_out_of_set(self) -> None:
        result = parse_mcq_prediction("Đáp án: E")
        self.assertIsNone(result.label)
        self.assertEqual(result.status, MCQ_INVALID_OUT_OF_SET)

    def test_parse_invalid_no_label(self) -> None:
        result = parse_mcq_prediction("Mình chưa chắc câu trả lời.")
        self.assertIsNone(result.label)
        self.assertEqual(result.status, MCQ_INVALID_NO_LABEL)

    def test_parse_invalid_empty(self) -> None:
        result = parse_mcq_prediction("   \n  ")
        self.assertIsNone(result.label)
        self.assertEqual(result.status, MCQ_INVALID_EMPTY)

    def test_parse_only_first_three_non_empty_lines(self) -> None:
        output = "Dòng mở đầu\nDòng thứ hai\nDòng thứ ba\nĐáp án: B"
        result = parse_mcq_prediction(output)
        self.assertIsNone(result.label)
        self.assertEqual(result.status, MCQ_INVALID_NO_LABEL)


class TestMCQNormalization(unittest.TestCase):
    def test_normalize_answer_key(self) -> None:
        self.assertEqual(normalize_mcq_answer_key(" B. "), "B")
        self.assertEqual(normalize_mcq_answer_key("đáp án: c"), "C")
        self.assertIsNone(normalize_mcq_answer_key("E"))

    def test_is_mcq_prediction_correct(self) -> None:
        is_correct, parsed, gold = is_mcq_prediction_correct("Đáp án: A", "a")
        self.assertTrue(is_correct)
        self.assertEqual(parsed.label, "A")
        self.assertEqual(gold, "A")


class TestLambadaParser(unittest.TestCase):
    def test_normalize_lambada_text(self) -> None:
        self.assertEqual(normalize_lambada_text(" Nations   League. "), "nations league")

    def test_is_lambada_prediction_correct(self) -> None:
        self.assertTrue(is_lambada_prediction_correct("nations league.", "Nations League"))
        self.assertFalse(is_lambada_prediction_correct("League Nations", "Nations League"))


if __name__ == "__main__":
    unittest.main()
