
import unittest
from typing import Tuple, Optional
from scripts.load_cloze_lm_retention_seed import (
    normalize_text, 
    extract_cloze_pair, 
    MIN_WORDS, 
    MAX_WORDS
)

class TestClozeLMRetentionSeed(unittest.TestCase):
    def test_basic_extraction(self):
        text = "Hôm nay tôi đi học ở trường đại học bách khoa hà nội."
        # Word count is 12.
        prompt, target, reason = extract_cloze_pair(text, min_words=5)
        self.assertIsNone(reason)
        self.assertEqual(target, "nội")
        self.assertTrue(prompt.startswith("Điền từ tiếp theo vào chỗ trống:"))
        self.assertIn("hà...", prompt)

    def test_too_short(self):
        text = "Một hai ba."
        prompt, target, reason = extract_cloze_pair(text, min_words=5)
        self.assertEqual(reason, "too_short_3")

    def test_invalid_target(self):
        text = "Giá của nó là 50000."
        prompt, target, reason = extract_cloze_pair(text, min_words=3)
        self.assertEqual(reason, "invalid_target_word")

    def test_punctuation_cleaning(self):
        text = "Tôi thích ăn cơm chiên hải sản!"
        prompt, target, reason = extract_cloze_pair(text, min_words=3)
        self.assertIsNone(reason)
        self.assertEqual(target, "sản")
        self.assertIn("hải...", prompt)

    def test_garbage_text(self):
        # Text with lots of numbers/symbols
        text = "--- 123456 !!! $$$ %%% ^^^ &&& *** (())"
        prompt, target, reason = extract_cloze_pair(text, min_words=3)
        self.assertEqual(reason, "invalid_target_word")

if __name__ == "__main__":
    unittest.main()
