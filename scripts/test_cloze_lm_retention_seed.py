
import unittest
import re
from typing import Tuple, Optional

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    # Basic normalization
    text = text.strip()
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_cloze_pair(text: str, min_words: int = 20, max_words: int = 150) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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
    # isalpha() in Python 3 handles Unicode characters.
    if not target_word.isalpha():
         return None, None, "invalid_target_word"

    # Context is everything but the last word
    context = " ".join(words[:-1])
    
    user_prompt = f"Điền từ tiếp theo vào chỗ trống:\n\n{context}..."
    assistant_response = target_word
    
    return user_prompt, assistant_response, None

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
