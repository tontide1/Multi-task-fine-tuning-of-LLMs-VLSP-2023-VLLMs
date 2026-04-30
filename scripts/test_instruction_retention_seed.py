import unittest
import re
from scripts.load_instruction_retention_seed import (
    normalize_text,
    clean_markers,
    is_mcq_like,
    extract_first_user_assistant_pair,
    make_dedup_hash,
)

class TestInstructionRetentionSeed(unittest.TestCase):
    def test_normalize_text(self):
        # 1. normalize giữ newline nhưng trim whitespace dư
        raw = "  Xin chào \n  Bạn   \n "
        self.assertEqual(normalize_text(raw), "Xin chào\nBạn")
        
        # Unicode normalization
        raw_unicode = "Xin cha\u0300o" # o + `
        normalized = normalize_text(raw_unicode)
        self.assertEqual(normalized, "Xin chào")

    def test_clean_markers(self):
        # 2. marker đầu dòng được clean
        self.assertEqual(clean_markers("### Human: Hello"), "Hello")
        self.assertEqual(clean_markers("Instruction: Do X"), "Do X")
        self.assertEqual(clean_markers("  Response :   Ok"), "Ok")
        
        # 3. marker trong nội dung không bị xóa nhầm
        content = "Hãy nói về ### Human Rights."
        self.assertEqual(clean_markers(content), content)

    def test_is_mcq_like(self):
        # 4. MCQ full block bị reject
        mcq_block = "Câu hỏi: Ai là người đầu tiên?\nA. A\nB. B\nC. C\nD. D"
        self.assertTrue(is_mcq_like(mcq_block, "Đáp án: A"))
        
        # 6. assistant "Đáp án: A" bị reject
        self.assertTrue(is_mcq_like("Câu hỏi gì đó?", "Đáp án: A"))
        
        # 5. text có chữ "Đáp án:" nhưng không phải MCQ không bị reject nhầm
        self.assertFalse(is_mcq_like("Bạn hãy giải thích đáp án này.", "Đây là lời giải thích dài."))

    def test_extract_first_user_assistant_pair(self):
        # 7. extract first user-assistant pair hoạt động khi có system message trước
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am fine."}
        ]
        u, a = extract_first_user_assistant_pair(messages)
        self.assertEqual(u, "Hello")
        self.assertEqual(a, "Hi there!")

    def test_make_dedup_hash(self):
        # 8. dedup hash giống nhau sau normalize whitespace
        h1 = make_dedup_hash(" Hello ", " World ")
        h2 = make_dedup_hash("Hello", "World")
        self.assertEqual(h1, h2)

if __name__ == "__main__":
    unittest.main()
