import unittest
import re
from scripts.load_instruction_retention_seed import (
    normalize_text,
    clean_markers,
    is_mcq_like,
    extract_first_user_assistant_pair,
    make_dedup_hash,
    process_vi_alpaca_row,
    MIN_CONTENT_CHARS,
    MAX_USER_CHARS,
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
        
        # Case: None and whitespace only
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text("   \n   "), "")

    def test_clean_markers(self):
        # 2. marker đầu dòng được clean
        self.assertEqual(clean_markers("### Human: Hello"), "Hello")
        self.assertEqual(clean_markers("Instruction: Do X"), "Do X")
        self.assertEqual(clean_markers("  Response :   Ok"), "Ok")
        
        # Case insensitive and multi-line
        self.assertEqual(clean_markers("instruction: Line 1\nLine 2"), "Line 1\nLine 2")
        
        # 3. marker trong nội dung không bị xóa nhầm
        content = "Hãy nói về ### Human Rights."
        self.assertEqual(clean_markers(content), content)

    def test_is_mcq_like(self):
        # 4. MCQ full block bị reject
        mcq_block = "Câu hỏi: Ai là người đầu tiên?\nA. A\nB. B\nC. C\nD. D"
        self.assertTrue(is_mcq_like(mcq_block, "Đáp án: A"))
        
        # 6. assistant "Đáp án: A" bị reject
        self.assertTrue(is_mcq_like("Câu hỏi gì đó?", "Đáp án: A"))
        self.assertTrue(is_mcq_like("Câu hỏi gì đó?", "Answer: B"))
        
        # 5. text có chữ "Đáp án:" nhưng không phải MCQ không bị reject nhầm
        self.assertFalse(is_mcq_like("Bạn hãy giải thích đáp án này.", "Đây là lời giải thích dài."))
        
        # Edge case: User has options but Assistant provides long explanation
        self.assertFalse(is_mcq_like("Chọn A, B, C hoặc D:\nA. 1\nB. 2\nC. 3\nD. 4", "Việc chọn lựa phụ thuộc vào nhiều yếu tố khách quan và chủ quan..."))

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
        
        # Test with list (some environments might pass list instead of ndarray)
        u_list, a_list = extract_first_user_assistant_pair(messages)
        self.assertEqual(u_list, "Hello")

    def test_make_dedup_hash(self):
        # 8. dedup hash giống nhau sau normalize whitespace
        h1 = make_dedup_hash(" Hello ", " World ")
        h2 = make_dedup_hash("Hello", "World")
        self.assertEqual(h1, h2)

    def test_process_vi_alpaca_row(self):
        # Normal row
        row = {"instruction": "Xin chào bạn nhé", "input": "", "output": "Tôi có thể giúp gì cho bạn?", "_row_index": 0}
        record, reason = process_vi_alpaca_row(row, "train")
        self.assertIsNotNone(record)
        self.assertIsNone(reason)
        
        # Too short
        row_short = {"instruction": "A", "input": "", "output": "B", "_row_index": 1}
        record, reason = process_vi_alpaca_row(row_short, "train")
        self.assertIsNone(record)
        self.assertEqual(reason, "user_too_short")
        
        # Too long
        row_long = {"instruction": "X" * (MAX_USER_CHARS + 1), "input": "", "output": "Ok", "_row_index": 2}
        record, reason = process_vi_alpaca_row(row_long, "train")
        self.assertIsNone(record)
        self.assertEqual(reason, "user_too_long")
        
        # MCQ contamination
        row_mcq = {"instruction": "A. 1\nB. 2\nC. 3\nD. 4", "input": "", "output": "Đáp án: A", "_row_index": 3}
        # Note: "Đáp án: A" is exactly 9 chars. MIN_CONTENT_CHARS is 10.
        # Let's use a slightly longer assistant to trigger MCQ filter without triggering length filter
        row_mcq["output"] = "Kết quả: A" # "Kết quả: A" is 10 chars
        record, reason = process_vi_alpaca_row(row_mcq, "train")
        self.assertIsNone(record)
        self.assertEqual(reason, "mcq_like_contamination")

if __name__ == "__main__":
    unittest.main()
