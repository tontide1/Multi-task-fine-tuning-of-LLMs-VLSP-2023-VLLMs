"""Unit tests for demo/model_utils.py."""

import unittest
from unittest.mock import MagicMock, patch

from demo.model_utils import predict_mcq


class TestPredictMCQ(unittest.TestCase):
    """Tests for predict_mcq."""

    def _make_mocks(self):
        tokenizer = MagicMock()
        model = MagicMock()
        return tokenizer, model

    @patch("demo.model_utils._generate")
    def test_prompt_contains_question_and_choices(self, mock_generate):
        """predict_mcq should build a prompt containing the question and all choices."""
        mock_generate.return_value = "A"
        tokenizer, model = self._make_mocks()
        question = "What is the capital of France?"
        choices = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Madrid"}

        predict_mcq(tokenizer, model, question, choices)

        mock_generate.assert_called_once()
        prompt = mock_generate.call_args[0][2]  # (tokenizer, model, prompt, max_new_tokens)
        self.assertIn(question, prompt)
        for label, text in choices.items():
            self.assertIn(f"{label}. {text}", prompt)

    @patch("demo.model_utils._generate")
    def test_returns_correct_letter(self, mock_generate):
        """predict_mcq should return the uppercase letter when _generate returns it."""
        mock_generate.return_value = "A"
        tokenizer, model = self._make_mocks()
        choices = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Madrid"}

        result = predict_mcq(tokenizer, model, "Q", choices)
        self.assertEqual(result, "A")

    @patch("demo.model_utils._generate")
    def test_raises_on_unexpected_token(self, mock_generate):
        """predict_mcq should raise ValueError when the model returns an unexpected token."""
        mock_generate.return_value = "X"
        tokenizer, model = self._make_mocks()
        choices = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Madrid"}

        with self.assertRaises(ValueError) as ctx:
            predict_mcq(tokenizer, model, "Q", choices)
        self.assertIn("unexpected MCQ token", str(ctx.exception))

    def test_raises_on_missing_choice_keys(self):
        """predict_mcq should raise ValueError when choices dict is missing required keys."""
        tokenizer, model = self._make_mocks()
        choices = {"A": "Paris", "B": "London", "C": "Berlin"}  # missing D

        with self.assertRaises(ValueError) as ctx:
            predict_mcq(tokenizer, model, "Q", choices)
        self.assertIn("missing keys", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
