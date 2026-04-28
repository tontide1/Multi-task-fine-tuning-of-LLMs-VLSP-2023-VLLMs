import unittest
from unittest.mock import patch

from google.api_core import exceptions as google_exceptions

import scripts.fix_math_issues_using_gemini as fixer


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModels:
    def __init__(self, effects: list) -> None:
        self._effects = list(effects)
        self.call_count = 0

    def generate_content(self, *args, **kwargs):
        self.call_count += 1
        effect = self._effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class FakeClient:
    def __init__(self, effects: list) -> None:
        self.models = FakeModels(effects)


class TestGeminiRetryBackoff(unittest.TestCase):
    def test_retryable_service_unavailable_uses_backoff(self) -> None:
        batch = [{"source_id": "1", "text": "x", "flags": []}]
        response = FakeResponse('[{"source_id": "1", "fixed_text": "ok"}]')
        client = FakeClient([google_exceptions.ServiceUnavailable("down"), response])

        with (
            patch.object(fixer, "MAX_RETRIES", 2),
            patch.object(fixer, "BASE_BACKOFF_SECONDS", 2),
            patch.object(fixer, "MAX_BACKOFF_SECONDS", 60),
            patch.object(fixer, "RETRY_JITTER", 0.25),
            patch("scripts.fix_math_issues_using_gemini.random.random", return_value=0.0),
            patch("scripts.fix_math_issues_using_gemini.time.sleep") as sleep_mock,
        ):
            result = fixer.call_gemini_batch(client, batch)

        self.assertEqual(result["1"], "ok")
        self.assertEqual(client.models.call_count, 2)
        sleep_mock.assert_called_once()
        self.assertEqual(sleep_mock.call_args[0][0], 2.0)

    def test_non_retryable_error_breaks_without_sleep(self) -> None:
        batch = [{"source_id": "1", "text": "x", "flags": []}]
        client = FakeClient([ValueError("boom")])

        with (
            patch.object(fixer, "MAX_RETRIES", 3),
            patch("scripts.fix_math_issues_using_gemini.time.sleep") as sleep_mock,
        ):
            with self.assertRaises(RuntimeError):
                fixer.call_gemini_batch(client, batch)

        self.assertEqual(client.models.call_count, 1)
        sleep_mock.assert_not_called()

    def test_retryable_error_exhausts_retries_and_raises(self) -> None:
        batch = [{"source_id": "1", "text": "x", "flags": []}]
        client = FakeClient(
            [
                google_exceptions.ServiceUnavailable("still down"),
                google_exceptions.ServiceUnavailable("still down"),
                google_exceptions.ServiceUnavailable("still down"),
            ]
        )

        with (
            patch.object(fixer, "MAX_RETRIES", 3),
            patch.object(fixer, "BASE_BACKOFF_SECONDS", 2),
            patch.object(fixer, "MAX_BACKOFF_SECONDS", 60),
            patch.object(fixer, "RETRY_JITTER", 0.25),
            patch("scripts.fix_math_issues_using_gemini.random.random", return_value=0.0),
            patch("scripts.fix_math_issues_using_gemini.time.sleep") as sleep_mock,
        ):
            with self.assertRaises(RuntimeError):
                fixer.call_gemini_batch(client, batch)

        self.assertEqual(len(sleep_mock.call_args_list), 3)
        sleep_durations = [c.args[0] for c in sleep_mock.call_args_list]
        self.assertEqual(sleep_durations, [2, 4, 8])


if __name__ == "__main__":
    unittest.main()
