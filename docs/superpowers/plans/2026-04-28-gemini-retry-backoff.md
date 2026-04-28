# Gemini Retry Backoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add retry/backoff for Gemini 503/429 errors and lower batch size to reduce transient failures without changing data schemas or prompt behavior.

**Architecture:** Update `call_gemini_batch` to detect retryable errors and apply exponential backoff with jitter, while treating non-retryable errors as hard failures. Keep changes localized to the existing script and add unittest coverage with mocked clients and sleep calls.

**Tech Stack:** Python 3.12, google-genai SDK, google-api-core exceptions, unittest.

---

## File Structure
- `scripts/fix_math_issues_using_gemini.py`: Config and Gemini call path. Add retry/backoff config, error classification, and backoff sleep logic.
- `scripts/test_fix_math_issues_using_gemini_retry.py`: New unittest module to verify retryable vs non-retryable behavior and backoff sleep timing.

### Task 1: Add retry/backoff unit tests

**Files:**
- Create: `scripts/test_fix_math_issues_using_gemini_retry.py`
- Test: `scripts/test_fix_math_issues_using_gemini_retry.py`

- [ ] **Step 1: Write the failing test file**

```python
import unittest
from unittest.mock import patch

from google.api_core import exceptions as google_exceptions

import scripts.fix_math_issues_using_gemini as fixer


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModels:
    def __init__(self, effects: list[object]) -> None:
        self._effects = list(effects)
        self.call_count = 0

    def generate_content(self, *args, **kwargs):
        self.call_count += 1
        effect = self._effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class FakeClient:
    def __init__(self, effects: list[object]) -> None:
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
        self.assertEqual(sleep_mock.call_args.args[0], 2.0)

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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to confirm failure**

Run: `python -m unittest scripts.test_fix_math_issues_using_gemini_retry`

Expected: FAIL because retry/backoff config and error classification are not yet implemented (for example, missing constants like `BASE_BACKOFF_SECONDS` or missing retry behavior).

### Task 2: Implement backoff config and retry logic

**Files:**
- Modify: `scripts/fix_math_issues_using_gemini.py:1-410`
- Test: `scripts/test_fix_math_issues_using_gemini_retry.py`

- [ ] **Step 1: Add config and imports**

```python
import random
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.api_core import exceptions as google_exceptions

USE_GEMINI = True
BATCH_SIZE = 2
SLEEP_SECONDS = 4
MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 2
MAX_BACKOFF_SECONDS = 60
RETRY_JITTER = 0.25
RETRYABLE_CODES = {429, 503}
```

- [ ] **Step 2: Update `call_gemini_batch` error handling to back off on 503/429**

```python
        except Exception as e:
            last_error = e
            code = None
            if isinstance(e, google_exceptions.TooManyRequests):
                code = 429
            elif isinstance(e, google_exceptions.ResourceExhausted):
                code = 429
            elif isinstance(e, google_exceptions.ServiceUnavailable):
                code = 503
            elif isinstance(e, google_exceptions.GoogleAPICallError):
                code = getattr(e, "code", None)
                if callable(code):
                    code = code()

            if code is None:
                code = getattr(getattr(e, "status_code", None), "value", None)
            if code is None:
                code = getattr(e, "status_code", None)

            if isinstance(code, int):
                retryable = code in RETRYABLE_CODES
            elif code is None:
                retryable = False
            else:
                retryable = str(code) in {str(c) for c in RETRYABLE_CODES}

            if retryable:
                backoff = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))
                jitter = backoff * RETRY_JITTER * random.random()
                sleep_seconds = backoff + jitter
                print(
                    f"[Gemini retry {attempt}/{MAX_RETRIES}] code={code} "
                    f"sleep={sleep_seconds:.2f}s error={e}"
                )
                time.sleep(sleep_seconds)
            else:
                print(f"[Gemini error] code={code} error={e}")
                break
```

- [ ] **Step 3: Run the tests to confirm they pass**

Run: `python -m unittest scripts.test_fix_math_issues_using_gemini_retry`

Expected: PASS

- [ ] **Step 4: Run a quick syntax check**

Run: `python -m py_compile scripts/fix_math_issues_using_gemini.py`

Expected: no output

- [ ] **Step 5: Commit**

```bash
git add scripts/fix_math_issues_using_gemini.py scripts/test_fix_math_issues_using_gemini_retry.py
git commit -m "feat: add Gemini retry backoff"
```
