# Gemini Retry Backoff Design

## Goal
Reduce transient Gemini API failures (503/429) by adding exponential backoff with jitter and lowering batch size, without changing data schemas or prompt behavior.

## Scope
- Modify only `scripts/fix_math_issues_using_gemini.py`.
- Add retry/backoff parameters and classification for retryable errors.
- Lower `BATCH_SIZE` to 2.

## Non-Goals
- No switch to Gemini CLI.
- No change to model selection or prompt content.
- No changes to rule-based fixing or validation logic.
- No persistent queue or resume mechanism.

## Architecture
The change is localized to the Gemini call path in `call_gemini_batch`. When a retryable error (503/429) is detected, the call sleeps for an exponential backoff interval with jitter, then retries until `MAX_RETRIES` is reached. Non-retryable errors stop the batch immediately.

## Configuration
- `BATCH_SIZE = 2`
- `MAX_RETRIES = 6`
- `BASE_BACKOFF_SECONDS = 2`
- `MAX_BACKOFF_SECONDS = 60`
- `RETRY_JITTER = 0.25`
- `RETRYABLE_CODES = {429, 503}`

Backoff formula:

```
sleep = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * 2^(attempt-1)) + jitter
jitter = sleep * RETRY_JITTER * random()
```

## Error Classification
Retryable errors are identified via `google.api_core.exceptions`:
- `TooManyRequests` and `ResourceExhausted` map to 429
- `ServiceUnavailable` maps to 503
If a numeric status is available from the exception, it is used directly.

## Logging
Each retry logs attempt count, error code, and sleep duration. Non-retryable errors log the error code and terminate the batch.

## Testing / Verification
- `python -m py_compile scripts/fix_math_issues_using_gemini.py`
- Run the script on a small batch and confirm retry logs appear when 503/429 occurs.

## Rollout
No migration required. Script behavior remains the same on success paths, with improved resilience under transient API overload.
