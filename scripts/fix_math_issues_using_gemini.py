import os
import re
import json
import random
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.api_core import exceptions as google_exceptions


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")

INPUT_JSONL = Path("seed_exports/wiki_mcq_seed_clean_new.jsonl")
ISSUES_CSV = Path("seed_exports/EDA/wiki_mcq_math_format_issues.csv")

OUTPUT_JSONL = Path("seed_exports/wiki_mcq_seed_clean_math_fixed.jsonl")
LOG_JSONL = Path("seed_exports/EDA/wiki_mcq_math_format_fix_log.jsonl")
REPORT_JSON = Path("seed_exports/EDA/wiki_mcq_math_format_fix_report.json")

USE_GEMINI = True
BATCH_SIZE = 1
SLEEP_SECONDS = 4
MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 2
MAX_BACKOFF_SECONDS = 60
RETRY_JITTER = 0.25
RETRYABLE_CODES = {429, 503}


# ============================================================
# GEMINI STRUCTURED OUTPUT SCHEMA
# ============================================================

class FixItem(BaseModel):
    source_id: str
    fixed_text: str


# ============================================================
# IO HELPERS
# ============================================================

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def chunk_data(data: list[dict], chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


# ============================================================
# USER CONTENT VALIDATION
# ============================================================

def clean_text(x) -> str:
    if x is None:
        return ""
    x = str(x)
    x = x.replace("\u00a0", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def is_valid_user_content(text: str) -> bool:
    if not isinstance(text, str):
        return False

    required_patterns = [
        r"^Câu hỏi:",
        r"\nA\.",
        r"\nB\.",
        r"\nC\.",
        r"\nD\.",
    ]

    return all(re.search(p, text) for p in required_patterns)


def parse_choices_from_user_content(text: str) -> dict | None:
    if not is_valid_user_content(text):
        return None

    choices = {}
    current_label = None

    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"^([ABCD])\.\s*(.*)$", stripped)

        if m:
            current_label = m.group(1)
            choices[current_label] = m.group(2).strip()
            continue

        if current_label is not None:
            choices[current_label] += " " + stripped

    if set(choices.keys()) != {"A", "B", "C", "D"}:
        return None

    return {k: clean_text(v) for k, v in choices.items()}


def validate_fixed_text(original_text: str, fixed_text: str) -> tuple[bool, str]:
    if not is_valid_user_content(fixed_text):
        return False, "fixed_text_broke_user_content_format"

    original_choices = parse_choices_from_user_content(original_text)
    fixed_choices = parse_choices_from_user_content(fixed_text)

    if original_choices is None:
        return False, "original_text_invalid_choice_format"

    if fixed_choices is None:
        return False, "fixed_text_invalid_choice_format"

    if set(original_choices.keys()) != set(fixed_choices.keys()):
        return False, "choice_labels_changed"

    return True, "ok"


# ============================================================
# FORMAT DETECTION
# ============================================================

CURRENCY_DOLLAR_PATTERN = re.compile(
    r"\$\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?|\$\s?\d+\.\d{2}"
)

LATEX_FRAC_PATTERN = re.compile(r"\\frac\{[^{}\n]+\}\{[^{}\n]+\}")
LATEX_SQRT_PATTERN = re.compile(r"\\sqrt\{[^{}\n]+\}")
POWER_PATTERN = re.compile(
    r"(?<![\w$])(?:\d+|[a-zA-Z])\^\{[^{}\n]+\}(?![\w$])"
    r"|(?<![\w$])(?:\d+|[a-zA-Z])\^\d+(?![\w$])"
)


def math_dollar_positions(text: str) -> list[int]:
    ignored_positions = {
        m.start()
        for m in CURRENCY_DOLLAR_PATTERN.finditer(text)
    }

    return [
        m.start()
        for m in re.finditer(r"\$", text)
        if m.start() not in ignored_positions
    ]


def is_inside_math_dollars(text: str, pos: int) -> bool:
    positions = math_dollar_positions(text)
    num_before = sum(1 for p in positions if p < pos)
    return num_before % 2 == 1


def has_unbalanced_math_dollars(text: str) -> bool:
    return len(math_dollar_positions(text)) % 2 != 0


def detect_math_format_flags(text: str) -> list[str]:
    flags = []

    if has_unbalanced_math_dollars(text):
        flags.append("unbalanced_math_dollar")

    for m in LATEX_FRAC_PATTERN.finditer(text):
        if not is_inside_math_dollars(text, m.start()):
            flags.append("raw_latex_frac")
            break

    for m in LATEX_SQRT_PATTERN.finditer(text):
        if not is_inside_math_dollars(text, m.start()):
            flags.append("raw_latex_sqrt")
            break

    for m in POWER_PATTERN.finditer(text):
        if not is_inside_math_dollars(text, m.start()):
            flags.append("raw_caret_power")
            break

    if "\\n" in text:
        flags.append("literal_backslash_n")

    if re.search(r"&[a-zA-Z]+;", text):
        flags.append("html_entity")

    return sorted(set(flags))


# ============================================================
# RULE-BASED FIXES
# ============================================================

def wrap_matches_outside_math(text: str, pattern: re.Pattern) -> str:
    parts = []
    last = 0

    for m in pattern.finditer(text):
        start, end = m.span()
        expr = m.group(0)

        if is_inside_math_dollars(text, start):
            continue

        parts.append(text[last:start])
        parts.append(f"${expr}$")
        last = end

    if last == 0:
        return text

    parts.append(text[last:])
    return "".join(parts)


def deterministic_math_fix(text: str) -> str:
    fixed = text
    fixed = wrap_matches_outside_math(fixed, LATEX_FRAC_PATTERN)
    fixed = wrap_matches_outside_math(fixed, LATEX_SQRT_PATTERN)
    fixed = wrap_matches_outside_math(fixed, POWER_PATTERN)
    return fixed


# ============================================================
# GEMINI
# ============================================================

SYSTEM_PROMPT = """Bạn là Data Engineer. Nhiệm vụ của bạn là sửa lỗi định dạng Toán học (LaTeX/Markdown) cho dữ liệu QA tiếng Việt.

QUY TẮC BẮT BUỘC:
1. CHỈ sửa lỗi định dạng LaTeX/Markdown.
2. Bọc công thức toán học đứng tự do bằng $...$ nếu cần.
3. Đóng dấu $ bị thiếu nếu có thể xác định rõ.
4. KHÔNG giải bài toán.
5. KHÔNG thay đổi nội dung câu hỏi.
6. KHÔNG thay đổi văn phong tiếng Việt.
7. KHÔNG đổi thứ tự hoặc nội dung các đáp án A/B/C/D.
8. PHẢI giữ nguyên format:
   Câu hỏi: ...
   A. ...
   B. ...
   C. ...
   D. ...
9. Đầu ra là JSON Array. Mỗi object có đúng:
   - source_id
   - fixed_text
"""


def make_client():
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY. Put it in .env or environment variables.")
    return genai.Client(api_key=GEMINI_API_KEY)


def parse_gemini_response(response_text: str, batch: list[dict]) -> dict[str, str]:
    batch_source_ids = {str(x["source_id"]) for x in batch}

    data = json.loads(response_text)

    if not isinstance(data, list):
        raise ValueError("Gemini response is not a JSON array.")

    fixed = {}

    for obj in data:
        if not isinstance(obj, dict):
            raise ValueError("Gemini response item is not an object.")

        if "source_id" not in obj or "fixed_text" not in obj:
            raise ValueError(f"Gemini response item missing keys: {obj}")

        source_id = str(obj["source_id"])
        fixed_text = obj["fixed_text"]

        if source_id not in batch_source_ids:
            raise ValueError(f"Gemini returned unexpected source_id: {source_id}")

        if not isinstance(fixed_text, str):
            raise ValueError(f"fixed_text is not string for source_id={source_id}")

        fixed[source_id] = fixed_text

    missing = batch_source_ids - set(fixed.keys())
    if missing:
        raise ValueError(f"Gemini response missing source_id(s): {sorted(missing)}")

    return fixed


def extract_gemini_error_code(err: Exception):
    code = None
    if isinstance(err, google_exceptions.TooManyRequests):
        code = 429
    elif isinstance(err, google_exceptions.ResourceExhausted):
        code = 429
    elif isinstance(err, google_exceptions.ServiceUnavailable):
        code = 503
    elif isinstance(err, google_exceptions.GoogleAPICallError):
        code = getattr(err, "code", None)
        if callable(code):
            code = code()

    if code is None:
        code = getattr(getattr(err, "status_code", None), "value", None)
    if code is None:
        code = getattr(err, "status_code", None)
    if code is None:
        match = re.search(r"\b(429|503)\b", str(err))
        if match:
            code = int(match.group(1))

    return code


def is_retryable_code(code) -> bool:
    if isinstance(code, int):
        return code in RETRYABLE_CODES
    if code is None:
        return False
    return str(code) in {str(c) for c in RETRYABLE_CODES}


def call_gemini_batch(client, batch: list[dict]) -> dict[str, str]:
    prompt = (
        SYSTEM_PROMPT
        + "\n\nDỮ LIỆU ĐẦU VÀO JSON:\n"
        + json.dumps(batch, ensure_ascii=False)
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                    "response_schema": list[FixItem],
                },
            )

            return parse_gemini_response(response.text, batch)

        except Exception as e:
            last_error = e
            code = extract_gemini_error_code(e)
            retryable = is_retryable_code(code)

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

    raise RuntimeError(f"Gemini batch failed after retries: {last_error}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading JSONL...")
    rows = read_jsonl(INPUT_JSONL)
    source_id_to_row = {
        str(row["metadata"]["source_id"]): row
        for row in rows
    }

    print("Rows in JSONL:", len(rows))

    print("Loading issue CSV...")
    df_issues = pd.read_csv(ISSUES_CSV)
    df_issues["source_id"] = df_issues["source_id"].astype(str)

    issue_source_ids = df_issues["source_id"].tolist()
    print("Issue rows:", len(issue_source_ids))

    fixed_text_by_source_id = {}
    log_rows = []

    gemini_candidates = []

    # --------------------------------------------------------
    # STEP 1: deterministic rule-based fix
    # --------------------------------------------------------

    for _, issue in df_issues.iterrows():
        source_id = str(issue["source_id"])
        csv_flags = str(issue.get("flags", ""))

        if source_id not in source_id_to_row:
            log_rows.append({
                "source_id": source_id,
                "status": "missing_source_id_in_jsonl",
                "csv_flags": csv_flags,
            })
            continue

        row = source_id_to_row[source_id]
        original_text = row["messages"][0]["content"]

        if not is_valid_user_content(original_text):
            log_rows.append({
                "source_id": source_id,
                "status": "invalid_original_user_content",
                "csv_flags": csv_flags,
            })
            continue

        before_flags = detect_math_format_flags(original_text)
        rule_fixed_text = deterministic_math_fix(original_text)
        after_rule_flags = detect_math_format_flags(rule_fixed_text)

        valid, reason = validate_fixed_text(original_text, rule_fixed_text)
        if not valid:
            log_rows.append({
                "source_id": source_id,
                "status": "rule_fix_rejected",
                "reason": reason,
                "csv_flags": csv_flags,
                "before_flags": before_flags,
                "after_rule_flags": after_rule_flags,
            })
            continue

        if not after_rule_flags:
            if rule_fixed_text != original_text:
                fixed_text_by_source_id[source_id] = rule_fixed_text
                status = "fixed_by_rule"
            else:
                status = "false_positive_or_no_change_needed"

            log_rows.append({
                "source_id": source_id,
                "status": status,
                "csv_flags": csv_flags,
                "before_flags": before_flags,
                "after_rule_flags": after_rule_flags,
            })
            continue

        # Still problematic after deterministic fix.
        if rule_fixed_text != original_text:
            input_for_gemini = rule_fixed_text
        else:
            input_for_gemini = original_text

        gemini_candidates.append({
            "source_id": source_id,
            "text": input_for_gemini,
            "flags": after_rule_flags,
        })

        log_rows.append({
            "source_id": source_id,
            "status": "needs_gemini",
            "csv_flags": csv_flags,
            "before_flags": before_flags,
            "after_rule_flags": after_rule_flags,
        })

    print("Rule-based fixed rows:", sum(1 for x in log_rows if x["status"] == "fixed_by_rule"))
    print("False positives/no change:", sum(1 for x in log_rows if x["status"] == "false_positive_or_no_change_needed"))
    print("Gemini candidates:", len(gemini_candidates))

    # --------------------------------------------------------
    # STEP 2: Gemini fallback
    # --------------------------------------------------------

    if USE_GEMINI and gemini_candidates:
        client = make_client()

        batches = list(chunk_data(gemini_candidates, BATCH_SIZE))

        for batch_idx, batch in enumerate(batches, start=1):
            print(f"Gemini batch {batch_idx}/{len(batches)}...")

            try:
                gemini_fixed = call_gemini_batch(client, batch)

                for item in batch:
                    source_id = item["source_id"]
                    original_jsonl_text = source_id_to_row[source_id]["messages"][0]["content"]
                    fixed_text = gemini_fixed[source_id]

                    valid, reason = validate_fixed_text(original_jsonl_text, fixed_text)
                    post_flags = detect_math_format_flags(fixed_text)

                    if not valid:
                        log_rows.append({
                            "source_id": source_id,
                            "status": "gemini_fix_rejected",
                            "reason": reason,
                            "post_flags": post_flags,
                        })
                        continue

                    fixed_text_by_source_id[source_id] = fixed_text

                    log_rows.append({
                        "source_id": source_id,
                        "status": "fixed_by_gemini",
                        "post_flags": post_flags,
                    })

            except Exception as e:
                for item in batch:
                    log_rows.append({
                        "source_id": item["source_id"],
                        "status": "gemini_batch_failed",
                        "error": str(e),
                    })

            time.sleep(SLEEP_SECONDS)

    elif gemini_candidates:
        print("Gemini disabled. Remaining candidates will not be fixed.")

    # --------------------------------------------------------
    # STEP 3: merge fixes into output JSONL
    # --------------------------------------------------------

    updated_rows = []
    updated_count = 0

    for row in rows:
        source_id = str(row["metadata"]["source_id"])

        if source_id in fixed_text_by_source_id:
            old_text = row["messages"][0]["content"]
            new_text = fixed_text_by_source_id[source_id]

            valid, reason = validate_fixed_text(old_text, new_text)
            if not valid:
                raise ValueError(f"Refusing to merge invalid fix for {source_id}: {reason}")

            row["messages"][0]["content"] = new_text

            row["metadata"].setdefault("qc_flags", [])
            if "math_format_fixed" not in row["metadata"]["qc_flags"]:
                row["metadata"]["qc_flags"].append("math_format_fixed")

            row["metadata"]["math_format_fix"] = {
                "old_text_char_len": len(old_text),
                "new_text_char_len": len(new_text),
                "post_fix_flags": detect_math_format_flags(new_text),
            }

            updated_count += 1

        updated_rows.append(row)

    updated_rows_by_source_id = {
        str(r["metadata"]["source_id"]): r
        for r in updated_rows
    }

    # --------------------------------------------------------
    # STEP 4: final recheck only issue rows
    # --------------------------------------------------------

    remaining_issue_flags = {}

    for source_id in issue_source_ids:
        if source_id not in source_id_to_row:
            continue

        row = updated_rows_by_source_id[source_id]
        flags = detect_math_format_flags(row["messages"][0]["content"])

        if flags:
            remaining_issue_flags[source_id] = flags

    # --------------------------------------------------------
    # STEP 5: write outputs
    # --------------------------------------------------------

    write_jsonl(OUTPUT_JSONL, updated_rows)
    write_jsonl(LOG_JSONL, log_rows)

    status_counter = {}
    for log in log_rows:
        status = log.get("status", "unknown")
        status_counter[status] = status_counter.get(status, 0) + 1

    report = {
        "input_jsonl": str(INPUT_JSONL),
        "issues_csv": str(ISSUES_CSV),
        "output_jsonl": str(OUTPUT_JSONL),
        "log_jsonl": str(LOG_JSONL),
        "model": MODEL_NAME,
        "use_gemini": USE_GEMINI,
        "jsonl_rows": len(rows),
        "issue_rows": len(issue_source_ids),
        "updated_rows": updated_count,
        "gemini_candidates": len(gemini_candidates),
        "status_counts": status_counter,
        "remaining_issue_rows_with_flags": len(remaining_issue_flags),
        "remaining_issue_flags_sample": dict(list(remaining_issue_flags.items())[:20]),
    }

    write_json(REPORT_JSON, report)

    print("\nDONE")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
