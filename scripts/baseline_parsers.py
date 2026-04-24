from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


VALID_MCQ_LABELS = {"A", "B", "C", "D"}

MCQ_OK_STRICT = "ok_strict"
MCQ_OK_FALLBACK = "ok_fallback"
MCQ_INVALID_NO_LABEL = "invalid_no_label"
MCQ_INVALID_MULTI_LABEL = "invalid_multi_label"
MCQ_INVALID_OUT_OF_SET = "invalid_out_of_set"
MCQ_INVALID_EMPTY = "invalid_empty"

STRICT_MCQ_RE = re.compile(r"^\s*Đáp án\s*:\s*([A-Za-z])\s*[.)]?\s*$", re.IGNORECASE)
SINGLE_MCQ_RE = re.compile(r"^\s*([A-Za-z])\s*[.)]?\s*$", re.IGNORECASE)
PREFIX_MCQ_RE = re.compile(r"^\s*(?:Phương án|Phuong an|Chọn|Chon)\s*([A-Za-z])\s*$", re.IGNORECASE)
INLINE_MCQ_RE = re.compile(r"\b(?:chọn|chon)\s*([A-Za-z])\b", re.IGNORECASE)
TRAILING_PUNCT_RE = re.compile(r"[\.,;:!?]+$")


@dataclass(frozen=True)
class MCQParseResult:
    label: Optional[str]
    status: str


def _normalize_unicode(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _first_non_empty_lines(text: str, limit: int = 3) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _normalize_label(label: str) -> str:
    return _normalize_unicode(label).strip().upper().rstrip(".)")


def normalize_mcq_answer_key(answer_key: str) -> Optional[str]:
    if answer_key is None:
        return None
    value = _normalize_unicode(str(answer_key)).strip()
    if not value:
        return None
    for pattern in (STRICT_MCQ_RE, SINGLE_MCQ_RE, PREFIX_MCQ_RE):
        matched = pattern.match(value)
        if not matched:
            continue
        label = _normalize_label(matched.group(1))
        return label if label in VALID_MCQ_LABELS else None
    alpha_match = re.search(r"[A-Za-z]", value)
    if not alpha_match:
        return None
    label = _normalize_label(alpha_match.group(0))
    return label if label in VALID_MCQ_LABELS else None


def parse_mcq_prediction(prediction: str) -> MCQParseResult:
    if prediction is None:
        return MCQParseResult(label=None, status=MCQ_INVALID_EMPTY)
    text = _normalize_unicode(str(prediction)).strip()
    if not text:
        return MCQParseResult(label=None, status=MCQ_INVALID_EMPTY)

    lines = _first_non_empty_lines(text, limit=3)
    if not lines:
        return MCQParseResult(label=None, status=MCQ_INVALID_EMPTY)

    valid_labels: list[str] = []
    valid_sources: list[str] = []
    out_of_set_labels: list[str] = []

    def register(raw_label: str, source: str) -> None:
        label = _normalize_label(raw_label)
        if label in VALID_MCQ_LABELS:
            valid_labels.append(label)
            valid_sources.append(source)
        else:
            out_of_set_labels.append(label)

    for line in lines:
        strict_match = STRICT_MCQ_RE.match(line)
        if strict_match:
            register(strict_match.group(1), MCQ_OK_STRICT)
            continue

        single_match = SINGLE_MCQ_RE.match(line)
        if single_match:
            register(single_match.group(1), MCQ_OK_FALLBACK)
            continue

        prefix_match = PREFIX_MCQ_RE.match(line)
        if prefix_match:
            register(prefix_match.group(1), MCQ_OK_FALLBACK)
            continue

        inline_match = INLINE_MCQ_RE.search(line)
        if inline_match:
            register(inline_match.group(1), MCQ_OK_FALLBACK)

    if valid_labels and out_of_set_labels:
        return MCQParseResult(label=None, status=MCQ_INVALID_MULTI_LABEL)

    if len(set(valid_labels)) > 1:
        return MCQParseResult(label=None, status=MCQ_INVALID_MULTI_LABEL)

    if valid_labels:
        status = MCQ_OK_STRICT if all(source == MCQ_OK_STRICT for source in valid_sources) else MCQ_OK_FALLBACK
        return MCQParseResult(label=valid_labels[0], status=status)

    if out_of_set_labels:
        return MCQParseResult(label=None, status=MCQ_INVALID_OUT_OF_SET)

    return MCQParseResult(label=None, status=MCQ_INVALID_NO_LABEL)


def is_mcq_prediction_correct(prediction: str, answer_key: str) -> tuple[bool, MCQParseResult, Optional[str]]:
    parsed = parse_mcq_prediction(prediction)
    gold_label = normalize_mcq_answer_key(answer_key)
    if gold_label is None:
        return False, parsed, None
    if parsed.label is None:
        return False, parsed, gold_label
    return parsed.label == gold_label, parsed, gold_label


def normalize_lambada_text(value: str) -> str:
    if value is None:
        return ""
    text = _normalize_unicode(str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    text = TRAILING_PUNCT_RE.sub("", text).strip()
    return text.casefold()


def is_lambada_prediction_correct(prediction: str, target_word: str) -> bool:
    normalized_target = normalize_lambada_text(target_word)
    if not normalized_target:
        return False
    return normalize_lambada_text(prediction) == normalized_target
