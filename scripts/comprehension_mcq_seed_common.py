import hashlib
import json
import re
from difflib import SequenceMatcher


ANSWER_LABELS = ["A", "B", "C", "D"]


def normalize_for_hash(text):
    if text is None:
        return ""
    normalized = str(text).replace("\u00a0", " ").lower().strip()
    return re.sub(r"\s+", " ", normalized)


def canonical_choice_text(text):
    if text is None:
        return ""
    value = str(text).replace("\u00a0", " ")
    value = re.sub(r"^\s*[A-Da-d]\s*[\.)\:\-]\s*", "", value)
    value = value.replace("_", " ")
    return re.sub(r"\s+", " ", value).strip().lower()


def build_mcq_user_content(context, question, choices):
    return (
        "Đọc đoạn văn sau và chọn đáp án đúng.\n\n"
        f"Đoạn văn: {context}\n\n"
        f"Câu hỏi: {question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}"
    )


def split_mcq_user_content(content):
    lines = str(content).splitlines()
    if len(lines) < 9:
        raise ValueError("expected MCQ user content with passage, question, and 4 choices")
    if lines[0] != "Đọc đoạn văn sau và chọn đáp án đúng.":
        raise ValueError("invalid prompt header")
    if lines[1] != "":
        raise ValueError("expected blank lines in MCQ prompt")
    if not lines[2].startswith("Đoạn văn: "):
        raise ValueError("missing context line")
    question_index = None
    for index in range(3, len(lines)):
        if lines[index].startswith("Câu hỏi: "):
            question_index = index
            break
    if question_index is None:
        raise ValueError("missing question line")

    context_lines = [lines[2][len("Đoạn văn: ") :]]
    context_lines.extend(lines[3:question_index])
    while context_lines and context_lines[-1] == "":
        context_lines.pop()

    choice_prefixes = ["A. ", "B. ", "C. ", "D. "]
    choice_start = None
    for index in range(question_index + 1, len(lines)):
        if any(lines[index].startswith(prefix) for prefix in choice_prefixes):
            choice_start = index
            break
    if choice_start is None:
        raise ValueError("missing choice lines")

    question_lines = [lines[question_index][len("Câu hỏi: ") :]]
    question_lines.extend(lines[question_index + 1 : choice_start])
    while question_lines and question_lines[-1] == "":
        question_lines.pop()

    choices = []
    choice_lines = lines[choice_start:]
    if len(choice_lines) != 4:
        raise ValueError("expected exactly 4 choice lines")
    for line, prefix in zip(choice_lines, choice_prefixes):
        if not line.startswith(prefix):
            raise ValueError("invalid choice line")
        choices.append(line[len(prefix) :])

    return "\n".join(context_lines), "\n".join(question_lines), choices


def make_dedup_hash(context, question, answer_text):
    payload = "\n".join(
        [
            normalize_for_hash(context),
            normalize_for_hash(question),
            normalize_for_hash(answer_text),
        ]
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def make_mcq_dedup_hash(context, question, choices):
    payload = "\n".join(
        [
            normalize_for_hash(context),
            normalize_for_hash(question),
            *[canonical_choice_text(choice) for choice in choices],
        ]
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def compute_context_hash(context):
    return hashlib.sha1(normalize_for_hash(context).encode("utf-8")).hexdigest()


def extract_json_object(raw_text):
    text = "" if raw_text is None else str(raw_text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def stable_choice_labels(seed_text):
    order = list(ANSWER_LABELS)
    digest = hashlib.sha1(normalize_for_hash(seed_text).encode("utf-8")).digest()
    for index in range(len(order) - 1, 0, -1):
        swap_index = digest[index % len(digest)] % (index + 1)
        order[index], order[swap_index] = order[swap_index], order[index]
    return order


def is_near_duplicate_text(left, right):
    a = canonical_choice_text(left)
    b = canonical_choice_text(right)
    if not a or not b:
        return False
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.92
