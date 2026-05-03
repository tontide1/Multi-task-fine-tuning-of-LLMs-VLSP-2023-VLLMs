"""
load_comprehension_short_answer_seed.py
=======================================
Chuyển đổi dữ liệu Comprehension thô sang dạng Tự luận ngắn (Short Answer).
Giúp Model rèn luyện kỹ năng trích xuất thông tin trực tiếp từ văn bản.

Input mặc định: seed UIT-only (taidng/UIT-ViQuAD2.0) từ filter script.
Không dùng ShynBui trong nhánh short-answer.
"""

import hashlib
import json
from pathlib import Path

from tqdm import tqdm

# Cấu hình đường dẫn
INPUT_FILE = "seed_exports/comprehension_seed_raw_uit_only.jsonl"
OUTPUT_FILE = "seed_exports/comprehension_short_answer_seed.jsonl"
REPORT_FILE = "seed_exports/comprehension_short_answer_report.json"
TASK_NAME = "comprehension_short_answer"
REQUIRED_SOURCE_DATASET = "taidng/UIT-ViQuAD2.0"


def build_instruction_prompt(context: str, question: str) -> str:
    return (
        f"Dựa vào đoạn văn sau đây, hãy trả lời câu hỏi ngắn gọn:\n\n"
        f"Đoạn văn: {context}\n\nCâu hỏi: {question}"
    )


def make_dedup_hash(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def main() -> None:
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(
            f"[Lỗi] Không tìm thấy file {INPUT_FILE}. "
            "Chạy scripts/filter_comprehension_raw_uit.py sau khi có comprehension_seed_raw.jsonl."
        )
        return

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    total_input = 0
    total_output = 0
    duplicates = 0
    skipped_missing_fields = 0
    skipped_non_target_source = 0
    sources: dict[str, int] = {}

    print(f"Đang đọc dữ liệu từ {INPUT_FILE}...", flush=True)
    with input_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Đang chuyển đổi"):
            if not line.strip():
                continue

            raw_record = json.loads(line)
            total_input += 1

            metadata = raw_record.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            source = metadata.get("source_dataset", "unknown")
            if source != REQUIRED_SOURCE_DATASET:
                skipped_non_target_source += 1
                continue

            context = raw_record.get("context")
            question = raw_record.get("question")
            answer = raw_record.get("answer_text")

            if not context or not question or not answer:
                skipped_missing_fields += 1
                continue

            user_content = build_instruction_prompt(str(context), str(question))
            assistant_content = str(answer)

            dedup_hash = make_dedup_hash(f"{user_content}|{assistant_content}")
            if dedup_hash in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(dedup_hash)

            record: dict[str, object] = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "metadata": {
                    "task": TASK_NAME,
                    "source": "public",
                    "source_dataset": source,
                    "source_id": metadata.get("source_id", "unknown"),
                    "language": "vi",
                    "difficulty": "medium",
                    "dedup_hash": dedup_hash,
                },
            }
            records.append(record)
            total_output += 1
            sources[str(source)] = sources.get(str(source), 0) + 1

    stats = {
        "input_file": str(input_path.resolve()),
        "required_source_dataset": REQUIRED_SOURCE_DATASET,
        "total_input": total_input,
        "total_output": total_output,
        "duplicates": duplicates,
        "skipped_missing_fields": skipped_missing_fields,
        "skipped_non_target_source": skipped_non_target_source,
        "sources": sources,
    }

    print(f"\nĐang lưu {len(records)} mẫu vào {OUTPUT_FILE}...")
    with output_path.open("w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Đang lưu báo cáo vào {REPORT_FILE}...")
    with Path(REPORT_FILE).open("w", encoding="utf-8") as rep:
        json.dump(stats, rep, indent=2, ensure_ascii=False)

    print("\n--- Hoàn tất! ---")
    print(f"Đầu vào: {total_input} mẫu")
    print(f"Đầu ra : {total_output} mẫu")
    print(f"Bỏ qua (thiếu field): {skipped_missing_fields}")
    print(f"Bỏ qua (source khác UIT): {skipped_non_target_source}")
    print(f"Trùng lặp: {duplicates} mẫu đã loại bỏ")


if __name__ == "__main__":
    main()
