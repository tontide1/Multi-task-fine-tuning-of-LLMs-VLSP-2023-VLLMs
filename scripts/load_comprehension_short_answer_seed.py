"""
load_comprehension_short_answer_seed.py
=======================================
Chuyển đổi dữ liệu Comprehension thô sang dạng Tự luận ngắn (Short Answer).
Giúp Model rèn luyện kỹ năng trích xuất thông tin trực tiếp từ văn bản.
"""

import json
import hashlib
from pathlib import Path
from tqdm import tqdm

# Cấu hình đường dẫn
INPUT_FILE = "seed_exports/comprehension_seed_raw.jsonl"
OUTPUT_FILE = "seed_exports/comprehension_short_answer_seed.jsonl"
REPORT_FILE = "seed_exports/comprehension_short_answer_report.json"
TASK_NAME = "comprehension_short_answer"

def build_instruction_prompt(context, question):
    return f"Dựa vào đoạn văn sau đây, hãy trả lời câu hỏi ngắn gọn:\n\nĐoạn văn: {context}\n\nCâu hỏi: {question}"

def make_dedup_hash(content):
    return hashlib.sha1(content.encode("utf-8")).hexdigest()

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"[Lỗi] Không tìm thấy file {INPUT_FILE}. Vui lòng chạy scripts/load_comprehension_seed_raw.py trước.")
        return

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    seen_hashes = set()
    stats = {
        "total_input": 0,
        "total_output": 0,
        "duplicates": 0,
        "sources": {}
    }

    print(f"Đang đọc dữ liệu từ {INPUT_FILE}...")
    with open(input_path, "r", encoding="utf-8") as f:
        # Đọc từng dòng để tiết kiệm RAM
        for line in tqdm(f, desc="Đang chuyển đổi"):
            if not line.strip():
                continue

            raw_record = json.loads(line)
            stats["total_input"] += 1

            context = raw_record.get("context")
            question = raw_record.get("question")
            answer = raw_record.get("answer_text")
            metadata = raw_record.get("metadata", {})
            source = metadata.get("source_dataset", "unknown")

            # Bỏ qua nếu thiếu trường dữ liệu
            if not context or not question or not answer:
                continue

            # Format lại thành câu lệnh (Instruction)
            user_content = build_instruction_prompt(context, question)
            assistant_content = answer

            # Hash để lọc trùng lặp (Dedup) dựa trên toàn bộ nội dung tương tác
            dedup_hash = make_dedup_hash(f"{user_content}|{assistant_content}")
            if dedup_hash in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(dedup_hash)

            # Xây dựng Record theo chuẩn dự án
            record = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ],
                "metadata": {
                    "task": TASK_NAME,
                    "source": "public",
                    "source_dataset": source,
                    "source_id": metadata.get("source_id", "unknown"),
                    "language": "vi",
                    "difficulty": "medium",
                    "dedup_hash": dedup_hash
                }
            }
            records.append(record)
            stats["total_output"] += 1
            stats["sources"][source] = stats["sources"].get(source, 0) + 1

    print(f"\nĐang lưu {len(records)} mẫu vào {OUTPUT_FILE}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Đang lưu báo cáo vào {REPORT_FILE}...")
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n--- Hoàn tất! ---")
    print(f"Đầu vào: {stats['total_input']} mẫu")
    print(f"Đầu ra : {stats['total_output']} mẫu chất lượng cao")
    print(f"Trùng lặp: {stats['duplicates']} mẫu đã loại bỏ")

if __name__ == "__main__":
    main()