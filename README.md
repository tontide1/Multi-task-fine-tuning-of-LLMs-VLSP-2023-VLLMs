# Multi-task Fine-tuning of LLMs — VLSP 2023 VLLMs

Pipeline benchmark + fine-tune (QLoRA) cho Vietnamese LLM (`unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit`)
trên bộ benchmark **VLSP 2023 VLLMs** gồm 4 task:

- `lambada_vi` — next-word prediction
- `wikipediaqa_vi` — MCQ kiến thức tổng quát (5-shot)
- `exams_vi` — MCQ học đường, **tách thành 7 task con** theo môn (`exams_dialy_vi`,
  `exams_hoahoc_vi`, `exams_lichsu_vi`, `exams_sinhhoc_vi`, `exams_toan_vi`,
  `exams_vatly_vi`, `exams_van_vi`)
- `comprehension_vi` — MCQ đọc hiểu trên VLSP benchmark (0-shot). **Dữ liệu huấn luyện tập trung hoàn toàn** vào nhánh **`comprehension_short_answer`** (trả lời ngắn / extractive QA từ UIT-ViQuAD2.0). Project loại bỏ hướng huấn luyện bằng MCQ cho đọc hiểu để đảm bảo chất lượng: **Better clean extractive QA than noisy generated MCQ**.

## Cấu trúc repo

```
.agents/skills/                # Skills nội bộ (HF CLI, ...)
AGENTS.md / GEMINI.md          # Quy tắc làm việc + môi trường (conda nlp, py3.11)
docs/
  plan/overview.md             # Plan 9 giai đoạn cho project
  Information_of_VLSP_2023_VLLMs_benchmarks.md
list_dataset_HF.md             # Nguồn HF cho từng bucket seed
notebooks/
  baseline-vlsp-2023.ipynb     # Chạy baseline (Colab T4)
  EDA_VLSP_2023.ipynb          # EDA 4 dataset benchmark
  exams_mcq_seed.ipynb         # Build exams_mcq seed
  wiki_mcq_seed.ipynb          # Build wiki_mcq seed
result/baseline/               # Kết quả baseline (JSON, mỗi task 1 file)
scripts/
  baseline_parsers.py                   # MCQ + Lambada answer parsers
  load_exams_mcq_seed.py               # ETL exams_mcq từ 2 nguồn HF
  load_wiki_mcq_seed.py                # ETL wiki_mcq từ okapi_mmlu (vi)
  load_comprehension_seed_raw.py       # ETL comprehension_seed_raw (HF)
  recheck_comprehension_seed_raw.py    # QC sau export raw pool
  filter_comprehension_raw_uit.py      # Lọc UIT-only cho nhánh short answer
  load_comprehension_short_answer_seed.py  # Export comprehension_short_answer_seed.jsonl
  recheck_exams_mcq_seed.py            # QC sau export
  recheck_wiki_mcq_seed.py             # QC sau export
  test_baseline_parsers.py             # Unit tests cho parsers
seed_exports/                  # Output ETL (jsonl + report)
```

## Yêu cầu môi trường

- Python 3.11
- `conda activate nlp` (xem `AGENTS.md`)
- HF account + W&B account (cho training/eval)

## Quick start

### 1. Chạy unit tests cho parsers

```bash
python -m unittest scripts.test_baseline_parsers -v
```

### 2. Build seed dataset (ETL từ HuggingFace)

```bash
# exams_mcq seed: ~7.1 MB jsonl
python scripts/load_exams_mcq_seed.py
python scripts/recheck_exams_mcq_seed.py

# wiki_mcq seed: ~13.8 MB jsonl
python scripts/load_wiki_mcq_seed.py
python scripts/recheck_wiki_mcq_seed.py
```

Output ghi vào `seed_exports/`.

Trạng thái hiện tại:

- `comprehension_seed_raw`: spec/plan trong `docs/superpowers/specs/2026-04-29-comprehension-seed-raw-design.md` và `docs/superpowers/plans/2026-04-29-comprehension-seed-raw.md`; ETL chạy bằng `scripts/load_comprehension_seed_raw.py` + `scripts/recheck_comprehension_seed_raw.py`.
- **Định hướng huấn luyện đọc hiểu:** Tập trung hoàn toàn vào nhánh **`comprehension_short_answer`** (`seed_exports/comprehension_short_answer_seed.jsonl`), build từ pool UIT đã lọc.
- Lý do: Không có API LLM/distractor generator đủ tin cậy để chuyển short answer sang MCQ sạch; **clean extractive QA** mang lại chất lượng dữ liệu tốt hơn hẳn so với **noisy generated MCQ**.
- Pipeline **comprehension MCQ** (sinh distractor + solver QC, …) chỉ còn trong repo dưới dạng **tài liệu/script lịch sử**; không còn là luồng huấn luyện của project.

### Comprehension short answer pipeline

```bash
# Raw extractive pool từ Hugging Face (UIT-ViQuAD2.0 + ShynBui trong raw; short answer chỉ dùng UIT sau bước lọc)
python scripts/load_comprehension_seed_raw.py
python scripts/recheck_comprehension_seed_raw.py

# UIT-only slice + export train short answer (metadata.task == comprehension_short_answer)
python scripts/filter_comprehension_raw_uit.py
python scripts/load_comprehension_short_answer_seed.py
```

Output chính:

- `seed_exports/comprehension_short_answer_seed.jsonl`
- `seed_exports/comprehension_short_answer_report.json`

Ghi chú file seed cho `wiki_mcq`:

- `seed_exports/wiki_mcq_seed.jsonl`: seed thô đã clean schema
- `seed_exports/wiki_mcq_seed_clean.jsonl`: seed sạch đã whitelist subject, dùng cho stage-2 public seed
- `seed_exports/wiki_mcq_seed_dropped.jsonl`: archive audit/drop

### 3. Chạy baseline (cần GPU T4 hoặc tương đương)

Mở `notebooks/baseline-vlsp-2023.ipynb` trên Colab. Notebook cài `ViLLM-Eval`
và chạy `lm_eval` với đúng `--num_fewshot` cho từng task (0 cho lambada/comprehension,
5 cho wikipediaqa/exams).

Sau khi chạy, lưu kết quả vào `result/baseline/{task}.json`.

## Roadmap

Xem `docs/plan/overview.md` để biết chi tiết 9 giai đoạn:

1. Khóa baseline và chuẩn đánh giá
2. Build bộ data train, dev set nội bộ, shadow eval
3. Ổn định pipeline huấn luyện
4. Run thăm dò đầu tiên
5. Run huấn luyện chính thức vòng 1
6. Đánh giá holdout lần 1 trên VLSP 2023
7. Vòng cải tiến có kiểm soát
8. Chốt model ứng viên cuối
9. Đóng project vòng 1

## Schema dữ liệu nội bộ

Tất cả seed/train sample đều dùng container chung `messages` + `metadata`. Ví dụ **MCQ** (`exams_mcq`, `wiki_mcq`):

```json
{
  "messages": [
    {"role": "user", "content": "Câu hỏi: ...\nA. ...\nB. ...\nC. ...\nD. ..."},
    {"role": "assistant", "content": "Đáp án: C"}
  ],
  "metadata": {"task": "exams_mcq", "source": "...", "difficulty": "medium"}
}
```

**Đọc hiểu train** (`comprehension_short_answer`) — prompt theo script export (không có A/B/C/D):

```json
{
  "messages": [
    {"role": "user", "content": "Dựa vào đoạn văn sau đây, hãy trả lời câu hỏi ngắn gọn:\n\nĐoạn văn: ...\n\nCâu hỏi: ..."},
    {"role": "assistant", "content": "...đáp án trích xuất ngắn..."}
  ],
  "metadata": {"task": "comprehension_short_answer", "source": "public", "source_dataset": "taidng/UIT-ViQuAD2.0", "difficulty": "medium"}
}
```

Chi tiết schema và quy ước bucket: `docs/plan/overview.md` mục 2.2.

## Tài liệu liên quan

- [Plan 9 giai đoạn](docs/plan/overview.md)
- [Mô tả 4 dataset VLSP 2023](docs/Information_of_VLSP_2023_VLLMs_benchmarks.md)
- [Danh sách dataset HF dùng cho seed](list_dataset_HF.md)
- [Quy tắc làm việc](AGENTS.md)
