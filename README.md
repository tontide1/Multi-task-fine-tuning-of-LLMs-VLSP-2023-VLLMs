# Multi-task Fine-tuning of LLMs — VLSP 2023 VLLMs

Pipeline benchmark + fine-tune (QLoRA) cho Vietnamese LLM (`unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit`)
trên bộ benchmark **VLSP 2023 VLLMs** gồm 4 task:

- `lambada_vi` — next-word prediction
- `wikipediaqa_vi` — MCQ kiến thức tổng quát (5-shot)
- `exams_vi` — MCQ học đường, **tách thành 7 task con** theo môn (`exams_dialy_vi`,
  `exams_hoahoc_vi`, `exams_lichsu_vi`, `exams_sinhhoc_vi`, `exams_toan_vi`,
  `exams_vatly_vi`, `exams_van_vi`)
- `comprehension_vi` — MCQ đọc hiểu (0-shot)

## Cấu trúc repo

```
.agents/skills/                # Skills nội bộ (HF CLI, ...)
AGENTS.md / GEMINI.md          # Quy tắc làm việc + môi trường (conda nlp, py3.12)
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
  baseline_parsers.py          # MCQ + Lambada answer parsers
  load_exams_mcq_seed.py       # ETL exams_mcq từ 2 nguồn HF
  load_wiki_mcq_seed.py        # ETL wiki_mcq từ okapi_mmlu (vi)
  recheck_exams_mcq_seed.py    # QC sau export
  recheck_wiki_mcq_seed.py     # QC sau export
  test_baseline_parsers.py     # Unit tests cho parsers
seed_exports/                  # Output ETL (jsonl + report)
```

## Yêu cầu môi trường

- Python 3.12
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

- `comprehension_seed_raw` đã có spec + plan, ETL chưa implement (xem `docs/superpowers/specs/2026-04-29-comprehension-seed-raw-design.md` và `docs/superpowers/plans/2026-04-29-comprehension-seed-raw.md`).

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

Tất cả seed/train sample đều dùng schema chung:

```json
{
  "messages": [
    {"role": "user", "content": "Câu hỏi: ...\nA. ...\nB. ...\nC. ...\nD. ..."},
    {"role": "assistant", "content": "Đáp án: C"}
  ],
  "metadata": {"task": "exams_mcq", "source": "...", "difficulty": "medium"}
}
```

Chi tiết schema và quy ước bucket: `docs/plan/overview.md` mục 2.2.

## Tài liệu liên quan

- [Plan 9 giai đoạn](docs/plan/overview.md)
- [Mô tả 4 dataset VLSP 2023](docs/Information_of_VLSP_2023_VLLMs_benchmarks.md)
- [Danh sách dataset HF dùng cho seed](list_dataset_HF.md)
- [Quy tắc làm việc](AGENTS.md)
