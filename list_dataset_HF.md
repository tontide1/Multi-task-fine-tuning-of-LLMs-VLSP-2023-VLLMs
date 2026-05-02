> **Trạng thái build seed (cập nhật cùng repo, để ý khi đọc):**
>
> | Seed file | Trạng thái | Ghi chú |
> |---|---|---|
> | `seed_exports/exams_mcq_seed.jsonl` | **Đã build** | xuất bởi `scripts/load_exams_mcq_seed.py` |
> | `seed_exports/wiki_mcq_seed.jsonl` | **Đã build** | seed thô đã clean schema (xuất bởi `scripts/load_wiki_mcq_seed.py`) |
> | `seed_exports/wiki_mcq_seed_clean.jsonl` | **Đã build** | seed sạch đã whitelist subject, dùng cho stage-2 public seed |
> | `seed_exports/wiki_mcq_seed_dropped.jsonl` | **Đã build** | archive audit/drop |
> | `seed_exports/comprehension_seed_raw.jsonl` | **Đã build** | xuất bởi `scripts/load_comprehension_seed_raw.py` |
> | `seed_exports/comprehension_short_answer_seed.jsonl` | **Đã build** | train đọc hiểu dạng trả lời ngắn (UIT-only); xuất bởi `scripts/load_comprehension_short_answer_seed.py` sau `filter_comprehension_raw_uit.py` |
> | `seed_exports/instruction_retention_seed.jsonl` | **Đã build** | xuất bởi `scripts/load_instruction_retention_seed.py` |
> | `seed_exports/cloze_lm_retention_seed.jsonl` | **Đã build** | xuất bởi `scripts/load_cloze_lm_retention_seed.py` |

## 1) `exams_mcq_seed.jsonl`

Dùng 2 nguồn:

* **`roshansk23/Vietnam_HighSchool_Exam_Dataset`**
  Vai trò: nguồn `exams_mcq` chính cho đề thi THPT kiểu MCQ. Dataset này có khoảng **6,663** mẫu, 8 môn, với các field quan trọng như `question`, `options`, `answer`. ([Hugging Face][1])

* **`hllj/vi_grade_school_math_mcq`**
  Vai trò: bổ sung thêm MCQ Toán tiếng Việt. Dataset này có khoảng **2,733** mẫu, với schema rất tiện cho bạn: `question`, `choices`, `answer`, `explanation`. Card cũng nói rõ dữ liệu chưa clean hoàn toàn, nên vẫn cần QC sau khi convert. ([Hugging Face][2])

## 2) `wiki_mcq_seed.jsonl`

Dùng 1 nguồn:

* **`lighteval/okapi_mmlu`** với config **`vi`** trong notebook
  Vai trò: nguồn `wiki_mcq` / general-knowledge MCQ. Dataset `lighteval/okapi_mmlu` là bản parquet, có schema phù hợp gồm `question`, `choices`, `answer`, `subject`, `id`, nên dùng được ngay thay cho bản script-based trước đó. ([Hugging Face][3])

## 3) `comprehension_seed_raw.jsonl` và train **`comprehension_short_answer`**

Pool thô `comprehension_seed_raw.jsonl` dùng 2 nguồn:

* **`taidng/UIT-ViQuAD2.0`**
  Vai trò: nguồn chính cho reading comprehension tiếng Việt sau khi thay `uit_viwikiqa`. Dataset này là **extractive QA**, format parquet, có khoảng **39,569** mẫu với các field `context`, `question`, `answers`, `is_impossible`, `plausible_answers`. Trong notebook, bạn đang lọc bỏ các mẫu `is_impossible=True`. ([Hugging Face][4])

* **`ShynBui/Vietnamese_Reading_Comprehension_Dataset`**
  Vai trò: nguồn comprehension bổ sung để tăng volume. Dataset này có khoảng **53,845** mẫu, với các field `context`, `question`, `answer`, `answer_start`. Card ghi rõ nó được tổng hợp từ internet/SQuAD/wiki và có phần dịch bằng Google Translate, nên dùng được nhưng cần QC ngôn ngữ kỹ hơn. ([Hugging Face][5])

**Train đọc hiểu trong repo** không còn dùng bucket MCQ (`comprehension_mcq`). Hướng chính hiện tại là **`comprehension_short_answer`** vì đã có extractive QA sạch từ UIT-ViQuAD2.0, trong khi chưa có API LLM/distractor generator đủ tin cậy để chuyển short answer thành MCQ sạch. Nói ngắn gọn: **Better clean extractive QA than noisy generated MCQ.**

Sau khi có raw pool, flow chuẩn là:

1. `scripts/filter_comprehension_raw_uit.py` → `seed_exports/comprehension_seed_raw_uit_only.jsonl`
2. `scripts/load_comprehension_short_answer_seed.py` → `seed_exports/comprehension_short_answer_seed.jsonl` (`metadata.task == comprehension_short_answer`, chỉ UIT-ViQuAD2.0)

## 4) `instruction_retention_seed.jsonl`

Dùng 2 nguồn:

* **`bkai-foundation-models/vi-alpaca`**
  Vai trò: giữ hành vi instruct tiếng Việt. Đây là bộ Vietnamese Alpaca khoảng **50K** instruction samples, được xây theo hướng Self-Instruct/Alpaca. ([Hugging Face][6])

* **`lamhieu/alpaca_multiturns_dialogue_vi`**
  Vai trò: bổ sung chat-style retention. Dataset này ở dạng `messages`, có khoảng **12.7k** mẫu, rất tiện để lấy cặp `user -> assistant` đầu tiên. ([Hugging Face][7])

## 5) `cloze_lm_retention_seed.jsonl`

Dùng 1 nguồn:

* **`VTSNLP/vietnamese_curated_dataset`**
  Vai trò: nguồn text corpus để tạo `cloze_lm_retention`. Dataset này rất lớn, khoảng **12.17 triệu** rows, tiếng Việt, được curate từ nhiều nguồn như C4, OSCAR, Wikipedia và news. Trong notebook bạn chỉ đang sample một subset nhỏ để convert sang prompt cloze. ([Hugging Face][8])

## 6) Dataset đã bỏ / không dùng nữa

* **`SEACrowd/uit_viwikiqa`**
  Hiện tại **không dùng** trong notebook nữa. Lý do là dataset này không kéo thẳng được trong flow hiện tại; card của nó ghi rõ đây là local dataset và phải lấy riêng từ homepage. Vì vậy nó đã được thay bằng `taidng/UIT-ViQuAD2.0`. ([Hugging Face][4])

## 7) Tóm tắt ngắn gọn theo bucket

* `exams_mcq` → `roshansk23/...` + `hllj/...`
* `wiki_mcq` → `lighteval/okapi_mmlu` (`vi`)
* `comprehension_seed_raw` → `taidng/UIT-ViQuAD2.0` + `ShynBui/...`
* `comprehension_short_answer` (train) → UIT-only slice từ pool trên (`scripts/filter_comprehension_raw_uit.py` + `scripts/load_comprehension_short_answer_seed.py`)
* `instruction_retention` → `bkai.../vi-alpaca` + `lamhieu/...`
* `cloze_lm_retention` → `VTSNLP/vietnamese_curated_dataset`


[1]: https://huggingface.co/datasets/roshansk23/Vietnam_HighSchool_Exam_Dataset?utm_source=chatgpt.com "roshansk23/Vietnam_HighSchool_Exam_Dataset · Datasets at Hugging Face"
[2]: https://huggingface.co/datasets/hllj/vi_grade_school_math_mcq?utm_source=chatgpt.com "hllj/vi_grade_school_math_mcq · Datasets at Hugging Face"
[3]: https://huggingface.co/datasets/lighteval/okapi_mmlu?utm_source=chatgpt.com "lighteval/okapi_mmlu · Datasets at Hugging Face"
[4]: https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0?utm_source=chatgpt.com "taidng/UIT-ViQuAD2.0 · Datasets at Hugging Face"
[5]: https://huggingface.co/datasets/ShynBui/Vietnamese_Reading_Comprehension_Dataset?utm_source=chatgpt.com "ShynBui/Vietnamese_Reading_Comprehension_Dataset · Datasets at Hugging Face"
[6]: https://huggingface.co/datasets/bkai-foundation-models/vi-alpaca?utm_source=chatgpt.com "bkai-foundation-models/vi-alpaca · Datasets at Hugging Face"
[7]: https://huggingface.co/datasets/lamhieu/alpaca_multiturns_dialogue_vi?utm_source=chatgpt.com "lamhieu/alpaca_multiturns_dialogue_vi · Datasets at Hugging Face"
[8]: https://huggingface.co/datasets/VTSNLP/vietnamese_curated_dataset?utm_source=chatgpt.com "VTSNLP/vietnamese_curated_dataset · Datasets at Hugging Face"
