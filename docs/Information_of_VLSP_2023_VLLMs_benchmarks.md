 **Tổng quan**

Hub `vlsp-2023-vllm` công khai 4 benchmark dataset mà bạn đang focus là **`lambada_vi`**, **`wikipediaqa_vi`** (GeneralQA / GKQA), **`exams_vi`**, và **`comprehension`**; ngoài ra org còn có `ViLLM-Eval` là repo benchmark dựa trên `lm-eval-harness` để chạy các task này. ([Hugging Face][1])

Về bản chất, đây là một bộ benchmark **tự động**, không cần human judge như VMLU. Tuy nhiên có hai kiểu task khác nhau: `lambada_vi` là dạng **predict next word**, còn `wikipediaqa_vi`, `exams_vi`, `comprehension` là dạng **multiple-choice** với nhãn đáp án. ([Hugging Face][2])

**Ghi chú train trong repo:** Benchmark `comprehension` trên VLSP là MCQ, nhưng project **tập trung hoàn toàn vào `comprehension_short_answer`** (trả lời ngắn extractive từ UIT-ViQuAD2.0) cho giai đoạn huấn luyện. Quyết định này nhằm đảm bảo chất lượng dữ liệu: **Better clean extractive QA than noisy generated MCQ.** Project không sử dụng pipeline tạo MCQ cho đọc hiểu do thiếu API LLM/distractor generator đủ tin cậy. Khi đánh giá điểm official `comprehension_vi`, cần lưu ý sự khác biệt format giữa dữ liệu huấn luyện (Short Answer) và benchmark (MCQ).

## 1) Thông tin từng dataset

**`lambada_vi`**
Bộ này có **10,246 dòng**, gồm **246 validation** và **10,000 test**. Các trường chính là `text`, `context`, `target_word`, `metadata`. Điểm khác biệt lớn nhất là nó **không có choices A/B/C/D**; thay vào đó nó cho `context` và `target_word`, nên đây là bài toán đo khả năng mô hình đoán đúng từ mục tiêu ở cuối ngữ cảnh. ([Hugging Face][3])

**`wikipediaqa_vi`**
Bộ này có **2,000 test samples**. Mỗi mẫu có `question`, `choices`, `answerKey`, `metadata`. `choices` chứa 4 phương án, và `answerKey` là nhãn đúng; `metadata` cho thấy dữ liệu đến từ nhiều nhóm như `ai_la_trieu_phu` hay `health`. Đây là bộ phù hợp để benchmark kiến thức tổng quát dạng trắc nghiệm. ([Hugging Face][4])

**`exams_vi`**
Bộ này có **19.2k test samples**. Mỗi mẫu có `question`, `id`, `choices`, `answerKey`, `metadata`; trong `metadata` có cả `grade` và `subject`, ví dụ `Lớp 12`, `MÔN ĐỊA`. Đây là bộ trắc nghiệm học đường nhiều môn, và trong repo benchmark nó được tách thành nhiều task con theo môn học thay vì chỉ chạy bằng một task name duy nhất. ([Hugging Face][5])

**`comprehension`**
Bộ này có **900 test samples**. Mỗi mẫu có `question`, `id`, `choices`, `answerKey`. Đây cũng là benchmark multiple-choice, nhưng câu hỏi thường dài hơn và thiên về đọc hiểu hơn là hỏi kiến thức ngắn. ([Hugging Face][6])

## 2) Repo benchmark chính thức

Cách benchmark chuẩn của họ là dùng repo **`vlsp-2023-vllm/ViLLM-Eval`**. Dataset card của repo này nói rõ họ dùng **`lm-eval-harness`** để chạy benchmark, và README của repo cung cấp lệnh cài đặt cũng như lệnh chạy cho từng task. README cũng lưu ý rằng có thể thêm `trust_remote_code=True` cho custom model, và có thể dùng `load_in_4bit=True` hoặc `load_in_8bit=True`, nhưng điều này **có thể làm giảm hiệu quả đánh giá**. ([Hugging Face][7])

Cài đặt:

```bash
git clone https://huggingface.co/datasets/vlsp-2023-vllm/ViLLM-Eval
cd ViLLM-Eval
pip install -e .
```

Lệnh trên đúng theo README của `ViLLM-Eval`. ([Hugging Face][7])

## 3) Cách benchmark từng bộ

**`lambada_vi`**

```bash
MODEL_ID=your_model
python main.py \
  --model hf-causal \
  --model_args pretrained=$MODEL_ID \
  --tasks lambada_vi \
  --device cuda:0
```

Repo benchmark dùng task name là `lambada_vi`. Vì dataset có `context` và `target_word` thay vì các lựa chọn trắc nghiệm, bạn nên xem đây là benchmark kiểu language-model scoring hơn là QA trắc nghiệm. ([Hugging Face][7])

**`wikipediaqa_vi`**

```bash
MODEL_ID=your_model
python main.py \
  --model hf-causal \
  --model_args pretrained=$MODEL_ID \
  --tasks wikipediaqa_vi \
  --num_fewshot 5 \
  --device cuda:0
```

README của `ViLLM-Eval` ghi rõ `wikipediaqa_vi` được chạy với `--num_fewshot 5`. Vì dataset có `choices` và `answerKey`, đây là benchmark trắc nghiệm dạng few-shot. ([Hugging Face][7])

**`exams_vi`**

```bash
MODEL_ID=your_model
python main.py \
  --model hf-causal \
  --model_args pretrained=$MODEL_ID \
  --tasks exams_dialy_vi,exams_hoahoc_vi,exams_lichsu_vi,exams_sinhhoc_vi,exams_toan_vi,exams_vatly_vi,exams_van_vi \
  --num_fewshot 5 \
  --device cuda:0
```

Điểm quan trọng là repo **không chạy `exams_vi` như một task duy nhất**, mà tách thành 7 task con: địa lý, hóa học, lịch sử, sinh học, toán, vật lý, văn. Đây là điểm bạn cần giữ nguyên nếu muốn bám sát benchmark gốc. ([Hugging Face][7])

**`comprehension`**

```bash
MODEL_ID=your_model
python main.py \
  --model hf-causal \
  --model_args pretrained=$MODEL_ID \
  --tasks comprehension_vi \
  --device cuda:0
```

README của repo dùng task name `comprehension_vi` cho dataset `comprehension`. Ở lệnh mẫu trong README không có `--num_fewshot 5`, nên nếu muốn tái hiện đúng repo hiện tại thì nên giữ nguyên như vậy. ([Hugging Face][7])

## 4) Hiểu đúng “cách benchmark”

Nếu nói đơn giản, với VLSP 2023 bạn có thể hiểu benchmark như sau:

* **`wikipediaqa_vi`, `exams_vi`, `comprehension`**: cho model làm bài multiple-choice và chấm tự động theo `answerKey`. Các bộ này khác nhau chủ yếu ở miền kiến thức và độ dài/độ khó câu hỏi. ([Hugging Face][4])
* **`lambada_vi`**: không phải MCQ; thay vì `answerKey`, nó có `target_word`, nên benchmark là đo mô hình có dự đoán đúng từ mục tiêu từ ngữ cảnh hay không. ([Hugging Face][3])

Nói ngắn gọn: **3 bộ sau là QA trắc nghiệm, 1 bộ đầu là next-word prediction**. ([Hugging Face][3])

[1]: https://huggingface.co/vlsp-2023-vllm?utm_source=chatgpt.com "vlsp-2023-vllm (VLSP 2023 - VLLM)"
[2]: https://huggingface.co/papers/2404.11086?utm_source=chatgpt.com "Paper page - ViLLM-Eval: A Comprehensive Evaluation Suite for Vietnamese Large Language Models"
[3]: https://huggingface.co/datasets/vlsp-2023-vllm/lambada_vi?utm_source=chatgpt.com "vlsp-2023-vllm/lambada_vi · Datasets at Hugging Face"
[4]: https://huggingface.co/datasets/vlsp-2023-vllm/wikipediaqa_vi?utm_source=chatgpt.com "vlsp-2023-vllm/wikipediaqa_vi · Datasets at Hugging Face"
[5]: https://huggingface.co/datasets/vlsp-2023-vllm/exams_vi?utm_source=chatgpt.com "vlsp-2023-vllm/exams_vi · Datasets at Hugging Face"
[6]: https://huggingface.co/datasets/vlsp-2023-vllm/comprehension?utm_source=chatgpt.com "vlsp-2023-vllm/comprehension · Datasets at Hugging Face"
[7]: https://huggingface.co/datasets/vlsp-2023-vllm/ViLLM-Eval?utm_source=chatgpt.com "vlsp-2023-vllm/ViLLM-Eval · Datasets at Hugging Face"
