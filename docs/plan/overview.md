# Plan for fine-tuning model unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit for VLSP 2023 Benchmarks VLLMs

* **QLoRA là mặc định**
* **HF Hub = model registry + eval archive**
* **W&B = source of truth cho experiment**
* **Kaggle = môi trường train chính**
* **Colab Free = môi trường debug / smoke test**
* **VLSP 2023 = holdout benchmark, không dùng để tuning hằng ngày**

Điểm cần giữ xuyên suốt là: **VLSP 2023 chỉ dùng để đo ở các mốc quan trọng**, vì benchmark này có protocol đánh giá riêng và không đồng nhất giữa các task: `lambada_vi` là next-word prediction, còn `wikipediaqa_vi`, `exams_vi`, `comprehension` là multiple-choice; riêng `exams_vi` còn được tách thành nhiều task con trong repo benchmark chính thức.

**Định hướng train đọc hiểu:** benchmark `comprehension` / `comprehension_vi` trên VLSP vẫn là **MCQ**, nhưng **pipeline seed đọc hiểu trong repo đã chuyển** từ bucket **`comprehension_mcq`** sang **`comprehension_short_answer`** (trả lời ngắn extractive trên UIT-ViQuAD2.0). Đây là lựa chọn thực dụng và ít rủi ro nhất hiện tại vì chưa có API LLM/distractor generator đáng tin cậy để tạo MCQ sạch. **Better clean extractive QA than noisy generated MCQ.** Chi tiết script và file export: `README.md`.

---

# Plan tổng thể

## Giai đoạn 1 — Khóa baseline và chuẩn đánh giá

### Mục tiêu

Thiết lập mốc xuất phát rõ ràng trước khi fine-tune.

### Việc cần làm

* Dùng đúng model đã chốt: `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit`
* Chạy baseline inference/evaluation **không fine-tune**
* Đánh giá theo đúng protocol benchmark gốc:

  * `lambada_vi`
  * `wikipediaqa_vi`
  * `exams_vi` theo đúng các task con
  * `comprehension`
* Ghi nhận kết quả theo 3 lớp:

  * tổng quan toàn benchmark
  * theo từng task
  * theo từng nhóm lỗi dễ thấy

### Kết quả đầu ra

* 1 baseline chính thức để về sau so sánh mọi run
* 1 mốc tham chiếu để biết fine-tune có thật sự giúp hay không

### Điều kiện hoàn thành

Bạn phải trả lời được:

* model gốc mạnh ở đâu
* yếu ở đâu
* yếu do kiến thức, đọc hiểu, hay format đáp án

---

## Giai đoạn 2 — Build bộ data train, dev set nội bộ, và shadow eval

### Mục tiêu

Chuyển 5 seed export hiện có thành bộ train/dev/shadow/probe đủ sạch cho vòng QLoRA đầu tiên. Trọng tâm không còn là “nở dữ liệu synthetic” ngay, mà là **khóa dữ liệu seed public**, kiểm định chất lượng, chống trùng/leakage, split deterministic, rồi mới chọn mixture để train.

Official VLSP vẫn giữ nguyên protocol: `wikipediaqa_vi`, `exams_vi`, `comprehension` là MCQ; riêng train đọc hiểu trong repo dùng **`comprehension_short_answer`** để rèn năng lực đọc hiểu grounded.

### Phạm vi đã chốt

Bộ dữ liệu vòng đầu có 5 nhánh:

* **`exams_mcq`**
* **`wiki_mcq`**
* **`comprehension_short_answer`**
* **`instruction_retention`**
* **`cloze_lm_retention`**

### Không làm trong vòng đầu

* chưa build `lambada_cloze` theo quy mô full dataset ngoài `cloze_lm_retention`
* không copy, paraphrase, hay dịch lại benchmark VLSP vào train/dev/shadow eval
* không sinh thêm synthetic bằng LLM trước khi seed hiện có được audit/split rõ ràng
* không tự chuyển `comprehension_short_answer` sang MCQ bằng rule-based distractor vì dễ tạo noisy generated MCQ
* không khôi phục pipeline `comprehension_mcq` làm đường train mặc định trong vòng này

---

## 2.1. Current seed inventory

### Mục tiêu

Khóa trạng thái dữ liệu hiện có trước khi train để mọi run đều truy vết được.

### Seed exports hiện có

| Bucket | File | Count hiện tại | Vai trò |
| --- | --- | ---: | --- |
| `exams_mcq` | `seed_exports/exams_mcq_seed.jsonl` | 11,956 | MCQ học đường / đề thi |
| `wiki_mcq` | `seed_exports/wiki_mcq_seed_final.jsonl` | 7,929 | MCQ kiến thức tổng quát |
| `comprehension_short_answer` | `seed_exports/comprehension_short_answer_seed.jsonl` | 21,676 | Đọc hiểu grounded dạng trả lời ngắn |
| `instruction_retention` | `seed_exports/instruction_retention_seed.jsonl` | 20,000 | Giữ hành vi Instruct/chat tiếng Việt |
| `cloze_lm_retention` | `seed_exports/cloze_lm_retention_seed.jsonl` | 10,000 | Giữ năng lực LM-style / next-word |

Tổng seed hiện có: **71,561 samples**.

### Nguồn và trạng thái

* `exams_mcq`: từ `roshansk23/Vietnam_HighSchool_Exam_Dataset` + `hllj/vi_grade_school_math_mcq`
* `wiki_mcq`: từ `lighteval/okapi_mmlu` config `vi`, bản final đã qua QC
* `comprehension_short_answer`: từ UIT-only slice của `comprehension_seed_raw`, không dùng ShynBui trong train short-answer v1
* `instruction_retention`: từ `bkai-foundation-models/vi-alpaca` + `lamhieu/alpaca_multiturns_dialogue_vi`
* `cloze_lm_retention`: từ `VTSNLP/vietnamese_curated_dataset`

### Kết quả đầu ra

* 1 manifest dữ liệu ghi rõ file, count, task, source, report path, script tạo file, và ngày khóa dữ liệu
* 1 snapshot QC/report cho từng seed file trước khi split

---

## 2.2. Unified schema contract

### Mục tiêu

Đảm bảo mọi sample train/eval nội bộ dùng chung container `messages`, nhưng không ép mọi task về cùng output format.

### MCQ schema (`exams_mcq`, `wiki_mcq`)

```json
{"messages":[{"role":"user","content":"Câu hỏi: ...\nA. ...\nB. ...\nC. ...\nD. ..."},{"role":"assistant","content":"Đáp án: C"}],"metadata":{"task":"exams_mcq","source":"public","difficulty":"medium"}}
```

Quy tắc:

* `user` có câu hỏi và đúng 4 choices A/B/C/D
* `assistant` chỉ trả lời `Đáp án: A/B/C/D`
* `metadata.task` là `exams_mcq` hoặc `wiki_mcq`

### Short-answer schema (`comprehension_short_answer`)

```json
{"messages":[{"role":"user","content":"Dựa vào đoạn văn sau đây, hãy trả lời câu hỏi ngắn gọn:\n\nĐoạn văn: ...\n\nCâu hỏi: ..."},{"role":"assistant","content":"...đáp án trích xuất ngắn..."}],"metadata":{"task":"comprehension_short_answer","source":"public","source_dataset":"taidng/UIT-ViQuAD2.0","difficulty":"medium"}}
```

Quy tắc:

* `user` có passage/context và câu hỏi
* `assistant` là answer text ngắn, không dùng `Đáp án: X`
* answer phải grounded trong passage theo tiêu chí extractive của nguồn

### Retention schema

```json
{"messages":[{"role":"user","content":"...instruction prompt..."},{"role":"assistant","content":"...instruction response..."}],"metadata":{"task":"instruction_retention","source":"public","difficulty":"medium"}}
```

```json
{"messages":[{"role":"user","content":"...cloze or next-word prompt..."},{"role":"assistant","content":"...target word or short completion..."}],"metadata":{"task":"cloze_lm_retention","source":"public","difficulty":"medium"}}
```

Retention dùng cùng container `messages`, nhưng không được gộp metric vào MCQ accuracy.

### Kết quả đầu ra

* validator nội bộ có thể phân biệt đúng format theo `metadata.task`
* không còn nhầm `comprehension_short_answer` với `comprehension` MCQ benchmark

---

## 2.3. QC và audit trước khi split

### Mục tiêu

Lọc và audit seed public trước khi trộn. Đây là gate chính của Giai đoạn 2.

### Check bắt buộc

* JSONL parse được toàn bộ
* mỗi record có đúng 2 messages: `user`, `assistant`
* `metadata.task` thuộc 5 bucket đã chốt
* content không rỗng, không quá ngắn/quá dài bất thường
* dedup hash nội bộ không trùng trong từng bucket
* distribution theo source/task/difficulty hợp lý
* random sample review đủ để phát hiện lỗi format lớn

### Check theo task

* **MCQ**: đúng 4 choices, answer label hợp lệ, assistant format đúng `Đáp án: X`, không còn raw choice prefix trong nội dung lựa chọn
* **`comprehension_short_answer`**: passage/question/answer không rỗng, answer grounded trong passage khi metadata/source cho phép kiểm tra, không bị lẫn prompt MCQ
* **`instruction_retention`**: không còn role marker thừa, response đủ dài để có tín hiệu train
* **`cloze_lm_retention`**: không nhiễm MCQ format, target ngắn và phù hợp dạng LM-style

### Kết quả đầu ra

* report QC cho từng seed export
* danh sách lỗi cần sửa hoặc chấp nhận có lý do
* chỉ seed đã qua gate này mới được đưa vào split

---

## 2.4. Cross-dedup và leakage check

### Mục tiêu

Giữ `train`, `val`, `shadow eval`, và official VLSP tách biệt thật sự.

### Cross-dedup nội bộ

* dedup exact/normalized trên cặp `user + assistant`
* phát hiện trùng gần đúng giữa `exams_mcq` và `wiki_mcq`
* phát hiện prompt/template lặp quá nhiều trong retention
* với `comprehension_short_answer`, dùng passage/source hash để tránh cùng passage xuất hiện ở nhiều split

### Leakage với VLSP

* 4 bộ benchmark VLSP chỉ dùng để eval
* không đưa benchmark vào train/dev/shadow
* không paraphrase, dịch, hoặc dùng benchmark làm nguồn sinh dữ liệu
* kiểm tra lexical/fuzzy match với benchmark khi có local benchmark dump
* với `comprehension`, chú ý gap format: benchmark là MCQ, train là short answer; leakage vẫn phải kiểm tra theo question/passage/fact pattern chứ không chỉ theo answer label

### Kết quả đầu ra

* tập seed đủ điều kiện split
* reject/audit log cho duplicate hoặc leakage suspects

---

## 2.5. Split strategy

### Mục tiêu

Tạo split deterministic đủ ổn định để chọn checkpoint hằng ngày mà không chạm official VLSP.

### Nguyên tắc split

* dùng `random_seed` cố định, ví dụ `42`
* split theo group key khi có nguy cơ leakage:
  * `comprehension_short_answer`: group theo passage/source/context hash
  * MCQ: group theo normalized question/source id nếu có
* `train` và `val` không được có duplicate normalized prompt
* `shadow_eval` khó hơn `val` và dùng để quyết định có đáng chạy official VLSP hay không
* retention không gộp vào `val_all.jsonl`; dùng probe riêng

### File đầu ra

* `train_all.jsonl`
* `val_all.jsonl`
* `shadow_eval.jsonl`
* `instruction_probe.jsonl`
* `cloze_probe.jsonl`

### Vai trò

* `train_all.jsonl`: dữ liệu train QLoRA
* `val_all.jsonl`: chọn checkpoint và theo dõi overfit
* `shadow_eval.jsonl`: kiểm tra gần benchmark ở mốc lớn
* probes: guardrail chống quên instruction behavior và LM-style

---

## 2.6. Mixture policy

### Mục tiêu

Chọn cách trộn có chủ đích thay vì để bucket lớn áp đảo tự nhiên.

### Baseline mix nên thử

#### Natural mix

* dùng gần đúng tỷ lệ dữ liệu hiện có sau QC
* ưu điểm: ít can thiệp, phản ánh volume thật
* rủi ro: `instruction_retention` và `comprehension_short_answer` có thể lấn át MCQ

#### Task-balanced mix

* cân bằng hơn giữa 5 bucket
* ưu điểm: dễ đọc tác động từng task, phù hợp run thăm dò
* rủi ro: phải downsample bucket lớn, có thể bỏ bớt dữ liệu tốt

### Khuyến nghị vòng đầu

Dùng **task-balanced mix nhẹ** cho run thăm dò:

* giữ đủ tín hiệu từ 5 bucket
* không để retention chiếm quá nhiều loss budget
* không ép `comprehension_short_answer` thành MCQ, chỉ theo dõi gap format khi đánh giá official `comprehension`

Tỷ lệ cụ thể nên được ghi trong manifest/mix config, không hardcode trong plan nếu chưa chạy smoke test.

### Kết quả đầu ra

* `train_all.jsonl` có `metadata.task` đầy đủ để phân tích theo task
* mix report ghi số sample theo bucket trước/sau sampling
* có thể tái tạo lại cùng split/mix từ seed và config

---

## 2.7. Dev set và shadow eval

### Mục tiêu

Tạo đánh giá nội bộ đủ nhạy để không phải chạy VLSP hằng ngày.

### `val_all.jsonl`

* đại diện cho `exams_mcq`, `wiki_mcq`, `comprehension_short_answer`
* sạch, khó vừa phải, không duplicate với train
* không gộp retention probes vào metric chính
* dùng để theo dõi loss/eval theo task trong W&B

### `shadow_eval.jsonl`

* khó hơn `val_all.jsonl`
* MCQ ưu tiên câu dễ nhầm / distractor tốt
* `comprehension_short_answer` ưu tiên passage dài hơn, câu hỏi cần đọc kỹ hơn, answer span không quá hiển nhiên
* dùng trước các mốc official VLSP

### Retention probes

* `instruction_probe.jsonl`: kiểm tra model còn giữ được hành vi Instruct tiếng Việt không
* `cloze_probe.jsonl`: kiểm tra năng lực LM-style có tụt quá mạnh không
* probes chạy riêng, không gộp vào MCQ accuracy

---

## 2.8. Khi nào mới sinh thêm synthetic?

### Mục tiêu

Chỉ mở rộng dữ liệu khi biết rõ seed hiện tại thiếu gì.

### Điều kiện mở rộng

Chỉ sinh thêm bằng LLM sau khi:

* đã có baseline run trên seed-first mix
* biết task nào yếu qua dev/shadow/VLSP milestone
* có giả thuyết rõ, ví dụ thiếu MCQ distractor tốt, thiếu passage dài, hoặc thiếu câu hỏi suy luận
* có QC đủ mạnh cho dạng dữ liệu mới
* có API LLM hoặc model local đủ tốt để sinh và kiểm tra distractor nhất quán

### Không làm mặc định

* không sinh hàng loạt MCQ comprehension chỉ để khớp format VLSP
* không dùng rule-based distractor để thay thế LLM QC cho `comprehension_short_answer`
* không tăng instruction/chat tổng quát nếu retention probe đã ổn
* không mở rộng dữ liệu khi lỗi chính nằm ở train config hoặc eval parser

---

## 2.9. Tiêu chí hoàn thành Giai đoạn 2

Giai đoạn 2 chỉ được xem là xong khi đạt đủ các điều kiện sau:

1. Có manifest cho 5 seed exports hiện có.
2. Có QC/report snapshot cho từng bucket.
3. Có cross-dedup và leakage check ở mức đủ dùng cho vòng 1.
4. Có split deterministic:

   * `train_all.jsonl`
   * `val_all.jsonl`
   * `shadow_eval.jsonl`
   * `instruction_probe.jsonl`
   * `cloze_probe.jsonl`
5. Có mix report ghi rõ số sample theo task.
6. `metadata.task` được giữ nguyên để phân tích theo task.
7. Có note rõ về gap format:

   * train đọc hiểu = `comprehension_short_answer`
   * official VLSP `comprehension` = MCQ

---

## 2.10. Kết quả cuối của Giai đoạn 2

Khi kết thúc giai đoạn này, bạn phải có:

* **một bộ train chính thức** cho vòng fine-tune đầu tiên
* **một dev set nội bộ** để chọn checkpoint hằng ngày
* **một shadow eval** để kiểm tra chất lượng trước khi chạm tới official VLSP
* **hai retention probes** để theo dõi quên instruction/LM-style
* **manifest + QC/mix reports** để tái tạo dữ liệu

Đây là nền để các run sau chỉ cần mở rộng hoặc tinh chỉnh có kiểm soát, không phải đập đi làm lại pipeline dữ liệu.

---

## Giai đoạn 3 — Ổn định pipeline huấn luyện

### Mục tiêu

Làm cho toàn bộ pipeline hoạt động ổn định trước khi chạy run thật.

### Việc cần làm

* Dùng **Colab Free** để smoke test:

  * dependency
  * load model
  * login Hugging Face
  * login W&B
  * chạy thử train rất ngắn
  * save checkpoint thử
* Dùng **Kaggle** để chuẩn bị cho run chính:

  * xác minh tài nguyên T4/T4x2
  * xác minh train loop ổn định
  * xác minh log đầy đủ lên W&B
  * xác minh push checkpoint lên HF không lỗi

### Vai trò từng môi trường

* **Colab Free**: kiểm tra nhanh, sửa lỗi nhanh
* **Kaggle**: chạy run chính thức

### Kết quả đầu ra

* pipeline train không lỗi
* pipeline log không lỗi
* pipeline lưu checkpoint không lỗi

### Điều kiện hoàn thành

Bạn phải có 1 run ngắn hoàn chỉnh từ đầu đến cuối, dù chưa có giá trị về chất lượng.

---

## Giai đoạn 4 — Run thăm dò đầu tiên

### Mục tiêu

Chứng minh rằng stack QLoRA của bạn chạy được trên dữ liệu thật và sinh ra checkpoint hợp lệ.

### Việc cần làm

* Chạy **run nhỏ đầu tiên** trên Kaggle
* Dùng `train_all.jsonl` từ seed-first split, ưu tiên **task-balanced mix nhẹ**
* Giữ mục tiêu thăm dò, không nhắm tối ưu điểm số
* Theo dõi chặt:

  * train loss
  * tốc độ train
  * độ ổn định
  * output mẫu
  * khả năng resume / lưu checkpoint
* Sau run, đánh giá trên:

  * `val_all.jsonl` theo từng `metadata.task`
  * `shadow_eval.jsonl` nếu tín hiệu dev đủ tốt
  * một số bài test mẫu thủ công
  * `instruction_probe.jsonl`
  * `cloze_probe.jsonl`
* Chưa cần chạy official VLSP ngay nếu tín hiệu quá yếu

### Kết quả đầu ra

* checkpoint đầu tiên hợp lệ
* xác nhận mô hình học được từ data của bạn
* biết được pipeline hiện tại có bottleneck ở đâu

### Điều kiện hoàn thành

Bạn phải biết:

* run có học thật hay không
* lỗi nằm ở data, config hay runtime
* có nên scale lên run lớn hơn không

---

## Giai đoạn 5 — Run huấn luyện chính thức vòng 1

### Mục tiêu

Tạo ra phiên bản fine-tuned đầu tiên đủ nghiêm túc để so với baseline.

### Việc cần làm

* Tăng quy mô train so với run thăm dò
* Giữ dataset mix có chủ đích:

  * instruction-retention để giữ hành vi Instruct
  * MCQ để hỗ trợ `wikipediaqa_vi` và `exams_vi`
  * **`comprehension_short_answer`** để rèn đọc hiểu grounded (eval official `comprehension` trên VLSP vẫn là MCQ)
  * cloze / LM-retention để giảm suy giảm trên `lambada_vi`
* So sánh ít nhất 2 mix nếu run thăm dò đủ ổn:

  * natural mix
  * task-balanced mix nhẹ
* Giữ triết lý:

  * chưa tối ưu quá sớm
  * ưu tiên tính ổn định
  * mọi thay đổi phải giải thích được

### Đánh giá sau run

* so với baseline trên `val_all.jsonl` và `shadow_eval.jsonl`
* xem mô hình cải thiện ở nhóm nào
* xem có dấu hiệu quên instruction behavior hay không
* xem output có bị verbose, lệch format, hay hallucination tăng không
* bắt buộc chạy `instruction_probe.jsonl` và `cloze_probe.jsonl` sau mỗi run để kiểm tra retention

### Kết quả đầu ra

* phiên bản fine-tune chính thức đầu tiên
* bộ nhận xét đầu tiên về tác động của data mix

### Điều kiện hoàn thành

Run này phải trả lời được:

* hướng train hiện tại có đúng không
* data mix có đang kéo đúng năng lực mong muốn không

---

## Giai đoạn 6 — Đánh giá holdout lần 1 trên VLSP 2023

### Mục tiêu

Kiểm tra mô hình mới có cải thiện thực chất trên benchmark chính thức hay không.

### Việc cần làm

* Chạy official evaluation theo protocol gốc
* So sánh trực tiếp với baseline
* Phân tích riêng:

  * `lambada_vi`
  * `wikipediaqa_vi`
  * từng task con của `exams_vi`
  * `comprehension`

### Cách đọc kết quả

* Nếu MCQ tăng nhưng `lambada_vi` giảm, nghĩa là fine-tune đang thiên về instruction/QA hơn năng lực LM-style
* Nếu `comprehension` không tăng, khả năng dữ liệu đọc hiểu dài còn yếu hoặc **gap format** giữa train short answer và benchmark MCQ (cần đọc kết hợp với probe/shadow và few-shot inference)
* Nếu `exams_vi` tăng không đồng đều giữa các môn, dữ liệu kiến thức đang lệch miền

### Kết quả đầu ra

* báo cáo so sánh baseline vs finetuned
* danh sách lỗi ưu tiên để sửa ở vòng sau

### Điều kiện hoàn thành

Bạn phải xác định được **vấn đề lớn nhất của mô hình hiện tại là gì**:

* thiếu kiến thức
* yếu đọc hiểu dài
* yếu xử lý distractor MCQ
* hay mất cân bằng giữa chat behavior và benchmark behavior

---

## Giai đoạn 7 — Vòng cải tiến có kiểm soát

### Mục tiêu

Làm 1–2 vòng cải tiến dựa trên lỗi thực tế, không thay đổi tràn lan.

### Việc cần làm

Mỗi vòng chỉ nên thay đổi **một nhóm nguyên nhân chính**:

* cải thiện dữ liệu MCQ và/hoặc **`comprehension_short_answer`**
* bổ sung dữ liệu đọc hiểu dài
* chỉnh data mix
* chỉnh chiến lược sampling
* chỉnh mức độ giữ instruction behavior

### Nguyên tắc

* Không thay nhiều biến cùng lúc
* Mỗi run phải có giả thuyết rõ
* Mỗi run phải so được với run trước qua W&B

### Sau mỗi vòng

* so trên dev nội bộ
* bắt buộc chạy `instruction_probe.jsonl` và `cloze_probe.jsonl` sau mỗi run cải tiến
* chỉ chạy VLSP ở các mốc đủ đáng giá
* cập nhật kết luận rất rõ:

  * thay đổi gì
  * kỳ vọng gì
  * kết quả thực tế ra sao

### Kết quả đầu ra

* 1 hoặc 2 run cải tiến có căn cứ
* hiểu rõ yếu tố nào thật sự tạo tăng trưởng

### Điều kiện hoàn thành

Bạn phải có bằng chứng khá rõ về:

* thành phần dữ liệu nào hữu ích nhất
* hướng cải tiến nào không đáng tiếp tục
* retention probes không tụt ngoài ngưỡng chấp nhận nội bộ

---

## Giai đoạn 8 — Chốt model ứng viên cuối

### Mục tiêu

Chọn một checkpoint tốt nhất để làm ứng viên cuối cho project.

### Tiêu chí chọn

Không chọn chỉ dựa trên một điểm số duy nhất. Chọn theo tổng hợp:

* hiệu quả trên dev nội bộ
* hiệu quả trên VLSP holdout
* độ ổn định output
* độ nhất quán format
* khả năng tái hiện run
* mức độ dễ dùng về sau

### Việc cần làm

* so sánh 2–3 checkpoint tốt nhất
* xem lại log W&B
* xem lại kết quả VLSP
* kiểm tra mẫu output đại diện

### Kết quả đầu ra

* 1 checkpoint được chọn làm ứng viên cuối
* 1 checkpoint dự phòng nếu checkpoint chính có trade-off khó chấp nhận

### Điều kiện hoàn thành

Bạn phải giải thích được vì sao checkpoint này là lựa chọn cuối cùng.

---

## Giai đoạn 9 — Đóng project vòng 1

### Mục tiêu

Kết thúc vòng đầu tiên của project với kết quả có thể dùng lại và mở rộng.

### Việc cần làm

* lưu checkpoint cuối lên HF
* lưu artifact eval lên HF
* chốt toàn bộ thông tin run trên W&B
* tổng hợp bài học rút ra:

  * cái gì hiệu quả
  * cái gì không hiệu quả
  * giới hạn của T4/T4x2
  * giới hạn của data hiện tại
  * bước tiếp theo nếu có vòng 2

### Kết quả đầu ra

* model cuối
* lịch sử experiment rõ ràng
* toàn bộ kết quả benchmark được lưu giữ
* đủ căn cứ để bước sang vòng 2 nếu muốn

### Điều kiện hoàn thành

Một người khác nhìn vào project cũng phải hiểu được:

* bạn đã train gì
* train như thế nào
* model nào tốt nhất
* vì sao nó tốt nhất

---

# Nhịp triển khai đề xuất

## Pha 1 — Ổn định hệ thống

Ưu tiên:

* baseline
* dev nội bộ
* smoke test pipeline

## Pha 2 — Fine-tune vòng 1

Ưu tiên:

* run thăm dò
* run chính thức đầu tiên
* đánh giá holdout lần 1

## Pha 3 — Cải tiến có kiểm soát

Ưu tiên:

* 1–2 vòng sửa theo lỗi thật
* không chạy quá nhiều run vô hướng

## Pha 4 — Chốt model cuối

Ưu tiên:

* chọn checkpoint
* lưu trữ chuẩn
* tổng hợp kết luận

---

# Quy tắc vận hành trong suốt project

1. **Mọi quyết định chọn run phải đi qua W&B**
2. **Mọi checkpoint đáng giữ phải lên HF**
3. **Mọi thay đổi lớn phải có baseline để so**
4. **Mọi tuning hằng ngày phải dùng dev nội bộ**
5. **VLSP chỉ chạy ở các mốc quan trọng**
6. **Không tối ưu đồng thời quá nhiều biến trong một vòng**
7. **Ưu tiên run ít nhưng có giả thuyết rõ hơn là nhiều run nhưng không rút ra được gì**

---

# Dấu hiệu plan đang đi đúng hướng

* baseline đã được khóa
* dev nội bộ đủ nhạy để phân biệt run tốt và run xấu
* pipeline train ổn định trên Kaggle
* log W&B sạch và nhất quán
* checkpoint trên HF có thể truy vết lại
* mỗi vòng cải tiến đều có kết luận rõ
* khi chạy VLSP, bạn hiểu được vì sao điểm tăng hoặc giảm chứ không chỉ nhìn con số
