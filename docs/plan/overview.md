# Plan for fine-tuning model unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit for VLSP 2023 Benchmarks VLLMs

* **QLoRA là mặc định**
* **HF Hub = model registry + eval archive**
* **W&B = source of truth cho experiment**
* **Kaggle = môi trường train chính**
* **Colab Free = môi trường debug / smoke test**
* **VLSP 2023 = holdout benchmark, không dùng để tuning hằng ngày**

Điểm cần giữ xuyên suốt là: **VLSP 2023 chỉ dùng để đo ở các mốc quan trọng**, vì benchmark này có protocol đánh giá riêng và không đồng nhất giữa các task: `lambada_vi` là next-word prediction, còn `wikipediaqa_vi`, `exams_vi`, `comprehension` là multiple-choice; riêng `exams_vi` còn được tách thành nhiều task con trong repo benchmark chính thức.

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

Xây một pipeline dữ liệu vòng đầu đủ sạch, đủ lớn, và đủ gần với 3 nhóm benchmark MCQ của VLSP 2023 để phục vụ fine-tune hằng ngày, trong khi vẫn giữ **VLSP 2023** là **holdout benchmark** chỉ dùng để đánh giá ở các mốc quan trọng.

### Phạm vi đã chốt

Bộ dữ liệu vòng đầu có 3 nhánh chính và 2 nhánh phụ:

* **`exams_mcq`**
* **`wiki_mcq`**
* **`comprehension_mcq`**

Ngoài 3 nhánh chính này, cần giữ thêm 2 nhánh phụ nhỏ để bảo toàn năng lực nền của model:

* **`instruction_retention`**
* **`cloze_lm_retention`**

### Không làm trong vòng đầu

* chưa build `lambada_cloze` theo quy mô full dataset
* không copy, paraphrase, hay dịch lại benchmark VLSP vào train/dev/shadow eval
* không mở rộng sang instruction/chat tổng quát ngoài phạm vi cần cho 3 nhánh MCQ và 2 nhánh retention nhỏ

---

## 2.1. Thu thập seed public

### Mục tiêu

Lấy một bộ seed public sạch để làm nền trước khi nở dữ liệu bằng LLM.

### Hướng thực hiện

Ưu tiên **Hugging Face** làm nguồn chính.

### 3 bucket dữ liệu cần tìm

* **MCQ học đường** → map vào `exams_mcq`
* **MCQ kiến thức tổng quát** → map vào `wiki_mcq`
* **reading comprehension có passage** → map vào `comprehension_mcq`

### Nguồn build 2 nhánh retention

* **`instruction_retention`**: lấy từ public instruction/chat data sạch trên Hugging Face, ưu tiên mẫu tiếng Việt tự nhiên hoặc mẫu có thể giữ format trả lời mà không cần mở rộng chat tổng quát.
* **`cloze_lm_retention`**: lấy từ public Vietnamese text corpus sạch hoặc nguồn text public có thể chuyển thành cloze/next-word samples; chỉ dùng subset nhỏ để giữ năng lực LM-style, không đụng benchmark VLSP.

### Quy mô mục tiêu

* lấy khoảng **5k–10k mẫu public sạch** làm seed

### Tiêu chí chọn seed

* có câu hỏi rõ ràng
* có 4 lựa chọn A/B/C/D hoặc chuyển đổi được về 4 lựa chọn
* có đáp án đúng rõ ràng
* nội dung không nhiễu, không lỗi format nặng
* phù hợp để Việt hóa hoặc đã là tiếng Việt tốt

### Kết quả đầu ra

* một seed set public đã được bucket hóa thành:

  * `exams_mcq`
  * `wiki_mcq`
  * `comprehension_mcq`
* 2 nhánh retention được chuẩn bị riêng và convert về schema nội bộ:

  * `instruction_retention`
  * `cloze_lm_retention`

Ghi chú riêng cho `wiki_mcq` seed file:

* `wiki_mcq_seed.jsonl`: seed thô đã clean schema
* `wiki_mcq_seed_clean.jsonl`: seed sạch đã whitelist subject, dùng cho stage-2 public seed
* `wiki_mcq_seed_dropped.jsonl`: archive audit/drop

---

## 2.2. Chuẩn hóa về một schema duy nhất

### Mục tiêu

Đảm bảo toàn bộ pipeline train/eval nội bộ dùng chung một format.

### Schema train đã chốt

```json
{"messages":[{"role":"user","content":"Câu hỏi: ...\nA. ...\nB. ...\nC. ...\nD. ..."},{"role":"assistant","content":"Đáp án: C"}],"metadata":{"task":"exams_mcq","source":"synthetic","difficulty":"medium"}}
```

### Schema retention đã chốt

```json
{"messages":[{"role":"user","content":"...instruction prompt..."},{"role":"assistant","content":"...instruction response..."}],"metadata":{"task":"instruction_retention","source":"public","difficulty":"medium"}}
```

```json
{"messages":[{"role":"user","content":"...cloze or next-word prompt..."},{"role":"assistant","content":"...target word or short completion..."}],"metadata":{"task":"cloze_lm_retention","source":"public","difficulty":"medium"}}
```

### Quy tắc schema

* mọi mẫu đều phải dùng `messages`
* `user` chứa:

  * câu hỏi
  * 4 lựa chọn
  * passage nếu là `comprehension_mcq`
* `assistant` chỉ trả lời ngắn gọn theo format:

  * `Đáp án: A/B/C/D`
* retention dùng cùng container `messages`, nhưng parser/eval tách riêng theo `task`
* `metadata` tối thiểu gồm:

  * `task`
  * `source`
  * `difficulty`

### Kết quả đầu ra

* toàn bộ seed public được convert về đúng schema trước khi sinh nở

---

## 2.3. Nở dữ liệu bằng LLM

### Mục tiêu

Mở rộng seed set thành bộ train dùng được cho vòng đầu.

### Công cụ đã chốt

* **ChatGPT / Gemini**: sinh câu hỏi mới theo đúng schema
* **DeepSeek**: giải thử / kiểm tra đáp án

### Cách sinh

Từ seed public, sinh thêm dữ liệu theo 3 hướng:

* viết lại câu hỏi cùng bản chất
* sinh thêm câu mới cùng chủ đề/cùng miền
* với `comprehension_mcq`, sinh thêm câu hỏi từ passage

### Ràng buộc khi sinh

Chỉ giữ mẫu nào:

* đúng format
* chỉ có 1 đáp án đúng
* distractor hợp lý
* tiếng Việt tự nhiên

### Quy mô mục tiêu sau khi nở

* từ seed **5k–10k mẫu**
* nở thành **20k–40k mẫu**

### Kết quả đầu ra

* một tập synthetic thô đủ lớn cho bước QC

---

## 2.4. QC dữ liệu

### Mục tiêu

Lọc mạnh để tránh đưa dữ liệu xấu vào train.

### QC rule-based

Loại các mẫu:

* thiếu A/B/C/D
* thiếu đáp án
* đáp án không thuộc A/B/C/D
* trùng lặp rõ ràng
* lỗi format
* lỗi encoding / xuống dòng / metadata

### QC bằng model

Dùng DeepSeek hoặc mô hình kiểm tra để:

* giải thử câu hỏi
* kiểm tra tính nhất quán đáp án
* phát hiện mẫu có nhiều hơn 1 đáp án đúng
* phát hiện distractor vô lý
* phát hiện câu tiếng Việt gượng, dịch máy, hoặc passage không phục vụ câu hỏi

### Tiêu chí giữ lại

Chỉ giữ mẫu nào đồng thời thỏa:

* đúng schema
* một đáp án đúng duy nhất
* distractor hợp lý
* tiếng Việt tự nhiên
* không quá giống benchmark VLSP

### Kết quả đầu ra

* một tập dữ liệu sạch, có thể dùng để build train/dev/shadow eval

---

## 2.5. Chống leakage với benchmark VLSP

### Mục tiêu

Giữ benchmark thật sự là holdout.

### Quy tắc cứng

* 4 bộ trong `benchmark/` chỉ để eval
* không đưa benchmark vào train
* không paraphrase benchmark vào train
* không dịch benchmark vào train
* không dùng benchmark làm nguồn để LLM sinh biến thể gần giống

### Kiểm tra leakage

* so khớp lexical gần đúng
* so khớp semantic với các mẫu synthetic nghi ngờ
* đặc biệt kiểm tra:

  * cùng fact pattern
  * cùng distractor pattern
  * cùng cấu trúc câu hỏi

### Kết quả đầu ra

* tập train/dev/shadow eval không đụng benchmark ở mức gây sai lệch đánh giá

---

## 2.6. Build bộ train vòng đầu

### Mục tiêu

Xuất bộ train chính thức cho vòng fine-tune đầu tiên.

### Tỷ lệ dữ liệu đã chốt

* **`exams_mcq`: 51%**
* **`wiki_mcq`: 17%**
* **`comprehension_mcq`: 17%**
* **`instruction_retention`: 10%**
* **`cloze_lm_retention`: 5%**

### Nguyên tắc

* sau QC mới trộn
* giữ đúng tổng tỷ lệ 100%
* không để một bucket synthetic áp đảo quá mạnh ngoài tỷ lệ mục tiêu

### File đầu ra

* `train_all.jsonl`

### Ý nghĩa

Đây là bộ dữ liệu dùng cho các run QLoRA vòng đầu trên Kaggle.

---

## 2.7. Build dev set nội bộ

### Mục tiêu

Tạo tập đánh giá nội bộ để chọn checkpoint trong quá trình train, thay vì đụng VLSP mỗi ngày.

### Nguyên tắc

* tách biệt hoàn toàn khỏi `train_all.jsonl`
* không chứa benchmark VLSP
* đủ đại diện cho 3 task MCQ
* ưu tiên mẫu sạch, khó vừa phải, có khả năng phân biệt checkpoint tốt/xấu
* retention không trộn vào `val_all.jsonl`, được theo dõi bằng probe riêng

### Nội dung dev set

* có thể gồm:

  * một phần seed public chưa dùng train
  * một phần synthetic chất lượng cao sau QC
* không cần giữ đúng 60/20/20 cứng như train, nhưng phải đủ cân bằng để đọc kết quả

### File đầu ra

* `val_all.jsonl`

### Retention probes nội bộ (nhỏ)

* `instruction_probe.jsonl`
* `cloze_probe.jsonl`

### Vai trò của probes

* theo dõi model có quên format/chat behavior hay không
* theo dõi model có tụt quá mạnh năng lực cloze/LM-style hay không
* chạy song song với dev nội bộ ở các mốc chọn checkpoint
* đo retention bằng metric riêng, không gộp vào MCQ accuracy

### Vai trò

* theo dõi trong lúc train
* chọn checkpoint
* phát hiện overfit
* hỗ trợ so sánh các run trên W&B

---

## 2.8. Build shadow eval

### Mục tiêu

Tạo một tập đánh giá nội bộ “gần benchmark” hơn dev set, nhưng vẫn không phải official VLSP.

### Vai trò của shadow eval

* mô phỏng một phần áp lực của benchmark thật
* dùng ở các mốc giữa các run lớn
* giúp quyết định có đáng chạy official VLSP hay chưa

### Nguyên tắc build

* không dùng bất kỳ mẫu nào từ benchmark VLSP
* độ khó cao hơn dev thông thường
* ưu tiên mẫu có distractor tốt, câu hỏi dễ nhầm, passage dài hơn, hoặc yêu cầu suy luận hơn
* vẫn giữ 3 nhánh:

  * `exams_mcq`
  * `wiki_mcq`
  * `comprehension_mcq`

### File đầu ra gợi ý

* `shadow_eval.jsonl`
  hoặc tách riêng:
* `shadow_exams.jsonl`
* `shadow_wiki.jsonl`
* `shadow_comprehension.jsonl`

### Vai trò trong workflow

* dev set dùng để theo dõi thường xuyên
* shadow eval dùng để kiểm tra trước các mốc lớn
* VLSP chỉ dùng ở mốc baseline / milestone / final
* retention probes (`instruction_probe`, `cloze_probe`) chạy riêng như guardrail, không gộp vào `shadow_eval.jsonl`

---

## 2.9. Tiêu chí hoàn thành Giai đoạn 2

Giai đoạn 2 chỉ được xem là xong khi đạt đủ các điều kiện sau:

1. Có **5k–10k mẫu public sạch** làm seed
2. Seed đã được bucket hóa thành:

   * `exams_mcq`
   * `wiki_mcq`
   * `comprehension_mcq`
3. Đã nở thành **20k–40k mẫu**
4. Đã QC mạnh theo rule-based + solver/model check
5. Đã kiểm soát leakage với benchmark VLSP
6. Đã xuất đủ:

   * `train_all.jsonl`
   * `val_all.jsonl`
   * `shadow_eval.jsonl` hoặc các file shadow riêng
7. `train_all.jsonl` giữ đúng tỷ lệ:

   * 51% `exams_mcq`
   * 17% `wiki_mcq`
   * 17% `comprehension_mcq`
   * 10% `instruction_retention`
   * 5% `cloze_lm_retention`
8. Đã xuất và dùng trong internal evaluation:

   * `instruction_probe.jsonl`
   * `cloze_probe.jsonl`

---

## 2.10. Kết quả cuối của Giai đoạn 2

Khi kết thúc giai đoạn này, bạn phải có:

* **một bộ train chính thức** cho vòng fine-tune đầu tiên
* **một dev set nội bộ** để chọn checkpoint hằng ngày
* **một shadow eval** để kiểm tra chất lượng trước khi chạm tới official VLSP
* và một quy trình dữ liệu đủ sạch để các run sau chỉ cần mở rộng hoặc tinh chỉnh, không phải đập đi làm lại

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
* Giữ mục tiêu thăm dò, không nhắm tối ưu điểm số
* Theo dõi chặt:

  * train loss
  * tốc độ train
  * độ ổn định
  * output mẫu
  * khả năng resume / lưu checkpoint
* Sau run, đánh giá trên:

  * dev nội bộ
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
  * reading comprehension để hỗ trợ `comprehension`
  * cloze / LM-retention để giảm suy giảm trên `lambada_vi`
* Giữ triết lý:

  * chưa tối ưu quá sớm
  * ưu tiên tính ổn định
  * mọi thay đổi phải giải thích được

### Đánh giá sau run

* so với baseline trên dev nội bộ
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
* Nếu `comprehension` không tăng, khả năng dữ liệu đọc hiểu dài còn yếu
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

* cải thiện dữ liệu MCQ
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
