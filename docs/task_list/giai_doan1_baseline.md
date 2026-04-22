# Giai đoạn 1: Khóa baseline và chuẩn đánh giá

## Mục tiêu ngày 1

Khóa cấu hình baseline và xác nhận pipeline eval chạy đúng cho toàn bộ task VLSP 2023 trước khi chạy full baseline.

## Checklist

- [ ] Chốt model baseline: `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit` (không fine-tune).
- [ ] Chốt config inference dùng chung: seed, temperature/top_p (hoặc greedy), max_new_tokens, batch size.
- [ ] Chốt môi trường chạy và dependency (python + package versions) để tái lập.
- [ ] Xác nhận đầy đủ 4 nhóm benchmark:
  - [ ] `lambada_vi` (next-word prediction)
  - [ ] `wikipediaqa_vi` (MCQ, `--num_fewshot 5`)
  - [ ] `exams_dialy_vi`, `exams_hoahoc_vi`, `exams_lichsu_vi`, `exams_sinhhoc_vi`, `exams_toan_vi`, `exams_vatly_vi`, `exams_van_vi` (đúng toàn bộ subtask, `--num_fewshot 5`)
  - [ ] `comprehension_vi` (MCQ)
- [ ] Khóa parser output cho từng nhóm task:
  - [ ] parser riêng cho `lambada_vi`
  - [ ] parser MCQ cho `wikipediaqa_vi`, `exams_*`, `comprehension_vi`
- [ ] Khóa rule metric theo 2 nhóm:
  - [ ] `lambada_vi`: next-word prediction theo `target_word`
  - [ ] MCQ: tính accuracy cho `wikipediaqa_vi`, `exams_*`, `comprehension_vi`
- [ ] Chạy smoke test nhỏ (mỗi task/subtask một ít mẫu) để kiểm tra:
  - [ ] load model OK
  - [ ] infer OK
  - [ ] parse đáp án OK
  - [ ] tính metric OK
- [ ] Ghi nhận lỗi kỹ thuật (nếu có) và fix ở mức tối thiểu.
- [ ] Chốt “baseline eval contract” v1: cùng 1 config, cùng 1 parser rule, cùng 1 quy tắc tính điểm, cùng task IDs, cùng few-shot setting cho các lần chạy sau.

## Deliverables cuối ngày

- [ ] `baseline_eval_config` (file cấu hình hoặc tài liệu tương đương)
- [ ] `smoke_eval_log_day1` (log chạy thử)
- [ ] `issues_day1.md` (lỗi + cách xử lý)
- [ ] `baseline_eval_contract_v1.md` (quy ước đánh giá đã khóa)
- [ ] `smoke_eval_matrix_day1.md` (pass/fail theo từng task/subtask)
- [ ] `baseline_protocol_lock.md` (task IDs, few-shot, parser rule, metric rule)

## Exit criteria ngày 1

- Pipeline eval chạy được end-to-end trên smoke set cho toàn bộ task.
- Không còn mơ hồ về protocol/parse/metric trước khi sang ngày 2 chạy full baseline.
