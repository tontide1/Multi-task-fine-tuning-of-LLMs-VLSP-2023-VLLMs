# Streamlit Demo — VietAI GPT-Neo 1.3B Vietnamese News

## Cài đặt

```bash
conda activate nlp
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động tải model từ HuggingFace về thư mục `demo/models/` nếu chưa có.

## Chức năng

- **Trắc nghiệm (MCQ):** Nhập câu hỏi và 4 đáp án A/B/C/D, model sẽ chọn đáp án đúng.
- **Dự đoán từ tiếp theo:** Nhập một đoạn văn, model sẽ dự đoán từ tiếp theo (1–10 từ).
