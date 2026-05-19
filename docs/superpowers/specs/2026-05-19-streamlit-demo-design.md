# Spec: Streamlit Demo App — VietAI/gpt-neo-1.3B-vietnamese-news

## 1. Overview

Build a lightweight Streamlit web application to demonstrate inference with the `VietAI/gpt-neo-1.3B-vietnamese-news` model. The app supports two distinct interaction modes and is fully localized in Vietnamese with basic custom styling.

## 2. Goals

- Provide an interactive demo for the Vietnamese GPT-Neo model.
- Support **MCQ answering** (structured A/B/C/D input) and **next-word prediction**.
- Cache the model locally to avoid repeated downloads.
- Keep the UI clean, intuitive, and in Vietnamese.

## 3. Modes

### 3.1 MCQ Mode

- **Input**: Structured form with separate fields for:
  - Câu hỏi (question text)
  - Đáp án A
  - Đáp án B
  - Đáp án C
  - Đáp án D
- **Output**: Model predicts a single letter `A`, `B`, `C`, or `D`.
- **Prompt Format**:
  ```
  Câu hỏi: {question}
  A. {choice_a}
  B. {choice_b}
  C. {choice_c}
  D. {choice_d}
  Đáp án:
  ```
- **Generation config**: `max_new_tokens=1`, constrain to `[A, B, C, D]` if possible (or parse first token).

### 3.2 Next-Word Prediction Mode

- **Input**: Free-text paragraph.
- **Output**: Predicted next token(s), appended to the input text.
- **Controls**: Slider for `num_tokens` (default 1, range 1–10).
- **Prompt Format**: Raw text as-is.
- **Generation config**: `max_new_tokens=num_tokens`.

## 4. UI/UX Design

- **Language**: 100% Vietnamese (labels, buttons, messages, errors).
- **Layout**:
  - Sidebar: Mode selector (`MCQ` / `Dự đoán từ tiếp theo`).
  - Main panel: Dynamic form based on selected mode.
  - Result area: Highlighted answer box with a spinner during inference.
- **Styling**:
  - Custom CSS via `st.markdown(..., unsafe_allow_html=True)` to improve default Streamlit aesthetics (font size, button colors, padding).
  - No external CSS/JS files needed; keep it self-contained.

## 5. Architecture

```
demo/
├── app.py           # Streamlit entry point; UI + session state
├── model_utils.py   # Model loading, caching, tokenization, inference
├── requirements.txt # Dependencies
└── README.md        # Run instructions
```

### 5.1 `app.py`

- **Responsibilities**:
  - Render UI elements.
  - Manage `st.session_state` for model instance and cache status.
  - Call `model_utils` for inference.
  - Display results and handle loading states.
- **Key Flow**:
  1. On startup, call `model_utils.load_model()` (cached).
  2. Show a loading spinner until the model is ready.
  3. Render sidebar mode selector.
  4. Render the appropriate form.
  5. On submit, call `model_utils.predict_mcq()` or `model_utils.predict_next_word()`.
  6. Display result.

### 5.2 `model_utils.py`

- **Responsibilities**:
  - Load tokenizer and model from HuggingFace.
  - Save to local cache directory (e.g., `models/vietai-gpt-neo-1.3b-vietnamese-news/`).
  - Check if model already exists locally before downloading.
  - Provide `predict_mcq(prompt: str) -> str`.
  - Provide `predict_next_word(prompt: str, num_tokens: int) -> str`.
- **Caching Strategy**:
  - Use `transformers` built-in cache dir (`cache_dir` parameter) or a local `models/` folder.
  - Wrap `load_model()` with `@st.cache_resource` in `app.py` to prevent reloading on every interaction.

## 6. Data Flow

```
User Input (UI)
  → app.py formats prompt
  → model_utils.tokenize(prompt)
  → model.generate()
  → model_utils.decode(output)
  → app.py displays result
```

## 7. Error Handling

- **Model not found / download fails**: Show Vietnamese error message; suggest checking internet connection.
- **GPU not available**: Automatically fall back to CPU with a warning banner.
- **Empty input**: Disable submit button or show validation message.
- **Generation timeout / OOM**: Catch exceptions, show friendly error, suggest shorter input.

## 8. Dependencies

```
streamlit>=1.28
torch>=2.0
transformers>=4.35
```

## 9. Testing Strategy

- **Manual tests**:
  - Run app locally (`streamlit run demo/app.py`).
  - Test MCQ with sample Vietnamese history question.
  - Test next-word with sample paragraph from user prompt.
  - Verify model caching works (restart app, observe no re-download).
  - Verify UI renders correctly on both modes.

## 10. Out of Scope

- Fine-tuning or model training.
- Multi-user concurrency / authentication.
- Deployment to cloud (local demo only).
- Support for models other than `VietAI/gpt-neo-1.3B-vietnamese-news`.
