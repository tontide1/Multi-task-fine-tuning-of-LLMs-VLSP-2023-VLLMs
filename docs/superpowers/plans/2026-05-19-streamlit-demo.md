# Streamlit Demo App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit web app that demos the `VietAI/gpt-neo-1.3B-vietnamese-news` model with MCQ answering and next-word prediction modes.

**Architecture:** Single-page Streamlit app with a sidebar mode selector. Model loading and inference are encapsulated in `model_utils.py` and cached via `@st.cache_resource` in `app.py`. UI is fully localized in Vietnamese with inline custom CSS.

**Tech Stack:** Python 3.11, Streamlit, PyTorch, Transformers (HuggingFace)

---

## File Structure

```
demo/
├── app.py           # Streamlit entry point; UI + session state
├── model_utils.py   # Model loading, caching, tokenization, inference
├── requirements.txt # Dependencies
└── README.md        # Run instructions
```

---

## Task 1: Install Streamlit

**Files:**
- Modify: `environment.yml` (add streamlit) — optional, keep requirements.txt as primary
- Create: `demo/requirements.txt`

- [ ] **Step 1: Create requirements.txt**

  ```bash
  mkdir -p /home/tontide1/coding/nlp_project/Multi-task-fine-tuning-of-LLMs-VLSP-2023-VLLMs/demo
  cat > /home/tontide1/coding/nlp_project/Multi-task-fine-tuning-of-LLMs-VLSP-2023-VLLMs/demo/requirements.txt << 'REQEOF'
  streamlit>=1.28.0
  torch>=2.0.0
  transformers>=4.35.0
  REQEOF
  ```

- [ ] **Step 2: Install streamlit into the conda environment**

  Run: `conda run -n nlp pip install streamlit>=1.28.0`
  Expected: Installation completes successfully, `streamlit --version` prints a version >= 1.28.

- [ ] **Step 3: Verify installation**

  Run: `conda run -n nlp streamlit --version`
  Expected: Output like `Streamlit, version 1.28.x` (or higher).

- [ ] **Step 4: Commit**

  ```bash
  git add demo/requirements.txt
  git commit -m "chore: add streamlit demo requirements"
  ```

---

## Task 2: Write model_utils.py

**Files:**
- Create: `demo/model_utils.py`

- [ ] **Step 1: Write the module**

  Create `demo/model_utils.py` with the following content:

  ```python
  import os
  import torch
  from transformers import AutoTokenizer, AutoModelForCausalLM

  MODEL_NAME = "VietAI/gpt-neo-1.3B-vietnamese-news"
  LOCAL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models", "vietai-gpt-neo-1.3b-vietnamese-news")


  def load_model():
      """Load tokenizer and model from HuggingFace, cache locally."""
      os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

      tokenizer = AutoTokenizer.from_pretrained(
          MODEL_NAME,
          cache_dir=LOCAL_CACHE_DIR,
          local_files_only=False,
      )
      model = AutoModelForCausalLM.from_pretrained(
          MODEL_NAME,
          cache_dir=LOCAL_CACHE_DIR,
          local_files_only=False,
          torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
          device_map="auto" if torch.cuda.is_available() else None,
      )
      if not torch.cuda.is_available():
          model = model.to("cpu")

      # Ensure pad_token exists
      if tokenizer.pad_token is None:
          tokenizer.pad_token = tokenizer.eos_token

      return tokenizer, model


  def _generate(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
      """Shared generation logic."""
      inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
      if torch.cuda.is_available():
          inputs = {k: v.to("cuda") for k, v in inputs.items()}
      else:
          inputs = {k: v.to("cpu") for k, v in inputs.items()}

      with torch.no_grad():
          outputs = model.generate(
              **inputs,
              max_new_tokens=max_new_tokens,
              do_sample=False,
              pad_token_id=tokenizer.pad_token_id,
          )

      # Decode only the newly generated tokens
      new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
      result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
      return result


  def predict_mcq(tokenizer, model, question: str, choices: dict) -> str:
      """
      Predict the MCQ answer (A/B/C/D).
      choices: {"A": "...", "B": "...", "C": "...", "D": "..."}
      """
      prompt = (
          f"Câu hỏi: {question}\n"
          f"A. {choices['A']}\n"
          f"B. {choices['B']}\n"
          f"C. {choices['C']}\n"
          f"D. {choices['D']}\n"
          f"Đáp án:"
      )
      result = _generate(tokenizer, model, prompt, max_new_tokens=1)
      # Force uppercase and take first char if it looks like a letter
      result = result.upper()
      if result and result[0] in "ABCD":
          return result[0]
      return result  # fallback to raw token if unexpected


  def predict_next_word(tokenizer, model, text: str, num_tokens: int = 1) -> str:
      """Predict the next word(s) and return the continuation."""
      return _generate(tokenizer, model, text, max_new_tokens=num_tokens)
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add demo/model_utils.py
  git commit -m "feat: add model loading and inference utilities"
  ```

---

## Task 3: Write app.py

**Files:**
- Create: `demo/app.py`

- [ ] **Step 1: Write the Streamlit app**

  Create `demo/app.py` with the following content:

  ```python
  import streamlit as st
  from model_utils import load_model, predict_mcq, predict_next_word

  # Page config
  st.set_page_config(
      page_title="Demo VietAI GPT-Neo 1.3B",
      page_icon="🇻🇳",
      layout="centered",
  )

  # Custom CSS
  st.markdown(
      """
      <style>
      .big-font {
          font-size: 20px !important;
          font-weight: 600;
      }
      .answer-box {
          background-color: #f0f2f6;
          padding: 1rem;
          border-radius: 0.5rem;
          border-left: 5px solid #ff4b4b;
      }
      </style>
      """,
      unsafe_allow_html=True,
  )

  @st.cache_resource(show_spinner=False)
  def _cached_load_model():
      return load_model()

  # Load model with spinner
  if "model_ready" not in st.session_state:
      st.session_state.model_ready = False

  if not st.session_state.model_ready:
      with st.spinner("Đang tải mô hình, vui lòng đợi..."):
          tokenizer, model = _cached_load_model()
      st.session_state.tokenizer = tokenizer
      st.session_state.model = model
      st.session_state.model_ready = True
      st.rerun()

  tokenizer = st.session_state.tokenizer
  model = st.session_state.model

  # Sidebar
  st.sidebar.title("Chế độ")
  mode = st.sidebar.radio(
      "Chọn chức năng:",
      ["Trắc nghiệm (MCQ)", "Dự đoán từ tiếp theo"],
  )

  st.title("🇻🇳 Demo VietAI GPT-Neo 1.3B Vietnamese News")
  st.markdown("---")

  if mode == "Trắc nghiệm (MCQ)":
      st.header("📝 Trả lời câu hỏi trắc nghiệm")
      with st.form("mcq_form"):
          question = st.text_area("Câu hỏi", height=100)
          col1, col2 = st.columns(2)
          with col1:
              choice_a = st.text_input("Đáp án A")
              choice_b = st.text_input("Đáp án B")
          with col2:
              choice_c = st.text_input("Đáp án C")
              choice_d = st.text_input("Đáp án D")
          submitted = st.form_submit_button("Trả lời")

      if submitted:
          if not question or not all([choice_a, choice_b, choice_c, choice_d]):
              st.warning("Vui lòng nhập đầy đủ câu hỏi và 4 đáp án.")
          else:
              with st.spinner("Mô hình đang suy nghĩ..."):
                  choices = {
                      "A": choice_a,
                      "B": choice_b,
                      "C": choice_c,
                      "D": choice_d,
                  }
                  answer = predict_mcq(tokenizer, model, question, choices)
              st.markdown("#### Kết quả")
              st.markdown(
                  f'<div class="answer-box"><span class="big-font">Đáp án: {answer}</span></div>',
                  unsafe_allow_html=True,
              )

  else:
      st.header("✍️ Dự đoán từ tiếp theo")
      text_input = st.text_area("Nhập đoạn văn", height=150)
      num_tokens = st.slider("Số từ cần dự đoán", min_value=1, max_value=10, value=1)

      if st.button("Dự đoán"):
          if not text_input.strip():
              st.warning("Vui lòng nhập đoạn văn.")
          else:
              with st.spinner("Mô hình đang suy nghĩ..."):
                  continuation = predict_next_word(tokenizer, model, text_input, num_tokens)
              st.markdown("#### Kết quả")
              full_text = text_input + continuation
              st.markdown(
                  f'<div class="answer-box"><span class="big-font">{full_text}</span></div>',
                  unsafe_allow_html=True,
              )
              st.markdown(f"**Từ được dự đoán thêm:** `{continuation}`")

  st.markdown("---")
  st.caption("Powered by VietAI/gpt-neo-1.3B-vietnamese-news via HuggingFace Transformers")
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add demo/app.py
  git commit -m "feat: add Streamlit demo app with MCQ and next-word modes"
  ```

---

## Task 4: Write README.md

**Files:**
- Create: `demo/README.md`

- [ ] **Step 1: Write instructions**

  ```bash
  cat > /home/tontide1/coding/nlp_project/Multi-task-fine-tuning-of-LLMs-VLSP-2023-VLLMs/demo/README.md << 'READMEEOF'
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
  READMEEOF
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add demo/README.md
  git commit -m "docs: add demo README with run instructions"
  ```

---

## Task 5: Manual Testing

**Files:**
- Run: `demo/app.py`

- [ ] **Step 1: Launch the app**

  Run: `cd /home/tontide1/coding/nlp_project/Multi-task-fine-tuning-of-LLMs-VLSP-2023-VLLMs/demo && conda run -n nlp streamlit run app.py`
  Expected: Streamlit starts, shows "Đang tải mô hình...", then UI appears. First run will download the model (~5GB) so it may take several minutes depending on connection.

- [ ] **Step 2: Test MCQ mode**

  In the app:
  - Select "Trắc nghiệm (MCQ)" in sidebar.
  - Enter a sample Vietnamese question and 4 choices.
  - Click "Trả lời".
  - Expected: A single letter A/B/C/D appears in the answer box.

- [ ] **Step 3: Test next-word mode**

  In the app:
  - Select "Dự đoán từ tiếp theo" in sidebar.
  - Paste the sample paragraph: `Ngày 17 tháng 9 năm 1939, sức kháng cự của Ba Lan bị quân Đức bẻ gãy...`
  - Set slider to 1.
  - Click "Dự đoán".
  - Expected: The full text with the predicted next token appended (e.g., continuing with "máu" or similar).

- [ ] **Step 4: Verify caching**

  - Stop the app (Ctrl+C).
  - Re-run `streamlit run app.py`.
  - Expected: Model loads instantly (no re-download) because it is cached in `demo/models/` and via `@st.cache_resource`.

- [ ] **Step 5: Final commit**

  ```bash
  git add demo/
  git commit -m "feat: complete streamlit demo app"
  ```

---

## Self-Review Checklist

1. **Spec coverage:**
   - MCQ structured form (A/B/C/D) → Task 3
   - Next-word prediction with slider → Task 3
   - Local model caching → Task 2 + Task 5 Step 4
   - Vietnamese UI + custom CSS → Task 3
   - Error handling (empty input warnings) → Task 3
   - CPU fallback → Task 2 (`device_map="auto"` + `torch_dtype` logic)

2. **Placeholder scan:**
   - No TBD/TODO/fill-in-details found.
   - Every task has exact file paths, code blocks, and commands.

3. **Type consistency:**
   - `predict_mcq` signature matches call in `app.py`.
   - `predict_next_word` signature matches call in `app.py`.
   - `_cached_load_model` returns `(tokenizer, model)` consistent with `load_model()`.

**Plan complete and saved to `docs/superpowers/plans/2026-05-19-streamlit-demo.md`.**

---

## Execution Options

**Plan complete. Two execution options:**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach would you like?**
