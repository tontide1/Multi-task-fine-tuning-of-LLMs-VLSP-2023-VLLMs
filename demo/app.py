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
    st.header("📝 Trả lờit câu hỏi trắc nghiệm")
    with st.form("mcq_form"):
        question = st.text_area("Câu hỏi", height=100)
        col1, col2 = st.columns(2)
        with col1:
            choice_a = st.text_input("Đáp án A")
            choice_b = st.text_input("Đáp án B")
        with col2:
            choice_c = st.text_input("Đáp án C")
            choice_d = st.text_input("Đáp án D")
        submitted = st.form_submit_button("Trả lờit")

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
                continuation = predict_next_word(tokenizer, model, text_input, max_new_tokens=num_tokens)
            st.markdown("#### Kết quả")
            full_text = text_input + continuation
            st.markdown(
                f'<div class="answer-box"><span class="big-font">{full_text}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Từ được dự đoán thêm:** `{continuation}`")

st.markdown("---")
st.caption("Powered by VietAI/gpt-neo-1.3B-vietnamese-news via HuggingFace Transformers")
