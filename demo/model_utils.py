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
