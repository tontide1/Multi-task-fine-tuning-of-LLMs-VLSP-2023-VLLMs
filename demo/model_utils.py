import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)

MODEL_NAME = "VietAI/gpt-neo-1.3B-vietnamese-news"
LOCAL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models", "vietai-gpt-neo-1.3b-vietnamese-news")


def load_model():
    """Load tokenizer and model from HuggingFace, cache locally."""
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

    # Try local-only first; fallback to remote if not cached
    local_only = os.path.exists(
        os.path.join(LOCAL_CACHE_DIR, "models--VietAI--gpt-neo-1.3B-vietnamese-news")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=LOCAL_CACHE_DIR,
        local_files_only=local_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=LOCAL_CACHE_DIR,
        local_files_only=local_only,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Ensure pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def _generate(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Shared generation logic."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    with torch.inference_mode():
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


def predict_mcq(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    question: str,
    choices: dict,
) -> str:
    """
    Predict the MCQ answer (A/B/C/D).
    choices: {"A": "...", "B": "...", "C": "...", "D": "..."}
    """
    missing = {"A", "B", "C", "D"} - set(choices.keys())
    if missing:
        raise ValueError(f"choices missing keys: {missing}")

    prompt = (
        f"Câu hỏi: {question}\n"
        f"A. {choices['A']}\n"
        f"B. {choices['B']}\n"
        f"C. {choices['C']}\n"
        f"D. {choices['D']}\n"
        f"Đáp án:"
    )
    result = _generate(tokenizer, model, prompt, max_new_tokens=1)
    result = result.upper()
    if result and result[0] in "ABCD":
        return result[0]
    raise ValueError(f"Model returned unexpected MCQ token: {result!r}")


def predict_next_word(tokenizer, model, text: str, num_tokens: int = 1) -> str:
    """Predict the next word(s) and return the continuation."""
    return _generate(tokenizer, model, text, max_new_tokens=num_tokens)
