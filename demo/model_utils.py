import os
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)

MODEL_NAME = "vlsp-2023-vllm/hoa-7b"
LOCAL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models", "vlsp-2023-vllm-hoa-7b")


def load_model() -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load tokenizer and model from HuggingFace, cache locally."""
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

    # Try local-only first; fallback to remote if not cached
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=LOCAL_CACHE_DIR,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=LOCAL_CACHE_DIR,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    except (OSError, ValueError):
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

    # Ensure pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    return tokenizer, model


def _generate(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Shared generation logic."""
    max_length = getattr(model.config, "max_position_embeddings", None)
    tokenizer_kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
    if max_length is not None:
        tokenizer_kwargs["max_length"] = max_length

    inputs = tokenizer(prompt, **tokenizer_kwargs)
    inputs = inputs.to(model.device)

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
    choices: dict[str, str],
) -> str:
    """
    Predict the MCQ answer (A/B/C/D).
    choices: {"A": "...", "B": "...", "C": "...", "D": "..."}
    """
    # Normalize keys to uppercase
    normalized_choices = {k.upper(): v for k, v in choices.items()}
    missing = {"A", "B", "C", "D"} - set(normalized_choices.keys())
    if missing:
        raise ValueError(f"choices missing keys: {missing}")
    unexpected = set(normalized_choices.keys()) - {"A", "B", "C", "D"}
    if unexpected:
        raise ValueError(f"choices has unexpected keys: {unexpected}")

    prompt = (
        f"Câu hỏi: {question}\n"
        f"A. {normalized_choices['A']}\n"
        f"B. {normalized_choices['B']}\n"
        f"C. {normalized_choices['C']}\n"
        f"D. {normalized_choices['D']}\n"
        f"Đáp án:"
    )
    result = _generate(tokenizer, model, prompt, max_new_tokens=5)
    match = re.search(r"[ABCD]", result.upper())
    if match:
        return match.group(0)
    raise ValueError(f"Model returned unexpected MCQ token: {result!r}")


def predict_next_word(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    text: str,
    max_new_tokens: int = 1,
) -> str:
    """Predict the next word(s) and return the continuation."""
    return _generate(tokenizer, model, text, max_new_tokens=max_new_tokens)
