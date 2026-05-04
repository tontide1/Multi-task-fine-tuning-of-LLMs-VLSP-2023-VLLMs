# Qwen2.5-3B Kaggle Fine-tuning Design Spec

## 1. Overview
This document outlines the architecture and configuration for fine-tuning the `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit` model on Kaggle using a single T4 GPU. The goal is to train the model on the VLSP 2023 reading comprehension and reasoning datasets stored in the Hugging Face dataset `tontide1/Dataset-for-fine-tuning-LLMS-VLSP-2023-benchmark`.

## 2. Environment Setup (Kaggle)
- **Hardware:** 1x NVIDIA Tesla T4 GPU (~15GB VRAM).
- **Core Libraries:**
  - `unsloth[colab-new]` from GitHub.
  - `xformers`, `trl`, `peft`, `accelerate`, `bitsandbytes`, `datasets`.
- **Authentication:** Kaggle Secrets will be used to securely access the `HF_TOKEN` for dataset downloading and model uploading.

## 3. Model & Tokenizer Initialization
- **Base Model:** `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit`
- **Settings:**
  - `max_seq_length = 4096`: Sufficient for long Vietnamese reading comprehension contexts.
  - `load_in_4bit = True`: Critical for fitting within T4 limits.

## 4. QLoRA Adapter Configuration
- **Rank (r):** 16 (Balances capacity for Vietnamese nuances and memory).
- **Alpha:** 16.
- **Target Modules:** `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (Comprehensive targeting for maximum reasoning adaptation).
- **Gradient Checkpointing:** `use_gradient_checkpointing = "unsloth"` (Saves ~30% VRAM).

## 5. Data Processing
- **Source:** `tontide1/Dataset-for-fine-tuning-LLMS-VLSP-2023-benchmark` (split: `train`).
- **Formatting:** Since the data is in OpenAI `messages` format, we will use Unsloth's `standardize_sharegpt` or `get_chat_template` mapping explicitly mapped to the `qwen-2.5` template. This ensures correct `<|im_start|>` and `<|im_end|>` token injection without manual string manipulation.

## 6. Training Configuration (SFTTrainer)
- **Batching:** `per_device_train_batch_size = 2`, `gradient_accumulation_steps = 4` (Effective batch size = 8).
- **Learning Rate:** `2e-4` with a `linear` scheduler and `warmup_steps = 5`.
- **Fault Tolerance:** 
  - `save_steps = 500` (or appropriate interval).
  - `save_total_limit = 2` to prevent disk overflow while ensuring recovery from Kaggle timeouts.
- **Precision:** `fp16 = True` (T4 does not support bfloat16 natively).

## 7. Export Strategy
- The resulting LoRA adapters will be pushed directly to the user's Hugging Face account (e.g., `tontide1/Qwen2.5-3B-VLSP-Adapter`) using `model.push_to_hub()`.
- Standard memory profiling scripts from Unsloth will be included to log VRAM usage before, during, and after training.
