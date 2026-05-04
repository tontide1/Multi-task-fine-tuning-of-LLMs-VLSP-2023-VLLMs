# Qwen2.5 Kaggle Fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a complete Jupyter Notebook (`.ipynb` format) intended to run on Kaggle with a Tesla T4 GPU to fine-tune `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit` using QLoRA.

**Architecture:** We will generate a single Python script that can be converted or copied directly into a Kaggle Notebook. The script will handle dependencies, heterogeneous JSONL data loading from Hugging Face, deterministic chat template formatting, balanced task mixing, model configuration, and SFTTrainer execution according to the 3-stage strategy outlined in the spec.

**Tech Stack:** Python, Unsloth, Hugging Face Transformers/Datasets/TRL, Kaggle.

---

### Task 1: Generate Notebook Code

**Files:**
- Create: `scripts/kaggle_qwen25_finetune.py` (We will write this as a robust Python script. The user can copy-paste its contents into Kaggle cells, or we can later convert it to `.ipynb` format if requested).

- [ ] **Step 1: Write setup and imports**
```python
# Kaggle setup (to be run in the first cell)
# !pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
# !pip install --no-deps xformers trl peft accelerate bitsandbytes datasets huggingface_hub

import os
import json
import torch
import random
from collections import defaultdict
from datasets import Dataset, load_dataset
from huggingface_hub import login
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Optional: Kaggle Secrets Login
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# login(token=user_secrets.get_secret("HF_TOKEN"))
```

- [ ] **Step 2: Write Data Loading & Normalization Logic**
Implement explicit JSONL downloading to handle heterogeneous metadata schemas without Arrow breaking.

```python
import requests

def download_file(url, local_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def load_jsonl_from_hf(repo_id, filename):
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    local_path = f"/tmp/{filename}"
    download_file(url, local_path)
    
    records = []
    with open(local_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
```

- [ ] **Step 3: Write Data Formatting & Task Mixing Logic**
```python
def prepare_training_mix(repo_id):
    raw_records = load_jsonl_from_hf(repo_id, "train_all.jsonl")
    
    # Bucket by task
    buckets = defaultdict(list)
    for record in raw_records:
        task = record.get("metadata", {}).get("task", "unknown")
        buckets[task].append(record)
        
    print("Original Train Counts:")
    for task, items in buckets.items():
        print(f"  {task}: {len(items)}")

    # Stage 2 Mix Strategy (Adjust counts based on actual dataset sizes)
    # The goal: exams/wiki kept fully, comprehension capped if too large, retention guardrails capped.
    random.seed(3407)
    
    mixed_records = []
    
    # Example logic (tune these numbers based on split_report.json):
    if "exams_mcq" in buckets: mixed_records.extend(buckets["exams_mcq"])
    if "wiki_mcq" in buckets: mixed_records.extend(buckets["wiki_mcq"])
    
    # Cap comprehension if it's huge
    comp_bucket = buckets.get("comprehension_short_answer", [])
    mixed_records.extend(comp_bucket[:min(len(comp_bucket), 8000)]) 
    
    # Guardrails
    inst_bucket = buckets.get("instruction_retention", [])
    mixed_records.extend(random.sample(inst_bucket, min(len(inst_bucket), 3000)))
    
    cloze_bucket = buckets.get("cloze_lm_retention", [])
    mixed_records.extend(random.sample(cloze_bucket, min(len(cloze_bucket), 1000)))

    random.shuffle(mixed_records)
    
    print("\nSampled Training Mix Counts:")
    final_counts = defaultdict(int)
    for r in mixed_records:
        final_counts[r.get("metadata", {}).get("task", "unknown")] += 1
    for task, count in final_counts.items():
        print(f"  {task}: {count} ({(count/len(mixed_records))*100:.1f}%)")
        
    return mixed_records

def format_dataset(records, tokenizer):
    texts = []
    tasks = []
    for r in records:
        # Qwen2.5 chat template application
        text = tokenizer.apply_chat_template(
            r["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
        tasks.append(r.get("metadata", {}).get("task", "unknown"))
    
    return Dataset.from_dict({"text": texts, "task": tasks})
```

- [ ] **Step 4: Write Model Configuration**
```python
def configure_model():
    max_seq_length = 4096 
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = torch.float16, # Or None
        load_in_4bit = True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
    )
    return model, tokenizer
```

- [ ] **Step 5: Write Validation Preparation**
```python
def prepare_validation(repo_id, tokenizer):
    raw_val = load_jsonl_from_hf(repo_id, "val_all.jsonl")
    return format_dataset(raw_val, tokenizer)
```

- [ ] **Step 6: Write Training Loop**
```python
def run_training(model, tokenizer, train_dataset, eval_dataset):
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = 4096,
        dataset_num_proc = 2,
        packing = False, # As per spec
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_ratio = 0.03,
            max_steps = 60, # Change to num_train_epochs = 1 for full run
            learning_rate = 2e-4,
            fp16 = True,
            bf16 = False,
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            save_steps = 500,
            save_total_limit = 2,
            evaluation_strategy="steps",
            eval_steps=500,
            report_to = "none", 
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()
    
    return model, tokenizer

# Main execution block
if __name__ == "__main__":
    REPO_ID = "tontide1/Dataset-for-fine-tuning-LLMS-VLSP-2023-benchmark"
    model, tokenizer = configure_model()
    
    raw_train = prepare_training_mix(REPO_ID)
    train_dataset = format_dataset(raw_train, tokenizer)
    eval_dataset = prepare_validation(REPO_ID, tokenizer)
    
    trained_model, tokenizer = run_training(model, tokenizer, train_dataset, eval_dataset)
    
    # Save local
    trained_model.save_pretrained("qwen25_3b_vlsp_lora")
    tokenizer.save_pretrained("qwen25_3b_vlsp_lora")
    
    # Push to hub (requires HF_TOKEN)
    # trained_model.push_to_hub("tontide1/Qwen2.5-3B-VLSP-Adapter")
```