# Auto-generated from notebooks/qwen25_kaggle_finetune.ipynb
# Notebook magics/shell lines are commented for Python compatibility.

# %% [code] cell 0
# Notebook install cell (see notebook for executable shell commands).
# Use the following pinned setup in Kaggle/Colab:
# - transformers==4.56.2
# - trl==0.22.2 (no-deps)
# - unsloth / unsloth_zoo / bitsandbytes / accelerate / xformers / peft / triton
# - datasets==4.3.0, huggingface_hub>=0.34.0, hf_transfer

# %% [code] cell 1
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# %% [code] cell 2
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import math
import re
import time
import unicodedata

import torch
import unsloth
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

# Unsloth / TRL imports are kept here so the notebook reads top-to-bottom.
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

try:
    import wandb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wandb"])
    import wandb

# MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit"
MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit"

DATASET_REPO = "tontide1/Dataset-for-fine-tuning-LLMS-VLSP-2023-benchmark"
DATASET_REVISION = "main"
SEED = 3407
MAX_SEQ_LENGTH = 2048
RUN_MODE = "full"  # đổi sang "full" khi smoke run ổn

SMOKE_CAPS: dict[str, int] = {
    "comprehension_short_answer": 100,
    "exams_mcq": 100,
    "wiki_mcq": 100,
    "instruction_retention": 100,
    "cloze_lm_retention": 100,
}

FULL_CAPS: dict[str, int] = {
    "comprehension_short_answer": 14_000,
    "exams_mcq": 10_760,
    "wiki_mcq": 7_136,
    "instruction_retention": 8_000,
    "cloze_lm_retention": 4_000,
}

if Path("/kaggle/working").exists():
    BASE_DIR = Path("/kaggle/working")
else:
    BASE_DIR = Path.cwd()

ARTIFACT_DIR = BASE_DIR / "qwen25_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = ARTIFACT_DIR / "hf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = ARTIFACT_DIR / "lora_model"
REPORT_DIR = ARTIFACT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_JSONL_NAME = "train_all.jsonl"
VAL_JSONL_NAME = "val_all.jsonl"
SHADOW_JSONL_NAME = "shadow_eval.jsonl"
INSTRUCTION_PROBE_JSONL_NAME = "instruction_probe.jsonl"
CLOZE_PROBE_JSONL_NAME = "cloze_probe.jsonl"
SPLIT_REPORT_NAME = "split_report.json"

TOKEN_STAT_KEYS = {
    "token_length",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "n_tokens",
}


def get_kaggle_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(name)
        return value.strip() if value else None
    except Exception:
        return None


HF_TOKEN = os.getenv("HF_TOKEN") or get_kaggle_secret("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY") or get_kaggle_secret("WANDB_API_KEY")
if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

WANDB_PROJECT = "qwen25-vlsp-finetune"
WANDB_RUN_NAME = f"qwen25-{RUN_MODE}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
WANDB_TAGS = [RUN_MODE, "Qwen2.5-1.5B", "qlora", "vlsp"]
WANDB_RUN = None


@dataclass
class RunContext:
    wandb_run: Any | None = None
    instruction_probe_rows: list[dict[str, Any]] = field(default_factory=list)
    cloze_probe_rows: list[dict[str, Any]] = field(default_factory=list)


RUN_CONTEXT = RunContext()

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("MODEL_ID:", MODEL_ID)
print("DATASET_REPO:", DATASET_REPO)
print("DATASET_REVISION:", DATASET_REVISION)
print("RUN_MODE:", RUN_MODE)
print("MAX_SEQ_LENGTH:", MAX_SEQ_LENGTH)
print("ARTIFACT_DIR:", ARTIFACT_DIR)
print("HF_TOKEN available:", HF_TOKEN is not None)
print("WANDB_API_KEY available:", WANDB_API_KEY is not None)
print("WANDB_PROJECT:", WANDB_PROJECT)
print("WANDB_RUN_NAME:", WANDB_RUN_NAME)


def init_wandb_run(mix_report: dict[str, Any], length_report: dict[str, Any]) -> Any | None:
    if not WANDB_API_KEY:
        print("W&B disabled: missing WANDB_API_KEY in env or Kaggle Secrets.")
        return None

    wandb.login(key=WANDB_API_KEY, relogin=True)
    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        tags=WANDB_TAGS,
        dir=str(ARTIFACT_DIR),
        config={
            "model_id": MODEL_ID,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "seed": SEED,
            "run_mode": RUN_MODE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "caps": mix_report["caps"],
            "train_counts": mix_report["train_counts"],
            "val_counts": mix_report["val_counts"],
            "length_summary_train": length_report["summary_train"],
            "length_summary_val": length_report["summary_val"],
        },
    )
    wandb.define_metric("train/*")
    wandb.define_metric("eval/*")
    wandb.define_metric("post_train/*")
    print("W&B run:", run.url)
    return run


def log_wandb_json(path: Path, artifact_type: str, context: RunContext | None = None) -> None:
    wandb_run = context.wandb_run if context is not None and context.wandb_run is not None else WANDB_RUN
    if wandb_run is None or not path.exists():
        return
    artifact = wandb.Artifact(path.stem, type=artifact_type)
    artifact.add_file(str(path))
    wandb_run.log_artifact(artifact)

# %% [code] cell 3
def download_jsonl(filename: str) -> Path:
    """Download one file from the Hugging Face dataset repo."""
    path_str = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        filename=filename,
        cache_dir=str(CACHE_DIR),
        token=HF_TOKEN,
    )
    return Path(path_str)


def read_jsonl(path: Path, max_lines: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_lines is not None and idx >= max_lines:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("Each row must contain exactly 2 messages: user and assistant.")

    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    normalized: dict[str, Any] = {
        "messages": messages,
        "task": str(metadata.get("task", "unknown")),
        "metadata": metadata,
    }

    for key in TOKEN_STAT_KEYS:
        if key in row:
            normalized[key] = row[key]
        elif key in metadata:
            normalized[key] = metadata[key]

    for key in ("subject", "choices", "options", "answer", "gold_answer", "correct_answer", "label"):
        if key in row:
            normalized[key] = row[key]
        elif key in metadata:
            normalized[key] = metadata[key]

    return normalized


def group_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("task", "unknown"))].append(row)
    return buckets


def make_training_mix(rows: list[dict[str, Any]], caps: dict[str, int], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets = group_by_task(rows)

    mixed: list[dict[str, Any]] = []
    for task in sorted(caps):
        task_rows = list(buckets.get(task, []))
        rng.shuffle(task_rows)
        mixed.extend(task_rows[: min(len(task_rows), caps[task])])

    rng.shuffle(mixed)
    return mixed


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def count_tasks(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("task", "unknown")) for row in rows))

# %% [code] cell 4
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def messages_to_text(messages: list[dict[str, Any]]) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    eos = tokenizer.eos_token or ""
    if eos and not text.rstrip().endswith(eos):
        text = text + eos
    return text


def rows_to_dataset(rows: list[dict[str, Any]]) -> Dataset:
    texts: list[str] = []
    tasks: list[str] = []
    for row in rows:
        texts.append(messages_to_text(row["messages"]))
        tasks.append(str(row.get("task", "unknown")))
    return Dataset.from_list([{ "text": text, "task": task } for text, task in zip(texts, tasks)])


def token_length(text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    index = int(round((len(values) - 1) * pct))
    return values[max(0, min(index, len(values) - 1))]


def length_summary(rows: list[dict[str, Any]], threshold: int = MAX_SEQ_LENGTH) -> dict[str, Any]:
    tasks = group_by_task(rows)
    summary: dict[str, Any] = {}
    for task in sorted(tasks):
        lengths = [token_length(messages_to_text(row["messages"])) for row in tasks[task]]
        if not lengths:
            continue
        summary[task] = {
            "n": len(lengths),
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths),
            f"trunc>{threshold}": round(sum(length > threshold for length in lengths) / len(lengths), 6),
        }
    return summary


def write_length_report(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = {
        "generated_at": "manual-run",
        "model_id": MODEL_ID,
        "seed": SEED,
        "thresholds": [MAX_SEQ_LENGTH],
        "summary_train": length_summary(train_rows),
        "summary_val": length_summary(val_rows),
    }
    save_json(REPORT_DIR / "token_length_report.json", report)
    return report


def save_long_examples(rows: list[dict[str, Any]], threshold: int = MAX_SEQ_LENGTH) -> Path:
    out_path = REPORT_DIR / f"long_examples_over_{threshold}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            text = messages_to_text(row["messages"])
            if token_length(text) > threshold:
                f.write(json.dumps({"task": row.get("task"), "messages": row["messages"]}, ensure_ascii=False) + "\n")
    return out_path


def print_formatted_sample(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No rows to sample.")
        return
    sample = rows[0]
    sample_text = messages_to_text(sample["messages"])
    print("Task:", sample.get("task", "unknown"))
    print("Has assistant marker:", "<|im_start|>assistant\n" in sample_text)
    print(sample_text[:1200])


def split_eval_by_task(rows: list[dict[str, Any]]) -> dict[str, Dataset]:
    grouped = group_by_task(rows)
    eval_sets: dict[str, Dataset] = {}
    for task in sorted(grouped):
        eval_sets[task] = rows_to_dataset(grouped[task])
    return eval_sets

# %% [code] cell 5
train_path = download_jsonl(TRAIN_JSONL_NAME)
val_path = download_jsonl(VAL_JSONL_NAME)
shadow_path = download_jsonl(SHADOW_JSONL_NAME)
instruction_probe_path = download_jsonl(INSTRUCTION_PROBE_JSONL_NAME)
cloze_probe_path = download_jsonl(CLOZE_PROBE_JSONL_NAME)
split_report_path = download_jsonl(SPLIT_REPORT_NAME)

raw_train_rows = read_jsonl(train_path)
raw_val_rows = read_jsonl(val_path)
raw_shadow_rows = read_jsonl(shadow_path)
raw_instruction_probe_rows = read_jsonl(instruction_probe_path)
raw_cloze_probe_rows = read_jsonl(cloze_probe_path)
split_report = json.loads(split_report_path.read_text(encoding="utf-8"))

train_rows = [normalize_row(row) for row in raw_train_rows]
val_rows = [normalize_row(row) for row in raw_val_rows]
shadow_rows = [normalize_row(row) for row in raw_shadow_rows]
instruction_probe_rows = [normalize_row(row) for row in raw_instruction_probe_rows]
cloze_probe_rows = [normalize_row(row) for row in raw_cloze_probe_rows]
RUN_CONTEXT.instruction_probe_rows = instruction_probe_rows
RUN_CONTEXT.cloze_probe_rows = cloze_probe_rows

print("Train rows:", len(train_rows))
print("Val rows:", len(val_rows))
print("Shadow rows:", len(shadow_rows))
print("Instruction probe rows:", len(instruction_probe_rows))
print("Cloze probe rows:", len(cloze_probe_rows))
print("Train task counts:", count_tasks(train_rows))
print("Val task counts:", count_tasks(val_rows))
print("Split report keys:", sorted(split_report.keys()))

current_caps = FULL_CAPS if RUN_MODE == "full" else SMOKE_CAPS
train_mix_rows = make_training_mix(train_rows, current_caps, SEED)
train_mix_counts = count_tasks(train_mix_rows)
val_counts = count_tasks(val_rows)

mix_report = {
    "model_id": MODEL_ID,
    "dataset_repo": DATASET_REPO,
    "dataset_revision": DATASET_REVISION,
    "seed": SEED,
    "run_mode": RUN_MODE,
    "caps": current_caps,
    "train_counts": train_mix_counts,
    "val_counts": val_counts,
    "source_files": {
        "train": str(train_path),
        "val": str(val_path),
        "shadow": str(shadow_path),
        "instruction_probe": str(instruction_probe_path),
        "cloze_probe": str(cloze_probe_path),
        "split_report": str(split_report_path),
    },
}
save_json(REPORT_DIR / "mix_report.json", mix_report)

print("Mix counts:", train_mix_counts)
print("Current caps:", current_caps)

length_report = write_length_report(train_mix_rows, val_rows)
long_examples_path = save_long_examples(train_mix_rows)
print("Saved length report to:", REPORT_DIR / "token_length_report.json")
print("Saved long examples to:", long_examples_path)
print("Train length summary:", length_report["summary_train"].keys())
print("Val length summary:", length_report["summary_val"].keys())

WANDB_RUN = init_wandb_run(mix_report, length_report)
RUN_CONTEXT.wandb_run = WANDB_RUN
log_wandb_json(REPORT_DIR / "mix_report.json", "mix-report", RUN_CONTEXT)
log_wandb_json(REPORT_DIR / "token_length_report.json", "token-length-report", RUN_CONTEXT)

print_formatted_sample(train_mix_rows)

# %% [code] cell 6
def choose_dtype() -> torch.dtype | None:
    if not torch.cuda.is_available():
        return None
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" in gpu_name:
        return torch.float16
    return None


def build_data_collator(tokenizer: Any) -> tuple[Any, str]:
    response_template = "<|im_start|>assistant\n"
    template_ids = tokenizer(response_template, add_special_tokens=False)["input_ids"]
    if not template_ids:
        raise RuntimeError("Assistant response template produced empty token ids.")

    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def find_subsequence(haystack: list[int], needle: list[int]) -> int:
        if len(needle) > len(haystack):
            return -1
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i : i + len(needle)] == needle:
                return i
        return -1

    class CompletionOnlyCollator:
        def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            batch = base_collator(features)
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            for row_idx in range(input_ids.size(0)):
                tokens = input_ids[row_idx].tolist()
                start_idx = find_subsequence(tokens, template_ids)
                if start_idx < 0:
                    labels[row_idx, :] = -100
                    continue
                assistant_start = start_idx + len(template_ids)
                labels[row_idx, :assistant_start] = -100

            batch["labels"] = labels
            return batch

    return CompletionOnlyCollator(), "assistant_only_completion_custom"


model_dtype = choose_dtype()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=model_dtype,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    use_rslora=False,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

DATA_COLLATOR, loss_mode = build_data_collator(tokenizer)

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("model_dtype:", model_dtype)
print("loss_mode:", loss_mode)
print("trainable parameters:")
model.print_trainable_parameters()


def build_trainer(train_dataset: Dataset, eval_dataset: Dataset) -> SFTTrainer:
    smoke_run = RUN_MODE == "smoke"
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03 if not smoke_run else 0.0,
        warmup_steps=5 if smoke_run else 0,
        optim="adamw_8bit",
        weight_decay=0.01,
        fp16=torch.cuda.is_available() and not is_bfloat16_supported(),
        bf16=bool(torch.cuda.is_available() and is_bfloat16_supported()),
        logging_steps=1 if smoke_run else 10,
        save_steps=10 if smoke_run else 200,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=10 if smoke_run else 200,
        save_strategy="steps",
        max_steps=20 if smoke_run else -1,
        num_train_epochs=1 if smoke_run else 3,
        report_to="wandb" if WANDB_RUN is not None else "none",
        seed=SEED,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        average_tokens_across_devices=True,
        disable_tqdm=True,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "dataset_text_field": "text",
        "max_seq_length": MAX_SEQ_LENGTH,
        "dataset_num_proc": os.cpu_count() or 4,
        "packing": False,
        "args": args,
        "callbacks": [
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)
        ],
    }
    if DATA_COLLATOR is not None:
        trainer_kwargs["data_collator"] = DATA_COLLATOR

    return SFTTrainer(**trainer_kwargs)


def tokenize_eval_dataset(dataset: Dataset) -> Dataset:
    """Create input_ids for datasets evaluated after trainer initialization."""
    if "input_ids" in dataset.column_names:
        return dataset

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=list(dataset.column_names),
        desc="Tokenizing eval dataset",
    )


def prepare_eval_dataset(trainer: SFTTrainer, dataset: Dataset, name: str) -> Dataset:
    """Use TRL preparation when available, then fall back to simple tokenization."""
    if "input_ids" in dataset.column_names:
        return dataset

    try:
        processing_class = getattr(trainer, "processing_class", tokenizer)
        eval_packing = getattr(trainer.args, "eval_packing", None)
        packing = getattr(trainer.args, "packing", False) if eval_packing is None else eval_packing
        return trainer._prepare_dataset(dataset, processing_class, trainer.args, packing, None, name)
    except Exception as exc:
        print(f"Falling back to manual tokenization for {name}: {exc}")
        return tokenize_eval_dataset(dataset)


def run_eval_by_task(trainer: SFTTrainer, datasets_by_task: dict[str, Dataset]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for task, dataset in datasets_by_task.items():
        prepared_dataset = prepare_eval_dataset(trainer, dataset, f"eval_{task}")
        metrics = trainer.evaluate(eval_dataset=prepared_dataset, metric_key_prefix=f"eval_{task}")
        results[task] = metrics
    return results


def weighted_main_score(task_metrics: dict[str, Any]) -> float:
    losses: list[float] = []
    for task in ["exams_mcq", "wiki_mcq", "comprehension_short_answer"]:
        task_result = task_metrics.get(task, {})
        loss = task_result.get(f"eval_{task}_loss")
        if loss is not None:
            losses.append(float(loss))
    return sum(losses) / len(losses) if losses else float("inf")


def collect_probe_metrics(trainer: SFTTrainer, context: RunContext) -> dict[str, Any]:
    probes = {
        "instruction_probe": rows_to_dataset(context.instruction_probe_rows),
        "cloze_probe": rows_to_dataset(context.cloze_probe_rows),
    }
    probe_results: dict[str, Any] = {}
    for name, dataset in probes.items():
        prepared_dataset = prepare_eval_dataset(trainer, dataset, name)
        probe_results[name] = trainer.evaluate(eval_dataset=prepared_dataset, metric_key_prefix=name)
    return probe_results


def remove_notebook_progress_callback(trainer: SFTTrainer) -> None:
    """Avoid Kaggle notebook callback errors during post-training evaluate calls."""
    callbacks = trainer.callback_handler.callbacks
    kept_callbacks = [cb for cb in callbacks if cb.__class__.__name__ != "NotebookProgressCallback"]
    removed_count = len(callbacks) - len(kept_callbacks)
    trainer.callback_handler.callbacks = kept_callbacks
    if removed_count:
        print(f"Removed {removed_count} NotebookProgressCallback before post-training evaluation.")


def flatten_numeric_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_numeric_metrics(value, full_key))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[full_key] = float(value)
    return flat


def log_wandb_scalars(metrics: dict[str, Any], step: int | None = None) -> dict[str, float]:
    flat = flatten_numeric_metrics(metrics)
    if WANDB_RUN is not None and flat:
        wandb.log(flat, step=step)
        for key, value in flat.items():
            WANDB_RUN.summary[key] = value
    return flat


def get_message_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item.get("value") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"(\d+)\.(\d+)", lambda match: f"{match.group(1)}__dot__{match.group(2)}", text)
    text = re.sub(r"(?<!\w)-(\d)", lambda match: f"__neg__{match.group(1)}", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = text.replace("__dot__", ".").replace("__neg__", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_subject(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("subject", "exam_subject", "category", "topic", "domain"):
        value = row.get(key) if key in row else metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def extract_choice_options(row: dict[str, Any]) -> list[tuple[str, str]]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    candidates = [row.get("choices"), row.get("options"), metadata.get("choices"), metadata.get("options")]
    for candidate in candidates:
        options: list[tuple[str, str]] = []
        if isinstance(candidate, dict):
            keys = [str(key) for key in candidate.keys()]
            if all(key in candidate for key in ("A", "B", "C", "D")):
                keys = ["A", "B", "C", "D"]
            for key in keys:
                value = candidate[key]
                if isinstance(value, dict):
                    text = value.get("text") or value.get("content") or value.get("label") or value.get("value") or ""
                else:
                    text = value
                text = str(text).strip()
                if text:
                    options.append((str(key).strip().upper(), text))
        elif isinstance(candidate, list):
            for index, value in enumerate(candidate):
                label = chr(ord("A") + index)
                if isinstance(value, dict):
                    text = value.get("text") or value.get("content") or value.get("label") or value.get("value") or ""
                    label = str(value.get("label") or value.get("key") or value.get("option") or label)
                else:
                    text = value
                text = str(text).strip()
                if text:
                    options.append((str(label).strip().upper(), text))
        if options:
            return options
    return []


def extract_choice_label(text: str) -> str | None:
    patterns = [
        r"(?:đáp\s*án|answer|chọn|choose)\s*[:：]?\s*([ABCD])\b",
        r"\b([ABCD])\s*[).:：]\s",
        r"(?:^|\n)\s*([ABCD])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None


def canonical_mcq_answer(text: str, options: list[tuple[str, str]]) -> str | None:
    normalized = normalize_text(text)
    if not normalized:
        return None

    label = extract_choice_label(text)
    if label:
        return label

    for index, (_, option_text) in enumerate(options):
        if normalize_text(option_text) == normalized:
            return chr(ord("A") + index)
    return normalized


def is_valid_mcq_prediction(text: str, options: list[tuple[str, str]]) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    if extract_choice_label(text):
        return True
    if not options:
        return True
    return any(normalize_text(option_text) == normalized for _, option_text in options)


def lcs_length(left_tokens: list[str], right_tokens: list[str]) -> int:
    if not left_tokens or not right_tokens:
        return 0
    if len(left_tokens) < len(right_tokens):
        left_tokens, right_tokens = right_tokens, left_tokens
    prev = [0] * (len(right_tokens) + 1)
    for left_token in left_tokens:
        curr = [0] * (len(right_tokens) + 1)
        for j, right_token in enumerate(right_tokens, start=1):
            if left_token == right_token:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def token_f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum(min(pred_counts[token], gold_counts[token]) for token in pred_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def batched_generate_texts(trainer: SFTTrainer, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    if not prompts:
        return []

    device = next(trainer.model.parameters()).device
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    predictions: list[str] = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            ).to(device)
            prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            generated = trainer.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
            )
            for sequence, prompt_length in zip(generated, prompt_lengths):
                continuation = sequence[int(prompt_length):]
                predictions.append(tokenizer.decode(continuation, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = previous_padding_side
    return predictions


def evaluate_mcq_rows(trainer: SFTTrainer, rows: list[dict[str, Any]], batch_size: int = 8) -> dict[str, Any]:
    started_at = time.perf_counter()
    prompts = [tokenizer.apply_chat_template([row["messages"][0]], tokenize=False, add_generation_prompt=True) for row in rows]
    golds = [get_message_content(row["messages"][1]) for row in rows]
    predictions = batched_generate_texts(trainer, prompts, max_new_tokens=8, batch_size=batch_size)

    accuracies: list[float] = []
    normalized_accuracies: list[float] = []
    invalid_rates: list[float] = []
    choice_coverage: list[float] = []
    subject_scores: dict[str, list[float]] = defaultdict(list)

    for row, prediction, gold in zip(rows, predictions, golds):
        options = extract_choice_options(row)
        pred_norm = normalize_text(prediction)
        gold_norm = normalize_text(gold)
        pred_canonical = canonical_mcq_answer(prediction, options)
        gold_canonical = canonical_mcq_answer(gold, options)
        valid = is_valid_mcq_prediction(prediction, options)
        accuracy = float(pred_canonical == gold_canonical) if pred_canonical is not None and gold_canonical is not None else float(pred_norm == gold_norm)
        acc_norm = float(pred_norm == gold_norm)
        subject = extract_subject(row) or row.get("task", "unknown")
        subject_scores[subject].append(accuracy)

        accuracies.append(accuracy)
        normalized_accuracies.append(acc_norm)
        invalid_rates.append(0.0 if valid else 1.0)
        choice_coverage.append(1.0 if options else 0.0)

    subject_accuracy = {subject: sum(scores) / len(scores) for subject, scores in subject_scores.items() if scores}
    runtime_seconds = time.perf_counter() - started_at
    return {
        "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "acc_norm": sum(normalized_accuracies) / len(normalized_accuracies) if normalized_accuracies else 0.0,
        "macro_accuracy_by_subject": sum(subject_accuracy.values()) / len(subject_accuracy) if subject_accuracy else 0.0,
        "invalid_answer_rate": sum(invalid_rates) / len(invalid_rates) if invalid_rates else 0.0,
        "acc_norm_coverage": sum(choice_coverage) / len(choice_coverage) if choice_coverage else 0.0,
        "n": len(rows),
        "num_subjects": len(subject_accuracy),
        "subject_accuracy": subject_accuracy,
        "runtime_seconds": runtime_seconds,
    }


def evaluate_short_answer_rows(trainer: SFTTrainer, rows: list[dict[str, Any]], batch_size: int = 4) -> dict[str, Any]:
    started_at = time.perf_counter()
    prompts = [tokenizer.apply_chat_template([row["messages"][0]], tokenize=False, add_generation_prompt=True) for row in rows]
    golds = [get_message_content(row["messages"][1]) for row in rows]
    predictions = batched_generate_texts(trainer, prompts, max_new_tokens=32, batch_size=batch_size)

    exact_matches: list[float] = []
    token_f1s: list[float] = []
    rouge_ls: list[float] = []
    empty_rates: list[float] = []

    for prediction, gold in zip(predictions, golds):
        pred_norm = normalize_text(prediction)
        gold_norm = normalize_text(gold)
        exact_matches.append(1.0 if pred_norm == gold_norm else 0.0)
        token_f1s.append(token_f1_score(prediction, gold))
        rouge_ls.append(rouge_l_score(prediction, gold))
        empty_rates.append(1.0 if not pred_norm else 0.0)

    runtime_seconds = time.perf_counter() - started_at
    return {
        "exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
        "token_f1": sum(token_f1s) / len(token_f1s) if token_f1s else 0.0,
        "rouge_l": sum(rouge_ls) / len(rouge_ls) if rouge_ls else 0.0,
        "empty_answer_rate": sum(empty_rates) / len(empty_rates) if empty_rates else 0.0,
        "n": len(rows),
        "runtime_seconds": runtime_seconds,
    }


def build_post_train_metrics(
    trainer: SFTTrainer,
    val_rows: list[dict[str, Any]],
    shadow_metrics: dict[str, Any],
    probe_metrics: dict[str, Any],
    context: RunContext,
) -> dict[str, Any]:
    rows_by_task = group_by_task(val_rows)
    mcq_tasks = {task: rows for task, rows in rows_by_task.items() if task in {"exams_mcq", "wiki_mcq"}}
    short_answer_rows = rows_by_task.get("comprehension_short_answer", [])

    mcq_metrics: dict[str, Any] = {}
    mcq_examples = 0
    mcq_accuracy_total = 0.0
    mcq_acc_norm_total = 0.0
    mcq_invalid_total = 0.0
    mcq_coverage_total = 0.0
    mcq_runtime_total = 0.0
    subject_accuracy_buckets: dict[str, list[float]] = defaultdict(list)

    for task_name, task_rows in mcq_tasks.items():
        task_metrics = evaluate_mcq_rows(trainer, task_rows)
        mcq_metrics[task_name] = task_metrics
        mcq_examples += int(task_metrics["n"])
        mcq_accuracy_total += task_metrics["accuracy"] * task_metrics["n"]
        mcq_acc_norm_total += task_metrics["acc_norm"] * task_metrics["n"]
        mcq_invalid_total += task_metrics["invalid_answer_rate"] * task_metrics["n"]
        mcq_coverage_total += task_metrics["acc_norm_coverage"] * task_metrics["n"]
        mcq_runtime_total += task_metrics["runtime_seconds"]
        for subject, subject_score in task_metrics["subject_accuracy"].items():
            subject_accuracy_buckets[subject].append(subject_score)

    if mcq_examples:
        subject_accuracy = {subject: sum(scores) / len(scores) for subject, scores in subject_accuracy_buckets.items() if scores}
        mcq_metrics["overall"] = {
            "accuracy": mcq_accuracy_total / mcq_examples,
            "acc_norm": mcq_acc_norm_total / mcq_examples,
            "invalid_answer_rate": mcq_invalid_total / mcq_examples,
            "acc_norm_coverage": mcq_coverage_total / mcq_examples,
            "n": mcq_examples,
            "runtime_seconds": mcq_runtime_total,
        }
        mcq_metrics["macro_accuracy_by_subject"] = sum(subject_accuracy.values()) / len(subject_accuracy) if subject_accuracy else 0.0
        mcq_metrics["subject_accuracy"] = subject_accuracy
    else:
        mcq_metrics["overall"] = {
            "accuracy": 0.0,
            "acc_norm": 0.0,
            "invalid_answer_rate": 0.0,
            "acc_norm_coverage": 0.0,
            "n": 0,
            "runtime_seconds": 0.0,
        }
        mcq_metrics["macro_accuracy_by_subject"] = 0.0
        mcq_metrics["subject_accuracy"] = {}

    short_answer_metrics = evaluate_short_answer_rows(trainer, short_answer_rows) if short_answer_rows else {
        "exact_match": 0.0,
        "token_f1": 0.0,
        "rouge_l": 0.0,
        "empty_answer_rate": 0.0,
        "n": 0,
        "runtime_seconds": 0.0,
    }

    cloze_loss = float(probe_metrics.get("cloze_probe", {}).get("cloze_probe_loss", float("inf")))
    instruction_loss = float(probe_metrics.get("instruction_probe", {}).get("instruction_probe_loss", float("inf")))
    retention_metrics = {
        "cloze_perplexity": math.exp(min(cloze_loss, 20.0)) if math.isfinite(cloze_loss) else float("inf"),
        "instruction_probe_loss": instruction_loss,
        "cloze_probe_loss": cloze_loss,
    }

    runtime_metrics = {
        "mcq_generation_seconds": mcq_metrics["overall"]["runtime_seconds"],
        "short_answer_generation_seconds": short_answer_metrics["runtime_seconds"],
        "shadow_eval_seconds": float(shadow_metrics.get("shadow_runtime", 0.0)),
        "probe_eval_seconds": float(probe_metrics.get("instruction_probe", {}).get("instruction_probe_runtime", 0.0))
        + float(probe_metrics.get("cloze_probe", {}).get("cloze_probe_runtime", 0.0)),
    }

    return {
        "exams_mcq": mcq_metrics.get("exams_mcq", {}),
        "wiki_mcq": mcq_metrics.get("wiki_mcq", {}),
        "mcq_overall": mcq_metrics.get("overall", {}),
        "mcq_macro_accuracy_by_subject": mcq_metrics.get("macro_accuracy_by_subject", 0.0),
        "mcq_subject_accuracy": mcq_metrics.get("subject_accuracy", {}),
        "comprehension_short_answer": short_answer_metrics,
        "retention": retention_metrics,
        "runtime": runtime_metrics,
    }

# %% [code] cell 7
train_dataset = rows_to_dataset(train_mix_rows)
val_dataset = rows_to_dataset(val_rows)
eval_by_task = split_eval_by_task(val_rows)
shadow_dataset = rows_to_dataset(shadow_rows)

trainer = build_trainer(train_dataset, val_dataset)
last_checkpoint = None
if OUTPUT_DIR.exists():
    checkpoint_candidates = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if checkpoint_candidates:
        last_checkpoint = str(checkpoint_candidates[-1])

train_output = trainer.train(resume_from_checkpoint=last_checkpoint)
remove_notebook_progress_callback(trainer)

trainer.model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

aggregate_metrics = trainer.evaluate(metric_key_prefix="eval_all")
task_metrics = run_eval_by_task(trainer, eval_by_task)
prepared_shadow_dataset = prepare_eval_dataset(trainer, shadow_dataset, "shadow")
shadow_metrics = trainer.evaluate(eval_dataset=prepared_shadow_dataset, metric_key_prefix="shadow")
probe_metrics = collect_probe_metrics(trainer, RUN_CONTEXT)
FastLanguageModel.for_inference(trainer.model)
post_train_metrics = build_post_train_metrics(trainer, val_rows, shadow_metrics, probe_metrics, RUN_CONTEXT)
main_score = weighted_main_score(task_metrics)
selected_checkpoint_path = trainer.state.best_model_checkpoint or str(OUTPUT_DIR)
selection_rationale = "Trainer loaded the best checkpoint by eval_loss with early stopping enabled; main-task, probe, and generation metrics were computed after training."


def generate_sample(row: dict[str, Any], max_new_tokens: int = 128) -> str:
    prompt_text = tokenizer.apply_chat_template(
        [row["messages"][0]],
        tokenize=False,
        add_generation_prompt=True,
    )
    device = next(trainer.model.parameters()).device
    inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
    outputs = trainer.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


final_train_log = next((record for record in reversed(trainer.state.log_history) if "loss" in record), {})
final_train_health = {
    "train/loss": float(train_output.metrics.get("train_loss", final_train_log.get("loss", 0.0))),
    "train/runtime": float(train_output.metrics.get("train_runtime", 0.0)),
    "train/samples_per_second": float(train_output.metrics.get("train_samples_per_second", 0.0)),
    "train/steps_per_second": float(train_output.metrics.get("train_steps_per_second", 0.0)),
    "train/learning_rate": float(final_train_log.get("learning_rate", 0.0)),
    "train/grad_norm": float(final_train_log.get("grad_norm", 0.0)),
    "eval/loss": float(aggregate_metrics.get("eval_all_loss", 0.0)),
    "eval/runtime": float(aggregate_metrics.get("eval_all_runtime", 0.0)),
    "eval/samples_per_second": float(aggregate_metrics.get("eval_all_samples_per_second", 0.0)),
    "eval/steps_per_second": float(aggregate_metrics.get("eval_all_steps_per_second", 0.0)),
}
post_train_wandb_metrics = flatten_numeric_metrics(post_train_metrics, prefix="post_train")

print("Selected checkpoint:", selected_checkpoint_path)
print("Main score:", main_score)
print("Selection rationale:", selection_rationale)
print("Aggregate metrics:", aggregate_metrics)
print("Task metrics:", task_metrics)
print("Shadow metrics:", shadow_metrics)
print("Probe metrics:", probe_metrics)
print("Post-train metrics:", post_train_metrics)

sample_outputs = []
for row in val_rows[:2]:
    sample_outputs.append(generate_sample(row))

for idx, output in enumerate(sample_outputs, start=1):
    print(f"\n=== Sample {idx} ===")
    print(output[:1500])

run_report = {
    "model_id": MODEL_ID,
    "dataset_repo": DATASET_REPO,
    "dataset_revision": DATASET_REVISION,
    "seed": SEED,
    "run_mode": RUN_MODE,
    "loss_mode": loss_mode,
    "selected_checkpoint_path": selected_checkpoint_path,
    "selection_rationale": selection_rationale,
    "main_score": main_score,
    "train_metrics": getattr(train_output, "metrics", {}),
    "aggregate_metrics": aggregate_metrics,
    "task_metrics": task_metrics,
    "shadow_metrics": shadow_metrics,
    "probe_metrics": probe_metrics,
    "post_train_metrics": post_train_metrics,
    "sample_outputs": sample_outputs,
}
eval_report_path = REPORT_DIR / "eval_report.json"
post_train_metrics_path = REPORT_DIR / "post_train_metrics.json"
save_json(eval_report_path, run_report)
save_json(post_train_metrics_path, post_train_metrics)

if WANDB_RUN is not None:
    wandb.log(final_train_health, step=trainer.state.global_step)
    wandb.log(post_train_wandb_metrics, step=trainer.state.global_step)
    for key, value in {**final_train_health, **post_train_wandb_metrics}.items():
        WANDB_RUN.summary[key] = value
    WANDB_RUN.summary["selected_checkpoint_path"] = selected_checkpoint_path
    WANDB_RUN.summary["selection_rationale"] = selection_rationale
    WANDB_RUN.summary["loss_mode"] = loss_mode
    WANDB_RUN.summary["main_score"] = main_score
    WANDB_RUN.summary["post_train/exams_mcq/accuracy"] = float(post_train_metrics["mcq_overall"]["accuracy"])
    WANDB_RUN.summary["post_train/wiki_mcq/accuracy"] = float(post_train_metrics["wiki_mcq"]["accuracy"])
    WANDB_RUN.summary["post_train/comprehension_short_answer/token_f1"] = float(post_train_metrics["comprehension_short_answer"]["token_f1"])
    WANDB_RUN.summary["post_train/retention/cloze_perplexity"] = float(post_train_metrics["retention"]["cloze_perplexity"])
    log_wandb_json(eval_report_path, "eval-report", RUN_CONTEXT)
    log_wandb_json(post_train_metrics_path, "post-train-metrics", RUN_CONTEXT)

PUSH_TO_HUB = (RUN_MODE == "full") and bool(os.getenv("PUSH_TO_HUB", "true").lower() == "true")
HF_REPO_ID = "tontide1/Qwen2.5-1.5B-VLSP-Adapter"
if PUSH_TO_HUB and HF_TOKEN:
    trainer.model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
else:
    print("Push skipped. Set PUSH_TO_HUB=True and provide HF_TOKEN when you are ready to upload, or keep RUN_MODE=smoke.")

if WANDB_RUN is not None:
    wandb.finish()

print("Done.")

