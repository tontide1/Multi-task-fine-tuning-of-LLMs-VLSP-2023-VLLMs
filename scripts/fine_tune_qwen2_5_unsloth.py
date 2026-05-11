#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import logging as hf_logging

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
hf_logging.set_verbosity_error()

# Attempt to import specialized libraries
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTConfig, SFTTrainer
except ImportError:
    print("Error: unsloth or trl not found. Please install them using requirements_kaggle.txt.")
    sys.exit(1)

try:
    import wandb
except ImportError:
    print("Warning: wandb not found. W&B logging will be disabled.")
    wandb = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Warning: rouge_score not found. Evaluation metrics may be limited.")
    rouge_scorer = None

# Global Token Stat Keys
TOKEN_STAT_KEYS = {
    "token_length",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "n_tokens",
}

# Special Token IDs for Qwen2.5
IM_START_ID = 151644
ASSISTANT_ID = 77091
NEWLINE_ID = 198
IM_END_ID = 151645


def get_kaggle_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(name)
        return value.strip() if value else None
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 for VLSP 2023")
    parser.add_argument(
        "--run-mode", type=str, default="smoke", choices=["smoke", "full"]
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--input-dir",
        type=str,
        default="seed_exports/splits",
        help="Path to local dataset directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="qwen25_artifacts", help="Base directory for artifacts"
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--wandb", type=str, default="yes", choices=["yes", "no"], help="Enable W&B logging"
    )
    parser.add_argument(
        "--save-model-hf",
        type=str,
        default="no",
        choices=["yes", "no"],
        help="Push model to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Repo ID to push to (e.g. username/repo)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default="tontide1/Dataset-for-fine-tuning-LLMS-VLSP-2023-benchmark",
    )
    parser.add_argument("--dataset-revision", type=str, default="main")
    return parser.parse_args()


def get_data_path(filename: str, args, hf_token: str | None, cache_dir: Path) -> Path:
    """Get path to a data file, checking local input-dir first."""
    local_path = Path(args.input_dir) / filename
    if local_path.exists():
        print(f"Using local file: {local_path}")
        return local_path

    print(f"File {filename} not found in {args.input_dir}. Downloading from HF...")
    path_str = hf_hub_download(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        revision=args.dataset_revision,
        filename=filename,
        cache_dir=str(cache_dir),
        token=hf_token,
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

    return normalized


def group_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("task", "unknown"))].append(row)
    return buckets


def make_training_mix(
    rows: list[dict[str, Any]], caps: dict[str, int], seed: int
) -> list[dict[str, Any]]:
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


def messages_to_text(messages: list[dict[str, Any]], tokenizer) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    eos = tokenizer.eos_token or ""
    if eos and not text.rstrip().endswith(eos):
        text = text + eos
    return text


def token_length(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    index = int(round((len(values) - 1) * pct))
    return values[max(0, min(index, len(values) - 1))]


def length_summary(rows: list[dict[str, Any]], tokenizer, threshold: int) -> dict[str, Any]:
    tasks = group_by_task(rows)
    summary: dict[str, Any] = {}
    for task in sorted(tasks):
        lengths = [
            token_length(messages_to_text(row["messages"], tokenizer), tokenizer)
            for row in tasks[task]
        ]
        if not lengths:
            continue
        summary[task] = {
            "n": len(lengths),
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths),
            f"trunc>{threshold}": round(
                sum(length > threshold for length in lengths) / len(lengths), 6
            ),
        }
    return summary


def preprocess_for_training(
    rows: list[dict[str, Any]],
    tokenizer,
    max_seq_length: int,
) -> Dataset:
    all_input_ids: list[list[int]] = []
    all_attention_masks: list[list[int]] = []
    all_assistant_masks: list[list[int]] = []

    for row in rows:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=max_seq_length
        )
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        amask = [0] * len(ids)
        i = 0
        while i < len(ids) - 2:
            if (
                ids[i] == IM_START_ID
                and ids[i + 1] == ASSISTANT_ID
                and ids[i + 2] == NEWLINE_ID
            ):
                j = i + 3
                while j < len(ids):
                    if ids[j] == IM_END_ID:
                        amask[j] = 1
                        if j + 1 < len(ids) and ids[j + 1] == NEWLINE_ID:
                            amask[j + 1] = 1
                            j += 2
                        else:
                            j += 1
                        break
                    else:
                        amask[j] = 1
                    j += 1
                i = j
            else:
                i += 1

        all_input_ids.append(ids)
        all_attention_masks.append(mask)
        all_assistant_masks.append(amask)

    return Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "assistant_masks": all_assistant_masks,
        }
    )


def calculate_exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0


def calculate_token_f1(pred: str, ref: str) -> float:
    pred_tokens = pred.strip().lower().split()
    ref_tokens = ref.strip().lower().split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens).intersection(set(ref_tokens))
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * (precision * recall) / (precision + recall)


def extract_mcq_answer(text: str) -> str | None:
    m = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def batch_generate(
    rows: list[dict[str, Any]],
    model,
    tokenizer,
    max_new_tokens: int = 64,
    batch_size: int = 1,
) -> list[str]:
    outputs: list[str] = []
    model.eval()
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i : i + batch_size]
        texts: list[str] = []
        for row in batch_rows:
            prompt_text = tokenizer.apply_chat_template(
                [row["messages"][0]],
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(prompt_text)
        device = next(model.parameters()).device
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            decoded = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
            )
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = decoded[:, input_length:]
        for text in tokenizer.batch_decode(generated_tokens, skip_special_tokens=True):
            outputs.append(text.strip())
    return outputs


def compute_mcq_metrics(rows: list[dict[str, Any]], outputs: list[str], task: str) -> dict[str, Any]:
    correct = 0
    invalid = 0
    subject_correct: dict[str, int] = {}
    subject_total: dict[str, int] = {}

    for row, output in zip(rows, outputs):
        raw_label = row["messages"][1]["content"].strip()
        label = extract_mcq_answer(raw_label) or raw_label
        pred = extract_mcq_answer(output)
        subject = str(row.get("task", "unknown"))
        metadata = row.get("metadata", {})
        if isinstance(metadata, dict) and "subject" in metadata:
            subject = str(metadata["subject"])
        subject_total[subject] = subject_total.get(subject, 0) + 1

        if pred is None:
            invalid += 1
        elif pred == label:
            correct += 1
            subject_correct[subject] = subject_correct.get(subject, 0) + 1

    total = len(rows)
    accuracy = correct / total if total else 0
    acc_norm = correct / (total - invalid) if (total - invalid) else 0
    invalid_rate = invalid / total if total else 0

    result: dict[str, Any] = {
        f"{task}_accuracy": accuracy,
        f"{task}_acc_norm": acc_norm,
        f"{task}_invalid_answer_rate": invalid_rate,
        f"{task}_invalid_count": invalid,
        f"{task}_total": total,
    }

    if task == "exams_mcq" and subject_total:
        for subj in sorted(subject_total):
            s_acc = subject_correct.get(subj, 0) / subject_total[subj]
            result[f"{task}_macro_accuracy_by_subject/{subj}"] = s_acc
        result[f"{task}_macro_accuracy_by_subject/_mean"] = sum(
            subject_correct.get(s, 0) / subject_total[s] for s in subject_total
        ) / len(subject_total)

    return result


def compute_short_answer_metrics(
    rows: list[dict[str, Any]], outputs: list[str], scorer, task: str = "comprehension_short_answer"
) -> dict[str, Any]:
    empty = 0
    em_scores: list[float] = []
    f1_scores: list[float] = []
    rouge_l_scores: list[float] = []

    for row, output in zip(rows, outputs):
        answer = row["messages"][1]["content"].strip()
        pred = output.strip()

        if not pred or pred.lower() in ("null", "none", "n/a", "-"):
            empty += 1

        em_scores.append(calculate_exact_match(pred, answer))
        f1_scores.append(calculate_token_f1(pred, answer))
        if scorer:
            scores = scorer.score(answer, pred)
            rouge_l_scores.append(scores["rougeL"].fmeasure)
        else:
            rouge_l_scores.append(0.0)

    total = len(rows)
    return {
        f"{task}_exact_match": sum(em_scores) / len(em_scores) if em_scores else 0,
        f"{task}_token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        f"{task}_rouge_l": sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0,
        f"{task}_empty_answer_rate": empty / total if total else 0,
        f"{task}_empty_count": empty,
        f"{task}_total": total,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup Paths
    if Path("/kaggle/working").exists():
        base_dir = Path("/kaggle/working")
    else:
        base_dir = Path.cwd()

    artifact_dir = base_dir / args.output_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = artifact_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir = artifact_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_model_dir = artifact_dir / "lora_model"

    # Secrets
    hf_token = os.getenv("HF_TOKEN") or get_kaggle_secret("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY") or get_kaggle_secret("WANDB_API_KEY")

    if args.wandb == "yes" and wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.login(key=wandb_api_key, relogin=True)
    else:
        wandb_api_key = None
        print("W&B logging disabled.")

    # Caps
    smoke_caps = {
        "comprehension_short_answer": 10,
        "exams_mcq": 10,
        "wiki_mcq": 10,
        "instruction_retention": 10,
        "cloze_lm_retention": 10,
    }
    full_caps = {
        "comprehension_short_answer": 10_000,
        "exams_mcq": 50_000,
        "wiki_mcq": 50_000,
        "instruction_retention": 8_000,
        "cloze_lm_retention": 4_000,
    }
    current_caps = full_caps if args.run_mode == "full" else smoke_caps

    # Load Data
    data_files = {
        "train": "train_all.jsonl",
        "val": "val_all.jsonl",
        "shadow": "shadow_eval.jsonl",
        "instruction_probe": "instruction_probe.jsonl",
        "cloze_probe": "cloze_probe.jsonl",
        "split_report": "split_report.json",
    }
    data_paths = {
        k: get_data_path(v, args, hf_token, cache_dir) for k, v in data_files.items()
    }

    train_rows = [normalize_row(r) for r in read_jsonl(data_paths["train"])]
    val_rows = [normalize_row(r) for r in read_jsonl(data_paths["val"])]
    shadow_rows = [normalize_row(r) for r in read_jsonl(data_paths["shadow"])]
    instruction_probe_rows = [normalize_row(r) for r in read_jsonl(data_paths["instruction_probe"])]
    cloze_probe_rows = [normalize_row(r) for r in read_jsonl(data_paths["cloze_probe"])]

    print(f"Loaded {len(train_rows)} train, {len(val_rows)} val rows.")

    train_mix_rows = make_training_mix(train_rows, current_caps, args.seed)
    train_mix_counts = count_tasks(train_mix_rows)
    val_counts = count_tasks(val_rows)

    # Tokenizer & Model
    model_dtype = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "T4" in gpu_name:
            model_dtype = torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        dtype=model_dtype,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reports
    length_rep = {
        "summary_train": length_summary(train_mix_rows, tokenizer, args.max_seq_length),
        "summary_val": length_summary(val_rows, tokenizer, args.max_seq_length),
    }
    save_json(report_dir / "token_length_report.json", length_rep)

    mix_report = {
        "run_mode": args.run_mode,
        "caps": current_caps,
        "train_counts": train_mix_counts,
        "val_counts": val_counts,
    }
    save_json(report_dir / "mix_report.json", mix_report)

    # W&B Init
    wandb_run = None
    if args.wandb == "yes" and wandb_api_key:
        run_name = f"qwen25-{args.run_mode}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project="qwen25-vlsp-finetune",
            name=run_name,
            tags=[args.run_mode, "qwen2.5-1.5b", "qlora", "vlsp"],
            dir=str(artifact_dir),
            config={**vars(args), **mix_report},
        )

    # Datasets
    train_dataset = preprocess_for_training(train_mix_rows, tokenizer, args.max_seq_length)
    val_dataset = preprocess_for_training(val_rows, tokenizer, args.max_seq_length)
    shadow_dataset = preprocess_for_training(shadow_rows, tokenizer, args.max_seq_length)
    instruction_probe_dataset = preprocess_for_training(
        instruction_probe_rows, tokenizer, args.max_seq_length
    )
    cloze_probe_dataset = preprocess_for_training(cloze_probe_rows, tokenizer, args.max_seq_length)
    eval_by_task = {
        task: preprocess_for_training(rows, tokenizer, args.max_seq_length)
        for task, rows in group_by_task(val_rows).items()
    }

    # Training
    smoke_run = args.run_mode == "smoke"
    sft_config = SFTConfig(
        output_dir=str(output_model_dir),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=5 if smoke_run else (int(0.03 * 1000)),
        optim="adamw_8bit",
        weight_decay=0.01,
        fp16=torch.cuda.is_available() and not is_bfloat16_supported(),
        bf16=bool(torch.cuda.is_available() and is_bfloat16_supported()),
        logging_steps=1 if smoke_run else 10,
        save_steps=10 if smoke_run else 1000,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=10 if smoke_run else 1000,
        save_strategy="steps",
        max_steps=20 if smoke_run else -1,
        num_train_epochs=1,
        report_to="wandb" if wandb_run else "none",
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        disable_tqdm=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        max_seq_length=args.max_seq_length,
        dataset_num_proc=1,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
    )

    trainer.train()

    # Evaluation
    FastLanguageModel.for_inference(model)
    task_metrics: dict[str, Any] = {}
    for task, dataset in eval_by_task.items():
        metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=f"eval_{task}")
        task_metrics[task] = metrics

    # Shadow & Probe metrics
    shadow_metrics = trainer.evaluate(eval_dataset=shadow_dataset, metric_key_prefix="shadow")
    instruction_probe_metrics = trainer.evaluate(
        eval_dataset=instruction_probe_dataset, metric_key_prefix="instruction_probe"
    )
    cloze_probe_metrics = trainer.evaluate(
        eval_dataset=cloze_probe_dataset, metric_key_prefix="cloze_probe"
    )

    # Generation Eval
    scorer = None
    if rouge_scorer:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    generation_results: dict[str, Any] = {}
    for task in ["exams_mcq", "wiki_mcq", "comprehension_short_answer"]:
        task_rows = [r for r in val_rows if r.get("task") == task]
        if not task_rows:
            continue
        print(f"Generating for {task}...")
        outputs = batch_generate(
            task_rows, model, tokenizer, max_new_tokens=(32 if "mcq" in task else 64)
        )
        if "mcq" in task:
            generation_results[task] = compute_mcq_metrics(task_rows, outputs, task)
        else:
            generation_results[task] = compute_short_answer_metrics(task_rows, outputs, scorer, task)

    # Save final report
    eval_report = {
        "task_metrics": task_metrics,
        "shadow_metrics": shadow_metrics,
        "probe_metrics": {
            "instruction": instruction_probe_metrics,
            "cloze": cloze_probe_metrics,
        },
        "generation_metrics": generation_results,
    }
    save_json(report_dir / "eval_report.json", eval_report)

    # W&B Log Artifacts
    if wandb_run:
        type_map = {
            "mix_report.json": "mix-report",
            "token_length_report.json": "token-length-report",
            "eval_report.json": "eval-report",
        }
        for f in ["mix_report.json", "token_length_report.json", "eval_report.json"]:
            p = report_dir / f
            if p.exists():
                artifact_type = type_map.get(f, "report")
                artifact = wandb.Artifact(p.stem, type=artifact_type)
                artifact.add_file(str(p))
                wandb_run.log_artifact(artifact)
        wandb.finish()

    # Save to HF Hub
    if args.save_model_hf == "yes":
        repo_id = args.hf_repo_id or f"qwen25-vlsp-finetune-{args.run_mode}"
        print(f"Pushing model to HF Hub: {repo_id}")
        model.push_to_hub(repo_id, token=hf_token)
        tokenizer.push_to_hub(repo_id, token=hf_token)


if __name__ == "__main__":
    main()
