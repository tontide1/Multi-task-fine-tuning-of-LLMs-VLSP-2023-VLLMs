# Qwen2.5-1.5B Kaggle Fine-tuning Design Spec

## 1. Goal

Fine-tune `unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit` on Kaggle with QLoRA, optimizing for model quality on the VLSP-style Vietnamese reading comprehension and reasoning tasks. The first serious run should be easy to compare against the locked baseline, not merely produce a LoRA adapter.

Primary quality goals:

- Improve task behavior on `exams_mcq`, `wiki_mcq`, and `comprehension_short_answer`.
- Preserve instruction-following behavior via `instruction_retention`.
- Reduce language-modeling regression via `cloze_lm_retention`.
- Select checkpoints using task-aware validation signals instead of one undifferentiated train loss.

## 2. Dataset Source

Use Hugging Face dataset `tontide1/Dataset-for-fine-tuning-LLMS-VLSP-2023-benchmark`.

Expected files:

- `train_all.jsonl`: training pool.
- `val_all.jsonl`: validation for main benchmark-proxy tasks.
- `shadow_eval.jsonl`: harder held-out internal eval.
- `instruction_probe.jsonl`: instruction-retention guardrail.
- `cloze_probe.jsonl`: LM-retention guardrail.
- `split_report.json`: task counts and split provenance.

Important loading constraint: do not rely on `load_dataset(repo, split="train")`. `train_all.jsonl` contains heterogeneous `metadata` schemas across tasks, which can break Arrow schema inference. The notebook should download each JSONL file with `hf_hub_download`, parse rows with Python `json`, extract only `messages`, `metadata.task`, and token statistics fields, then build normalized `datasets.Dataset` objects from lists. Streaming may be used for quick inspection, but not for the first quality run because deterministic sampling and per-task evaluation are easier to reproduce with local parsed rows.

## 3. Data Formatting

Each record is already in OpenAI-style `messages` format:

- `messages[0]`: user prompt.
- `messages[1]`: assistant answer.

Formatting should be deterministic and template-safe:

- Use the Qwen2.5 chat template through `tokenizer.apply_chat_template(...)`.
- Produce a single `text` column for `SFTTrainer`.
- Use `add_generation_prompt=False` for supervised training examples.
- Ensure each formatted example terminates with the tokenizer EOS token if the chat template does not already add it.
- Do not manually concatenate `<|im_start|>` / `<|im_end|>` tokens unless the tokenizer template is unavailable.

The training dataset passed to `SFTTrainer` should contain only fields needed for training and analysis, typically `text` and `task`.

Loss masking:

- Default to assistant-only loss for QA and MCQ fine-tuning. The model should learn the answer behavior, not spend gradient budget modeling the user prompt and passage text.
- Use TRL/Unsloth completion-only masking when compatible with the Kaggle package versions.
- The expected assistant response marker for Qwen chat formatting is `<|im_start|>assistant\n`; verify this from one formatted sample before training.
- If completion-only masking is not compatible, fall back to full-sequence loss and record `loss_mode = "full_sequence_fallback"` in the run report.

## 4. Training Mix Strategy

Use a task-balanced mix light enough to preserve the natural distribution but prevent retention tasks and the large short-answer bucket from dominating the run.

First quality run uses deterministic caps with `seed = 3407`:

- `comprehension_short_answer`: `14_000`
- `exams_mcq`: `10_760` (all available train examples)
- `wiki_mcq`: `7_136` (all available train examples)
- `instruction_retention`: `8_000`
- `cloze_lm_retention`: `4_000`

This creates about `43_896` training examples with an intended ratio near:

- `comprehension_short_answer`: 31.9%
- `exams_mcq`: 24.5%
- `wiki_mcq`: 16.3%
- `instruction_retention`: 18.2%
- `cloze_lm_retention`: 9.1%

Sampling rule:

- Group parsed `train_all.jsonl` rows by `metadata.task`.
- Shuffle each group deterministically with `random.Random(3407)`.
- Take up to the configured cap for each task.
- Concatenate selected rows and shuffle the final list once with the same seed.
- Print and save `mix_report.json` with counts, caps, seed, and source file SHA or dataset revision.

## 5. Model Configuration

Base model:

- `unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit`

Core settings:

- `load_in_4bit = True`.
- `dtype = None` or explicit `torch.float16` on T4.
- Start with `max_seq_length = 2048`. The EDA report `notebooks/token_length_report.json` shows train `p99 = 591`, max sampled length `2421`, validation max `1008`, and overall train truncation above 2048 around `0.004%`.
- Keep `max_seq_length = 4096` only for a later ablation or if a full-data token report shows important examples are being truncated.

Token-length gate:

- Before training, compute token-length summary for the final sampled train mix and validation set: `p50`, `p90`, `p95`, `p99`, `max`, and truncation rate at 2048 by task.
- If overall truncation above 2048 is `<= 0.5%`, keep `max_seq_length = 2048`.
- If `comprehension_short_answer` truncation above 2048 is `> 1%`, inspect long examples before training; do not silently train on heavily truncated reading-comprehension examples.
- Log long examples above 2048 separately. Drop only clear outliers; otherwise keep them and record the truncation decision in the run report.

QLoRA adapter:

- `r = 16`.
- `lora_alpha = 16`.
- `lora_dropout = 0`.
- `bias = "none"`.
- `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
- `use_gradient_checkpointing = "unsloth"`.
- `random_state = 3407`.
- `use_rslora = False` for the first quality run.

## 6. Training Configuration

Use `SFTTrainer` with a `text` field, following the Unsloth reference script pattern.

Initial configuration:

- `per_device_train_batch_size = 2`.
- `gradient_accumulation_steps = 4`.
- Effective batch size: 8.
- `learning_rate = 2e-4`.
- `lr_scheduler_type = "linear"`.
- `warmup_ratio = 0.03` preferred for full runs; `warmup_steps = 5` only for very short smoke runs.
- `optim = "adamw_8bit"`.
- `weight_decay = 0.01`.
- `fp16 = True`, `bf16 = False` on Tesla T4.
- `packing = False` for the first run to avoid mixing unrelated tasks inside packed sequences.
- `seed = 3407`.
- `logging_steps = 10` for full runs, `1` for smoke runs.

Checkpointing:

- Save checkpoints often enough for Kaggle recovery and comparison, e.g. `save_steps = 250` or `500` depending on total steps.
- Use `save_total_limit = 2` or `3` to avoid disk pressure.
- Always save the final LoRA adapter and tokenizer locally before uploading.

## 7. Evaluation Strategy

Validation should be task-aware.

Main validation:

- Use `val_all.jsonl`.
- Format with the same Qwen2.5 chat template.
- Build separate eval datasets by `metadata.task`: `eval_exams_mcq`, `eval_wiki_mcq`, and `eval_comprehension_short_answer`.
- Evaluate each task separately with `trainer.evaluate(eval_dataset=..., metric_key_prefix=f"eval_{task}")`.
- Record both aggregate validation loss and per-task validation loss. Per-task losses are required for checkpoint comparison.

Shadow evaluation:

- Use `shadow_eval.jsonl` after a run is stable.
- Treat it as an internal held-out benchmark proxy, not as a training-time tuning set.

Probe evaluation:

- Use `instruction_probe.jsonl` to detect instruction-following regression.
- Use `cloze_probe.jsonl` to detect LM-retention regression.
- Run probes as separate evaluation datasets after training and after candidate checkpoint reloads.
- Probe degradation should block checkpoint promotion even if main validation improves.

Checkpoint selection:

- Primary score: weighted per-task validation loss, with `exams_mcq`, `wiki_mcq`, and `comprehension_short_answer` weighted equally for the first quality run.
- Guardrail rule: do not promote a checkpoint if `instruction_probe` or `cloze_probe` loss is clearly worse than the base-model or smoke-run reference. If no base probe exists yet, compare against the earliest stable checkpoint and inspect samples manually.
- Secondary check: manual sample outputs for Vietnamese reasoning, answer format, and hallucination risk.
- Save `eval_report.json` containing all task losses, probe losses, selected checkpoint path, and selection rationale.

## 8. Run Stages

Stage 1: quality smoke run

- Small sampled mix from every task.
- Purpose: verify formatting, assistant-only loss masking or fallback, token-length gate, loss decrease, and per-task evaluation reports.
- Do not use this adapter as a final result.

Stage 2: first quality run

- Use the task-balanced mix.
- Evaluate on `val_all` by task.
- Run probes after training.
- Save and upload the best candidate adapter only after `eval_report.json` identifies the selected checkpoint.

Stage 3: comparison run

- Compare task-balanced mix against natural mix or a slightly MCQ-heavier mix.
- Only run this if Stage 2 beats baseline proxies without probe regression.

## 9. Export Strategy

Export LoRA adapters first:

- Save local adapter directory, e.g. `lora_model/`.
- Save tokenizer with the adapter.
- Push adapter to Hugging Face, e.g. `tontide1/Qwen2.5-1.5B-VLSP-Adapter`.

Do not merge to 16-bit or GGUF in the first Kaggle training notebook unless the adapter has passed validation and probe checks. Merging/export conversion should be a separate step after checkpoint selection.

## 10. Success Criteria

A run is considered useful if:

- Training loss decreases without unstable spikes.
- `val_all` improves on at least one main task without degrading the others severely.
- Probe performance does not show obvious instruction or language-modeling regression.
- Sample generations follow the required Vietnamese answer format.
- The selected checkpoint, mix counts, training config, and eval results are recorded well enough to reproduce the run.
