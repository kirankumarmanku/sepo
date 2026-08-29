"""
train/sft.py — Unified SFT training for SEPO models
====================================================
Supports Gemma 3, Gemma 4, Qwen, and other HuggingFace causal LMs via LoRA.

Usage:
  python -m train.sft --model Qwen/Qwen3.5-4B --data-dir sepo_sft_data_multi
  python -m train.sft --model google/gemma-4-E4B-it --data-dir sepo_sft_data_kuhn --token-type-ids
  python -m train.sft --model google/gemma-3-4b-it --data-dir sepo_sft_data --epochs 1 --lora-rank 8
"""

import argparse
import json

import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class TokenTypeIdsCollator:
    """Injects token_type_ids=zeros for multimodal models (Gemma 3/4) in text-only training."""

    def __init__(self, tokenizer):
        self.base = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def __call__(self, features):
        batch = self.base(features)
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


def detect_lora_targets(model) -> list:
    """Auto-detect LoRA targets, scoping to the language model only.

    Handles:
    - Gemma 4 multimodal (ClippableLinear wrappers in vision/audio, plain Linear in text)
    - Gemma 3 (q_proj.linear wrappers vs plain q_proj)
    - Standard models (plain projection layers)
    """
    text_targets = []
    vision_audio_count = 0

    for name, module in model.named_modules():
        if any(name.endswith(f"self_attn.{p}") for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
            if not isinstance(module, nn.Linear):
                vision_audio_count += 1
                continue
            if "language_model" in name or name.startswith("model.layers"):
                text_targets.append(name)

    if vision_audio_count > 0:
        print(f"  Skipped {vision_audio_count} vision/audio projections")

    if text_targets:
        print(f"  Targeting {len(text_targets)} language_model projections")
        print(f"  Example: {text_targets[0]}")
        return text_targets

    # Fallback: check for Gemma 3-style .linear wrappers
    named = {n for n, _ in model.named_modules()}
    if any(n.endswith("q_proj.linear") for n in named):
        print("  Detected Gemma 3 ClippableLinear wrappers")
        return ["q_proj.linear", "v_proj.linear"]

    print("  Using default projection targets")
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def main():
    p = argparse.ArgumentParser(description="Unified SFT training for SEPO models")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--data-dir", required=True, help="Directory with train.jsonl and valid.jsonl")
    p.add_argument("--output-dir", default="./outputs/sft", help="Output directory")
    p.add_argument("--hf-repo", default=None, help="Push adapter to this HF repo")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--no-eval", action="store_true", help="Disable evaluation during training")
    p.add_argument("--token-type-ids", action="store_true",
                   help="Inject token_type_ids=zeros (required for Gemma 3/4 multimodal)")
    p.add_argument("--resume", default=None, help="Resume from checkpoint path")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Loading data from {args.data_dir}/")
    train_rows = load_jsonl(f"{args.data_dir}/train.jsonl")
    valid_rows = load_jsonl(f"{args.data_dir}/valid.jsonl")

    def format_example(row):
        return {"text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )}

    train_dataset = Dataset.from_list(train_rows).map(format_example)
    valid_dataset = Dataset.from_list(valid_rows).map(format_example)
    print(f"Train: {len(train_dataset)}  Valid: {len(valid_dataset)}")

    # ── Model + LoRA ──────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if supports_bf16 else torch.float16
    print(f"dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
    )

    lora_targets = detect_lora_targets(model)
    print(f"  LoRA target_modules: {lora_targets[:4]}{'...' if len(lora_targets) > 4 else ''}")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=lora_targets,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg, autocast_adapter_dtype=False)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    if torch.cuda.is_available():
        print(f"VRAM after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ── Training config ───────────────────────────────────────────────────────
    eval_strategy = "no" if args.no_eval else "steps"

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # trl >= 1.10 dropped warmup_ratio from SFTConfig; 100 steps approximates
        # the previous 5% warmup for our dataset sizes (~2k steps/epoch)
        warmup_steps=100,
        gradient_checkpointing=True,
        bf16=supports_bf16,
        fp16=not supports_bf16,
        logging_steps=args.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_strategy != "no" else None,
        save_strategy="steps" if eval_strategy != "no" else "epoch",
        save_steps=args.save_steps if eval_strategy != "no" else None,
        save_total_limit=3,
        load_best_model_at_end=(eval_strategy != "no"),
        metric_for_best_model="eval_loss" if eval_strategy != "no" else None,
        max_length=args.max_length,
        dataset_text_field="text",
        report_to="none",
    )

    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if eval_strategy != "no" else None,
        processing_class=tokenizer,
    )
    if args.token_type_ids:
        print("  Using TokenTypeIdsCollator for multimodal text-only training")
        trainer_kwargs["data_collator"] = TokenTypeIdsCollator(tokenizer)

    trainer = SFTTrainer(**trainer_kwargs)

    print("\nStarting SFT training...")
    trainer.train(resume_from_checkpoint=args.resume)
    print("Training complete.")

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_path = f"{args.output_dir}/final_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved -> {adapter_path}")

    if args.hf_repo:
        print(f"Uploading to HF Hub: {args.hf_repo}")
        model.push_to_hub(args.hf_repo, private=True)
        tokenizer.push_to_hub(args.hf_repo, private=True)
        print(f"Done -> {args.hf_repo}")


if __name__ == "__main__":
    main()
