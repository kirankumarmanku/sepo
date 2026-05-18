"""
sft_train.py — Stage 1: SFT Warm Start
=======================================
Plain LoRA fine-tune of google/gemma-3-4b-it on SEPO IPD demonstrations.

Usage:
  python sft_train.py
  python sft_train.py --output-dir ./sft_output --hf-repo kartiinx/gemma-3-4b-sepo-sft-hf
"""

import argparse
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class TextOnlyCollator:
    """Gemma 3 is multimodal — injects token_type_ids=zeros for text-only training.
    Skipped for Gemma 4 and other models that don't require token_type_ids."""
    def __init__(self, tokenizer, inject_token_type_ids: bool = False):
        self.base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        self.inject = inject_token_type_ids

    def __call__(self, features):
        batch = self.base(features)
        if self.inject:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="google/gemma-3-4b-it")
    p.add_argument("--data-dir",    default="sepo_sft_data")
    p.add_argument("--output-dir",  default="./sft_output")
    p.add_argument("--hf-repo",     default=None, help="Push adapter to this HF repo (optional)")
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--lr",          type=float, default=1e-5)
    p.add_argument("--max-length",  type=int,   default=256)
    p.add_argument("--lora-rank",   type=int,   default=8)
    p.add_argument("--token-type-ids", action="store_true",
                   help="Inject token_type_ids=zeros (required for Gemma 3, not Gemma 4)")
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
    # Use bf16 on A-series GPUs (A40/A100), fp16 on RTX (4090 etc.)
    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if supports_bf16 else torch.float16
    print(f"dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
    )

    # Gemma 4 uses q_proj.linear / v_proj.linear; Gemma 3 uses plain q_proj / v_proj
    named = {n for n, _ in model.named_modules()}
    lora_targets = (["q_proj.linear", "v_proj.linear"]
                    if "model.layers.0.self_attn.q_proj.linear" in named
                    else ["q_proj", "v_proj"])

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=lora_targets,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg, autocast_adapter_dtype=False)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print(f"VRAM after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    collator = TextOnlyCollator(tokenizer, inject_token_type_ids=args.token_type_ids)

    # ── Training ──────────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        gradient_checkpointing=True,
        bf16=supports_bf16,
        fp16=not supports_bf16,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=args.max_length,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    print("\nStarting SFT training...")
    trainer.train()
    print("Training complete.")

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_path = f"{args.output_dir}/final_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved → {adapter_path}")

    # ── Upload to HF Hub ──────────────────────────────────────────────────────
    if args.hf_repo:
        print(f"Uploading to HF Hub: {args.hf_repo}")
        model.push_to_hub(args.hf_repo, private=True)
        tokenizer.push_to_hub(args.hf_repo, private=True)
        print(f"Done → {args.hf_repo}")


if __name__ == "__main__":
    main()
