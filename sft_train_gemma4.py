"""
sft_train_gemma4.py — SFT for Gemma 4 E4B on SEPO data
=========================================================
Adapts the Qwen training pipeline for Gemma 4:
- Auto-detects ClippableLinear wrapping (Gemma 4) vs plain projections
- Optionally injects token_type_ids=zeros for the multimodal model
- Default model: google/gemma-4-E4B-it

Usage:
  python sft_train_gemma4.py --data-dir sepo_sft_data_multi --output-dir ./sft_gemma4_multi
  python sft_train_gemma4.py --data-dir sepo_sft_data_kuhn  --output-dir ./sft_gemma4_kuhn

Note: Gemma 4 is license-gated on HuggingFace.  Accept the license at
https://huggingface.co/google/gemma-4-E4B-it before first download, and
run `huggingface-cli login` if you haven't already.
"""

import argparse
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class TokenTypeIdsCollator:
    """Inject token_type_ids=zeros — required for Gemma 4 multimodal text-only training."""
    def __init__(self, tokenizer):
        self.base = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def __call__(self, features):
        batch = self.base(features)
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


def detect_lora_targets(model) -> list:
    """Auto-detect LoRA targets, scoping to the language model only.

    Multimodal Gemma 4 has vision/audio encoders with Gemma4ClippableLinear
    wrappers AND a text decoder with plain nn.Linear projections. We must
    target only the text decoder — vision/audio modules never run during
    text-only training, so LoRA on them gets zero gradients.
    """
    import torch.nn as nn

    text_targets = []
    vision_audio_count = 0

    for name, module in model.named_modules():
        # Match self-attention projections in the language model only
        if any(name.endswith(f"self_attn.{p}") for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
            # Is this a Gemma4ClippableLinear wrapper (vision/audio)? Skip it.
            if not isinstance(module, nn.Linear):
                vision_audio_count += 1
                continue
            # Is it in the language model? Keep its full path.
            if "language_model" in name or name.startswith("model.layers"):
                text_targets.append(name)

    if vision_audio_count > 0:
        print(f"  Skipped {vision_audio_count} vision/audio projections")

    if not text_targets:
        # Fallback for text-only Gemma variants
        print("  No language_model paths found — falling back to suffix match")
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    print(f"  Targeting {len(text_targets)} language_model projections")
    print(f"  Example: {text_targets[0]}")
    return text_targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="google/gemma-4-E4B-it",
                   help="HuggingFace model ID (default: google/gemma-4-E4B-it)")
    p.add_argument("--data-dir",    default="sepo_sft_data_multi")
    p.add_argument("--output-dir",  default="./sft_output_gemma4")
    p.add_argument("--hf-repo",     default=None,
                   help="Push adapter to this HF repo (optional)")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--max-length",  type=int,   default=512)
    p.add_argument("--lora-rank",   type=int,   default=32)
    p.add_argument("--batch-size",  type=int,   default=2)
    p.add_argument("--token-type-ids", action="store_true",
                   help="Inject token_type_ids=zeros in batches (required for Gemma 4 "
                        "multimodal models; harmless when not needed)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    print(f"\nLoading model: {args.model}")
    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if supports_bf16 else torch.float16
    print(f"dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
    )

    # Detect LoRA targets BEFORE wrapping in PEFT
    lora_targets = detect_lora_targets(model)
    print(f"  LoRA target_modules: {lora_targets}")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=lora_targets,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg, autocast_adapter_dtype=False)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print(f"VRAM after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        bf16=supports_bf16,
        fp16=not supports_bf16,
        logging_steps=25,
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

    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
    )
    if args.token_type_ids:
        print("  Using TokenTypeIdsCollator for multimodal text-only training")
        trainer_kwargs["data_collator"] = TokenTypeIdsCollator(tokenizer)

    trainer = SFTTrainer(**trainer_kwargs)

    print("\nStarting SFT training...")
    trainer.train()
    print("Training complete.")

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
