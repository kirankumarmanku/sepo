"""
test_gemma4_grad.py — Smoke test for Gemma 4 LoRA gradient flow
================================================================
Confirms that gradients actually flow back to LoRA weights through
the Gemma4ClippableLinear wrapper. Run BEFORE committing to a long
training run.

Run twice:
  python test_gemma4_grad.py                 # with default settings
  python test_gemma4_grad.py --no-checkpoint # without gradient checkpointing

If grad-norm prints "0.0" or "nan", LoRA isn't training.
If grad-norm prints a positive number (e.g. 0.5, 1.2), gradients flow.
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable gradient checkpointing")
    p.add_argument("--target", choices=["inner", "outer"], default="inner",
                   help="Target Linear inside wrapper ('inner') or wrapper itself ('outer')")
    args = p.parse_args()

    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    if args.target == "inner":
        targets = ["q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear"]
    else:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    print(f"LoRA targets: {targets}")

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
        target_modules=targets, lora_dropout=0.0, bias="none",
    )
    model = get_peft_model(model, cfg, autocast_adapter_dtype=False)
    model.enable_input_require_grads()

    if not args.no_checkpoint:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing: ENABLED")
    else:
        print("Gradient checkpointing: DISABLED")

    model.print_trainable_parameters()

    # Synthetic batch
    text = "The capital of France is Paris. The capital of Germany is Berlin. The"
    inputs = tok(text, return_tensors="pt").to(model.device)
    inputs["labels"] = inputs["input_ids"].clone()

    # Forward + backward
    model.train()
    out = model(**inputs)
    print(f"Loss: {out.loss.item():.4f}")
    out.loss.backward()

    # Check LoRA gradients
    total_grad = 0.0
    n_lora_params = 0
    n_zero = 0
    n_nonzero = 0
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            n_lora_params += 1
            if param.grad is None:
                n_zero += 1
            else:
                g = param.grad.norm().item()
                total_grad += g
                if g == 0.0:
                    n_zero += 1
                else:
                    n_nonzero += 1

    print()
    print(f"LoRA params with grad:    {n_lora_params}")
    print(f"  with non-zero grad:     {n_nonzero}")
    print(f"  with zero/None grad:    {n_zero}")
    print(f"Sum of LoRA grad norms:   {total_grad:.6f}")
    print()
    if total_grad > 0 and n_nonzero > 0:
        print("PASS — gradients flow to LoRA weights.")
    else:
        print("FAIL — no gradients reaching LoRA weights.")


if __name__ == "__main__":
    main()
