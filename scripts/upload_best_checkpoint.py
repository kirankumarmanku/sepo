"""
upload_best_checkpoint.py — Find best checkpoint by eval loss and upload to HF Hub
Usage:
  python upload_best_checkpoint.py --output-dir ./sft_output --hf-repo kartiinx/gemma-3-4b-sepo-sft-hf
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="./sft_output")
    p.add_argument("--hf-repo",    required=True)
    p.add_argument("--base-model", default="google/gemma-3-4b-it")
    args = p.parse_args()

    out_dir = Path(args.output_dir)

    # ── Find best checkpoint ───────────────────────────────────────────────────
    # trainer_state.json may be at top level or inside each checkpoint folder
    top_state = out_dir / "trainer_state.json"
    if top_state.exists():
        state_path = top_state
    else:
        # Read from the last checkpoint
        ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        state_path = ckpts[-1] / "trainer_state.json"

    with open(state_path) as f:
        state = json.load(f)

    # Print all eval losses
    print("Eval loss history:")
    eval_entries = [x for x in state["log_history"] if "eval_loss" in x]
    for x in eval_entries:
        print(f"  step {x['step']:4d}: eval_loss={x['eval_loss']:.4f}")

    best_ckpt = state.get("best_model_checkpoint")
    if best_ckpt is None:
        best = min(eval_entries, key=lambda x: x["eval_loss"])
        best_ckpt = str(out_dir / f"checkpoint-{best['step']}")

    print(f"\nBest checkpoint: {best_ckpt}")

    # ── Load and upload ────────────────────────────────────────────────────────
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(best_ckpt)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, best_ckpt, autocast_adapter_dtype=False)

    print(f"Uploading to {args.hf_repo} ...")
    tokenizer.push_to_hub(args.hf_repo, private=True)
    model.push_to_hub(args.hf_repo, private=True)
    print(f"Done → {args.hf_repo}")

if __name__ == "__main__":
    main()
