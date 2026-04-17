#!/bin/bash
# RunPod Setup — SEPO Training (Stage 1 SFT + Stage 2 GRPO)
# ==========================================================
# Run once after the pod starts.
#
# Recommended pods:
#   Stage 1 SFT  (Gemma 3 4B LoRA)  → RTX 4090 24GB  (~$0.44/hr)
#   Stage 2 GRPO (Gemma 3 4B LoRA)  → RTX 4090 24GB  (~$0.44/hr)
#   Stage 2 GRPO (Gemma 4 e2b LoRA) → A40 48GB        (~$0.79/hr)
#
# Usage:
#   bash setup_runpod.sh

set -e

echo "=== Installing dependencies ==="
pip install -q torch transformers accelerate peft trl datasets \
               bitsandbytes huggingface_hub numpy scipy openai

echo ""
echo "=== Cloning repo (grpo-stage2 branch) ==="
git clone -b grpo-stage2 https://github.com/kirankumarmanku/sepo.git
cd sepo

echo ""
echo "=== Logging into HuggingFace ==="
echo "Paste your HF token when prompted (needs read+write access)"
huggingface-cli login

echo ""
echo "======================================================"
echo "Setup complete. Choose what to run:"
echo "======================================================"
echo ""
echo "── Stage 1: SFT warm start (Gemma 3 4B) ─────────────"
echo ""
echo "  python sft_train.py \\"
echo "    --hf-repo kartiinx/gemma-3-4b-sepo-sft-hf"
echo ""
echo "  # With custom output dir:"
echo "  python sft_train.py \\"
echo "    --output-dir ./sft_gemma3 \\"
echo "    --hf-repo kartiinx/gemma-3-4b-sepo-sft-hf"
echo ""
echo "── Stage 2: GRPO (start after SFT adapter is on HF) ──"
echo ""
echo "  # NOTE: kartiinx/gemma-3-4b-sepo-sft-hf is a LoRA adapter repo."
echo "  # grpo_sepo.py auto-detects this and merges the adapter before training."
echo ""
echo "  # RTX 4090 (LoRA GRPO policy + 4bit ref model, ~22GB VRAM):"
echo "  python grpo_sepo.py \\"
echo "    --model kartiinx/gemma-3-4b-sepo-sft-hf \\"
echo "    --base-model google/gemma-3-4b-it \\"
echo "    --game ipd \\"
echo "    --lora --ref-4bit \\"
echo "    --token-type-ids \\"
echo "    --output-dir grpo_gemma3_ipd"
echo ""
echo "  # A40/A100 48GB (full finetune, no LoRA):"
echo "  python grpo_sepo.py \\"
echo "    --model kartiinx/gemma-3-4b-sepo-sft-hf \\"
echo "    --base-model google/gemma-3-4b-it \\"
echo "    --game ipd \\"
echo "    --token-type-ids \\"
echo "    --output-dir grpo_gemma3_ipd"
echo ""
echo "── Stage 2: GRPO directly on Gemma 4 (no SFT) ───────"
echo ""
echo "  # A40 48GB (LoRA GRPO, 4bit ref model):"
echo "  python grpo_sepo.py \\"
echo "    --model google/gemma-4-e2b-it \\"
echo "    --game ipd \\"
echo "    --lora --ref-4bit \\"
echo "    --max-new-tokens 512 \\"
echo "    --temperature 1.0 \\"
echo "    --output-dir grpo_gemma4_ipd"
echo ""
