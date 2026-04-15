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
echo "  # Download SFT adapter first:"
echo "  huggingface-cli download kartiinx/gemma-3-4b-sepo-sft-hf \\"
echo "    --local-dir ./gemma3-sepo-sft"
echo ""
echo "  # RTX 4090 (LoRA + 4bit ref):"
echo "  python grpo_sepo.py \\"
echo "    --model ./gemma3-sepo-sft \\"
echo "    --game ipd \\"
echo "    --lora --ref-4bit \\"
echo "    --output-dir grpo_gemma3_ipd"
echo ""
echo "  # A40/A100 (full finetune):"
echo "  python grpo_sepo.py \\"
echo "    --model ./gemma3-sepo-sft \\"
echo "    --game ipd \\"
echo "    --output-dir grpo_gemma3_ipd"
echo ""
