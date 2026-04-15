#!/bin/bash
# RunPod Setup — SEPO Stage 2 GRPO Training
# ==========================================
# Run this once after the pod starts.
#
# Recommended pods:
#   Gemma 3 4B  (bf16, full finetune) → A40 48GB  or A100 40GB
#   Gemma 3 4B  (LoRA + 4bit ref)     → RTX 4090 24GB
#   Gemma 4 e2b (bf16, full finetune) → A100 80GB
#   Gemma 4 e2b (LoRA + 4bit ref)     → A40 48GB
#
# Usage:
#   bash setup_runpod.sh

set -e

echo "=== Installing dependencies ==="
pip install -q torch transformers accelerate peft bitsandbytes huggingface_hub numpy scipy

echo "=== Logging into HuggingFace ==="
echo "Run: huggingface-cli login"
echo "Then paste your HF token (needs read access to private repos)"
huggingface-cli login

echo "=== Downloading SFT checkpoint ==="
echo "Choose one:"
echo "  Gemma 3 4B: huggingface-cli download kartiinx/gemma-3-4b-sepo-sft --local-dir ./gemma3-sepo-sft"
echo "  Gemma 4 e2b: huggingface-cli download kartiinx/gemma-4-e2b-sepo-sft --local-dir ./gemma4-sepo-sft"

echo ""
echo "=== Setup complete. Training commands: ==="
echo ""
echo "# Gemma 3 4B — full finetune (needs A40/A100):"
echo "python grpo_sepo.py --model ./gemma3-sepo-sft --game ipd --output-dir grpo_gemma3_ipd"
echo ""
echo "# Gemma 3 4B — LoRA + 4bit ref (fits RTX 4090):"
echo "python grpo_sepo.py --model ./gemma3-sepo-sft --game ipd --lora --ref-4bit --output-dir grpo_gemma3_ipd_lora"
echo ""
echo "# Gemma 4 e2b — LoRA + 4bit ref (fits A40):"
echo "python grpo_sepo.py --model ./gemma4-sepo-sft --game ipd --lora --ref-4bit --output-dir grpo_gemma4_ipd_lora"
echo ""
echo "# Custom SEPO weights:"
echo "python grpo_sepo.py --model ./gemma3-sepo-sft --game ipd --lambda-e 0.5 --lambda-c 2.0 --lambda-x 1.0"
