#!/bin/bash
# Stage 1b: Fuse LoRA adapter into Gemma 4 base model
# Run: bash fuse_and_export_gemma4.sh
# Then: hf upload kartiinx/gemma-4-e2b-sepo-sft sepo_sft_fused_gemma4/ .

set -e
cd "$(dirname "$0")"

# Use iter 200 checkpoint (best plasticity for GRPO warm start)
# Update this if Gemma 4 converges at a different iter
cp adapters/sepo_qlora_gemma4/0000200_adapters.safetensors adapters/sepo_qlora_gemma4/adapters.safetensors

.venv/bin/mlx_lm.fuse \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --adapter-path adapters/sepo_qlora_gemma4 \
  --save-path sepo_sft_fused_gemma4

echo "Fused model saved to sepo_sft_fused_gemma4/"
echo "Next: hf upload kartiinx/gemma-4-e2b-sepo-sft sepo_sft_fused_gemma4/ ."
