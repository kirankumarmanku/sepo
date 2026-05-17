#!/bin/bash
# Stage 1b: Fuse LoRA adapter into base model after SFT completes
# Run: bash fuse_and_export.sh
# Then upload sepo_sft_fused/ to HuggingFace Hub

set -e
cd "$(dirname "$0")"

# Using iter 200 checkpoint (val loss 0.012) — best plasticity for GRPO warm start
cp adapters/sepo_qlora/0000200_adapters.safetensors adapters/sepo_qlora/adapters.safetensors

.venv/bin/mlx_lm.fuse \
  --model mlx-community/gemma-3-4b-it-4bit \
  --adapter-path adapters/sepo_qlora \
  --save-path sepo_sft_fused

echo "Fused model saved to sepo_sft_fused/"
echo "Next: huggingface-cli upload <your-repo> sepo_sft_fused/"
