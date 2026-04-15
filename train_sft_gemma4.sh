#!/bin/bash
# Stage 1: QLoRA SFT Warm Start — Gemma 4 e4b (MoE, 4B active params)
# Run: bash train_sft_gemma4.sh

set -e
cd "$(dirname "$0")"

mkdir -p adapters/sepo_qlora_gemma4

.venv/bin/mlx_lm.lora \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --train \
  --data sepo_sft_data \
  --batch-size 1 \
  --num-layers 8 \
  --iters 1000 \
  --grad-checkpoint \
  --mask-prompt \
  --max-seq-length 512 \
  --grad-accumulation-steps 4 \
  --adapter-path adapters/sepo_qlora_gemma4 \
  --save-every 100
