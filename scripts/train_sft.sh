#!/bin/bash
# Stage 1: QLoRA SFT Warm Start (local Mac)
# Run: bash train_sft.sh

set -e
cd "$(dirname "$0")"

.venv/bin/mlx_lm.lora \
  --model mlx-community/gemma-3-4b-it-4bit \
  --train \
  --data sepo_sft_data \
  --batch-size 1 \
  --num-layers 8 \
  --iters 1000 \
  --grad-checkpoint \
  --mask-prompt \
  --max-seq-length 512 \
  --grad-accumulation-steps 4 \
  --adapter-path adapters/sepo_qlora \
  --save-every 100
