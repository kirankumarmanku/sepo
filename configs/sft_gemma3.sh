#!/bin/bash
# SFT config for Gemma 3 4B (lighter config — 1 epoch, smaller rank)
python -m train.sft \
  --model google/gemma-3-4b-it \
  --data-dir sepo_sft_data \
  --output-dir outputs/sft_gemma3 \
  --epochs 1 \
  --lr 1e-5 \
  --lora-rank 8 \
  --lora-dropout 0.0 \
  --batch-size 1 \
  --max-length 256 \
  --no-eval \
  --token-type-ids
