#!/bin/bash
# SFT config for Gemma 4 E4B
python -m train.sft \
  --model google/gemma-4-E4B-it \
  --data-dir sepo_sft_data_multi \
  --output-dir outputs/sft_gemma4 \
  --epochs 3 \
  --lr 2e-5 \
  --lora-rank 32 \
  --batch-size 2 \
  --max-length 512 \
  --token-type-ids
