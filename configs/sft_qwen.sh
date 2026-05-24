#!/bin/bash
# SFT config for Qwen 3.5-4B
python -m train.sft \
  --model Qwen/Qwen3.5-4B \
  --data-dir sepo_sft_data_multi \
  --output-dir outputs/sft_qwen \
  --epochs 3 \
  --lr 2e-5 \
  --lora-rank 32 \
  --batch-size 2 \
  --max-length 512
