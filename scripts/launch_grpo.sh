#!/bin/bash
# launch_grpo.sh — Launch the real 100-iter GRPO training run
# =============================================================
# Uses the hyperparameters you confirmed via probe_temperature.sh and probe_lr.sh.
# Defaults to Gemma's known-working values if you skip probing.
#
# Usage:
#   ./launch_grpo.sh                          # uses defaults
#   ./launch_grpo.sh 0.8 3e-6 0.05            # temp lr beta
#
# Runs in tmux so SSH disconnect won't kill it.
# Monitor with: tmux attach -t grpo

set -e

# Defaults (Gemma's known-working values)
TEMP=${1:-0.8}
LR=${2:-3e-6}
BETA=${3:-0.05}

MODEL="kirankumarmanku/Qwen3.5-4B-sepo-sft-v2"
BASE="Qwen/Qwen3.5-4B"
OUTDIR="grpo_qwen_v2_final"

echo "================================================================"
echo "Launching GRPO with:"
echo "  temperature  = $TEMP"
echo "  learning rate = $LR"
echo "  beta         = $BETA"
echo "  output dir   = $OUTDIR"
echo "================================================================"

# Verify model exists / can be loaded (quick sanity)
python -c "from huggingface_hub import list_repo_files; list_repo_files('$MODEL')" \
    || { echo "ERROR: cannot access $MODEL"; exit 1; }

# Kill any existing tmux session named 'grpo'
tmux kill-session -t grpo 2>/dev/null || true

# Launch in tmux
tmux new-session -d -s grpo "
CUDA_VISIBLE_DEVICES=0 python grpo_sepo.py \
    --model $MODEL \
    --base-model $BASE \
    --game all \
    --lora --lora-rank 16 --ref-4bit \
    --iters 100 \
    --n-rollouts 4 \
    --temperature $TEMP \
    --max-new-tokens 512 \
    --lr $LR \
    --beta $BETA \
    --lambda-e 2.4 --lambda-c 2.4 --lambda-x 2.4 \
    --save-every 25 \
    --log-every 5 \
    --output-dir $OUTDIR 2>&1 | tee ${OUTDIR}.log
"

echo ""
echo "GRPO launched in tmux session 'grpo'."
echo ""
echo "Monitor live:    tmux attach -t grpo"
echo "Detach:          Ctrl-b then d"
echo "Check log:       tail -f ${OUTDIR}.log"
echo "Check progress:  grep 'Step' ${OUTDIR}.log | tail -20"
echo "Check parse fails: grep -c 'PARSE FAIL' ${OUTDIR}.log"
echo "Check KL:        grep -oP 'kl=\d+\.\d+' ${OUTDIR}.log | tail -10"
echo ""
echo "Expected runtime: ~20-30 hours for 100 iters."
echo "Checkpoints saved to: $OUTDIR/step_0025, step_0050, etc."
