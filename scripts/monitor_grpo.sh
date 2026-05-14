#!/bin/bash
# monitor_grpo.sh — Check GRPO training health
# =============================================
# Run anytime during training. Reports key signals.
#
# Usage: ./monitor_grpo.sh [log_file]
#   default log: grpo_qwen_v2_final.log

LOG=${1:-grpo_qwen_v2_final.log}

if [ ! -f "$LOG" ]; then
    echo "ERROR: $LOG not found"
    echo "Usage: ./monitor_grpo.sh [log_file]"
    exit 1
fi

echo "================================================================"
echo "GRPO HEALTH CHECK: $LOG"
echo "================================================================"

# Tmux session status
echo ""
echo "--- tmux session ---"
tmux has-session -t grpo 2>/dev/null && echo "  RUNNING (session: grpo)" || echo "  not running"

# Latest progress
echo ""
echo "--- Latest step ---"
grep "Step" "$LOG" | tail -3

# Parse fail rate
echo ""
echo "--- Parse fails ---"
TOTAL_ROLLOUTS=$(grep -c "^    \[" "$LOG")
PARSE_FAILS=$(grep -c "PARSE FAIL" "$LOG")
if [ "$TOTAL_ROLLOUTS" -gt 0 ]; then
    RATE=$(echo "scale=1; $PARSE_FAILS * 100 / $TOTAL_ROLLOUTS" | bc)
    echo "  parse fails: $PARSE_FAILS / $TOTAL_ROLLOUTS rollouts (${RATE}%)"
    echo "  threshold:   <5% healthy, 5-20% degraded, >20% broken"
else
    echo "  no rollouts yet"
fi

# Recent KL
echo ""
echo "--- Recent KL trajectory ---"
grep -oP "kl=\d+\.\d+" "$LOG" | tail -10
echo "  threshold: <0.05 too low, 0.1-0.5 healthy, >1.0 diverging"

# Recent loss
echo ""
echo "--- Recent loss values ---"
grep -oP "loss=\-?\d+\.\d+" "$LOG" | tail -10

# Recent SEPO metrics (only refresh steps have real values)
echo ""
echo "--- Recent exploit (SEPO refresh steps) ---"
grep "Step" "$LOG" | grep -oP "e=\d+\.\d+" | grep -v "e=0.000" | tail -10
echo "  (should trend DOWN over time)"

echo ""
echo "--- Recent utility ---"
grep "Step" "$LOG" | grep -oP "u=\d+\.\d+" | tail -10
echo "  (should stay stable or trend up)"

# Checkpoints
echo ""
echo "--- Checkpoints saved ---"
OUTDIR=$(echo "$LOG" | sed 's/\.log$//')
if [ -d "$OUTDIR" ]; then
    ls -d "$OUTDIR"/step_* 2>/dev/null | tail -5 || echo "  none yet"
else
    echo "  output dir not found: $OUTDIR"
fi

# GPU status
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -2

echo ""
echo "================================================================"
echo "Re-run this script anytime to check progress."
echo "================================================================"
