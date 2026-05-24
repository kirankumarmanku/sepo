#!/usr/bin/env bash
# run_all_games_gemma4.sh — GTBench evaluation with Gemma 4
#
# Gemma 4 model sizes on Ollama:
#   gemma4:e4b  — 9.6GB   (default, ~4B effective params, edge-optimised)
#   gemma4:e2b  — 7.2GB   (lighter, ~2B effective params)
#   gemma4:26b  — 18GB    (needs 16GB+ VRAM)
#   gemma4:31b  — larger  (needs 24GB+ VRAM)
#
# Pull your chosen model first:
#   ollama pull gemma4:e4b
#
# Then run:
#   bash run_all_games_gemma4.sh

BACKEND="openai"
BASE_URL="http://localhost:11434/v1"
MODEL="gemma4:e4b"        # change to gemma4:e2b / gemma4:26b / gemma4:31b as needed
EPISODES=20
OUTDIR="./results_gemma4"
SEED=42

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTDIR"

games=("gemma_kuhn_poker" "gemma_blind_auction" "gemma_negotiation" "gemma_pig" "gemma_ipd_baseline")
extra_args=("--games 20" "--episodes 20" "--episodes 20" "--games 20" "--episodes 20")

for i in "${!games[@]}"; do
    script="$SCRIPT_DIR/${games[$i]}.py"
    extra="${extra_args[$i]}"

    if [ ! -f "$script" ]; then
        echo "Skipping $script (not found)"
        continue
    fi

    echo "=============================="
    echo "  ${games[$i]}  |  prompt  |  $MODEL"
    echo "=============================="
    python "$script" \
        --backend "$BACKEND" \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        $extra \
        --max-tokens 8192 \
        --reasoning prompt \
        --seed "$SEED" \
        --output-dir "$OUTDIR"

    echo "=============================="
    echo "  ${games[$i]}  |  cot  |  $MODEL"
    echo "=============================="
    python "$script" \
        --backend "$BACKEND" \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        $extra \
        --max-tokens 8192 \
        --reasoning cot \
        --seed "$SEED" \
        --output-dir "$OUTDIR"
done

echo ""
echo "All done. Results in $OUTDIR/"
ls "$OUTDIR"/*.json 2>/dev/null
