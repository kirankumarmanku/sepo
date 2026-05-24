#!/usr/bin/env bash
# run_all_games.sh
# Runs all 4 GTBench games with both prompt and CoT reasoning.
# Edit BACKEND, BASE_URL, and MODEL to match your setup.
# Results written to ./results/

BACKEND="openai"
BASE_URL="http://localhost:11434/v1"
MODEL="gemma3:4b"
EPISODES=20
OUTDIR="./results"
SEED=42

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTDIR"

games=("gemma_kuhn_poker" "gemma_blind_auction" "gemma_negotiation" "gemma_pig")
extra_args=("--games 20" "--episodes 20" "--episodes 20" "--games 20")

for i in "${!games[@]}"; do
    script="$SCRIPT_DIR/${games[$i]}.py"
    extra="${extra_args[$i]}"

    echo "=============================="
    echo "  ${games[$i]}  |  prompt"
    echo "=============================="
    python3 "$script" \
        --backend "$BACKEND" \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        $extra \
        --reasoning prompt \
        --seed "$SEED" \
        --output-dir "$OUTDIR"

    echo "=============================="
    echo "  ${games[$i]}  |  cot"
    echo "=============================="
    python3 "$script" \
        --backend "$BACKEND" \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        $extra \
        --reasoning cot \
        --seed "$SEED" \
        --output-dir "$OUTDIR"
done

echo ""
echo "All done. Results in $OUTDIR/"
ls "$OUTDIR"/*.json
