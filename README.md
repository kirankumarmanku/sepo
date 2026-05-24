# SEPO — Strategic Equilibrium Policy Optimization

RL framework for training LLMs on game-theoretic objectives (social efficiency, exploitability, Pareto optimality) across multiple games.

## Project Structure

```
sepo/
├── games/              # Game environments (IPD, Kuhn Poker, Auction, Negotiation, etc.)
├── data/               # SFT data generation scripts
│   ├── generate_ipd.py
│   ├── generate_kuhn.py
│   ├── generate_multi.py
│   └── generate_neg_gtbench.py
├── train/              # Training scripts
│   ├── sft.py          # Unified SFT (supports Gemma 3/4, Qwen, etc.)
│   └── grpo.py         # GRPO reinforcement learning
├── eval/               # Evaluation
│   ├── eval_sepo.py    # Main SEPO metric evaluation
│   ├── gtbench/        # Per-game baselines (IPD, Auction, Kuhn, Negotiation, Pig)
│   └── ...
├── configs/            # Training/eval configurations
├── scripts/            # Shell scripts (setup, launch, monitor, export)
├── notebooks/          # Jupyter notebooks
├── docs/               # Writeup, analysis, and notes
├── outputs/            # .gitignored — checkpoints, logs, results
└── requirements.txt
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Generate SFT data

```bash
python -m data.generate_multi --output-dir sepo_sft_data_multi
python -m data.generate_kuhn --output-dir sepo_sft_data_kuhn
```

### 2. SFT training

```bash
# Qwen
python -m train.sft --model Qwen/Qwen3.5-4B --data-dir sepo_sft_data_multi --output-dir outputs/sft_qwen

# Gemma 4
python -m train.sft --model google/gemma-4-E4B-it --data-dir sepo_sft_data_multi \
  --output-dir outputs/sft_gemma4 --token-type-ids

# Gemma 3 (smaller config)
python -m train.sft --model google/gemma-3-4b-it --data-dir sepo_sft_data \
  --output-dir outputs/sft_gemma3 --epochs 1 --lora-rank 8 --no-eval --token-type-ids
```

### 3. GRPO training

```bash
python -m train.grpo --model outputs/sft_qwen/final_adapter \
  --base-model Qwen/Qwen3.5-4B --game all --output-dir outputs/grpo_qwen
```

### 4. Evaluation

```bash
python -m eval.eval_sepo --model Qwen/Qwen3.5-4B \
  --adapter outputs/grpo_qwen/final_adapter --game all
```

## GTBench Baselines

```bash
# Run all games with Ollama
cd eval/gtbench && bash run_all_games.sh
```
