# SEPO Training Command Reference

Complete commands for Qwen3.5-4B and Gemma 4 E4B-it across all games. Uses refactored module structure (`train.sft`, `train.grpo`, `eval.eval_sepo`).

**Scope:**
- **Qwen3.5-4B**: 4 games — IPD, Auction, Negotiation (original), Kuhn
- **Gemma 4 E4B-it**: 4 games — IPD, Auction, Negotiation_GT (new), Kuhn

GPU layout (2× A40 46GB): run training on GPU 0, evals on GPU 1 in parallel once first checkpoint exists.

## Setup

```bash
cd /root/workspace/sepo
git pull
mkdir -p logs eval_results outputs

huggingface-cli login   # accept Gemma 4 license first at hf.co/google/gemma-4-E4B-it
```

## Data Generation

Generate once, reused across both models for shared games.

```bash
# Multi-game data (IPD + Auction + Negotiation + Resource) — used by Qwen
python -m data.generate_multi \
    --episodes-per-opponent 200 \
    --output-dir sepo_sft_data_multi \
    --balance-games

# Kuhn data (single-game, used by both models)
python -m data.generate_kuhn \
    --episodes-per-opponent 200 \
    --output-dir sepo_sft_data_kuhn

# GTBench negotiation data (used only by Gemma 4)
python -m data.generate_neg_gtbench \
    --episodes-per-opponent 200 \
    --output-dir sepo_sft_neg_gtbench
```

---

# Qwen3.5-4B

Games: IPD, Auction, Negotiation (original), Kuhn.

## Qwen SFT

### Multi-game SFT (covers IPD, Auction, Negotiation, Resource jointly)

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.sft \
    --model Qwen/Qwen3.5-4B \
    --data-dir sepo_sft_data_multi \
    --output-dir outputs/sft_qwen_multi \
    --epochs 3 \
    --lr 2e-5 \
    --lora-rank 32 \
    --max-length 512 \
    --batch-size 2 \
    --hf-repo kirankumarmanku/Qwen3.5-4B-sepo-sft-v2 \
    2>&1 | tee logs/sft_qwen_multi.log
```

Expected: ~6 hr, loss → ~0.025.

### Kuhn-only SFT

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.sft \
    --model Qwen/Qwen3.5-4B \
    --data-dir sepo_sft_data_kuhn \
    --output-dir outputs/sft_qwen_kuhn \
    --epochs 3 \
    --lr 2e-5 \
    --lora-rank 32 \
    --max-length 512 \
    --batch-size 2 \
    --hf-repo kirankumarmanku/Qwen3.5-4B-sepo-kuhn-sft \
    2>&1 | tee logs/sft_qwen_kuhn.log
```

Expected: ~2.5 hr, loss → ~0.05.

## Qwen GRPO

### Multi-game joint GRPO (IPD + Auction + Negotiation)

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.grpo \
    --model kirankumarmanku/Qwen3.5-4B-sepo-sft-v2 \
    --base-model Qwen/Qwen3.5-4B \
    --game all \
    --lora --lora-rank 16 --ref-4bit \
    --iters 100 \
    --n-rollouts 4 --temperature 0.8 --max-new-tokens 512 \
    --lr 1e-5 --beta 0.1 \
    --lambda-e 2.4 --lambda-c 2.4 --lambda-x 2.4 \
    --save-every 25 --log-every 5 \
    --output-dir grpo_qwen_all \
    2>&1 | tee logs/grpo_qwen_all.log
```

Expected: ~20-25 hr.

### Kuhn-only GRPO (safer hyperparameters)

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.grpo \
    --model kirankumarmanku/Qwen3.5-4B-sepo-kuhn-sft \
    --base-model Qwen/Qwen3.5-4B \
    --game kuhn \
    --lora --lora-rank 16 --ref-4bit \
    --iters 100 \
    --n-rollouts 4 --temperature 0.8 --max-new-tokens 512 \
    --lr 3e-6 --beta 0.2 \
    --lambda-e 1.5 --lambda-c 2.4 --lambda-x 0.0 \
    --save-every 25 --log-every 5 \
    --output-dir grpo_qwen_kuhn \
    2>&1 | tee logs/grpo_qwen_kuhn.log
```

Lower lr (3e-6) and higher beta (0.2) prevent KL drift seen with default 1e-5/0.1 on Kuhn. λx=0 because Kuhn is zero-sum.

Expected: ~12-15 hr.

## Qwen Evaluation

Run on GPU 1 in parallel once step_0025 checkpoint exists.

### IPD / Auction / Negotiation (from multi-game GRPO)

```bash
# Base
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model Qwen/Qwen3.5-4B \
    --game all --episodes 8 --temperature 0.0 --max-tokens 512 \
    --lambda-e 2.4 --lambda-c 2.4 --lambda-x 2.4 \
    --label "base" --output-dir eval_results/qwen_all_base \
    2>&1 | tee logs/eval_qwen_all_base.log

# SFT
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model Qwen/Qwen3.5-4B \
    --adapter kirankumarmanku/Qwen3.5-4B-sepo-sft-v2 \
    --game all --episodes 8 --temperature 0.0 --max-tokens 512 \
    --lambda-e 2.4 --lambda-c 2.4 --lambda-x 2.4 \
    --label "sft" --output-dir eval_results/qwen_all_sft \
    2>&1 | tee logs/eval_qwen_all_sft.log

# GRPO checkpoints
for STEP in step_0025 step_0050 step_0075 final; do
    if [ -d "grpo_qwen_all/$STEP" ]; then
        CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
            --model Qwen/Qwen3.5-4B \
            --sft-adapter kirankumarmanku/Qwen3.5-4B-sepo-sft-v2 \
            --adapter "grpo_qwen_all/$STEP" \
            --game all --episodes 8 --temperature 0.0 --max-tokens 512 \
            --lambda-e 2.4 --lambda-c 2.4 --lambda-x 2.4 \
            --label "grpo_$STEP" \
            --output-dir "eval_results/qwen_all_grpo_$STEP" \
            2>&1 | tee "logs/eval_qwen_all_$STEP.log"
    fi
done
```

### Kuhn (from Kuhn-only GRPO)

```bash
# Base
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model Qwen/Qwen3.5-4B \
    --game kuhn --episodes 8 --temperature 0.0 --max-tokens 512 \
    --lambda-e 1.5 --lambda-c 2.4 --lambda-x 0.0 \
    --label "base" --output-dir eval_results/qwen_kuhn_base \
    2>&1 | tee logs/eval_qwen_kuhn_base.log

# SFT
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model Qwen/Qwen3.5-4B \
    --adapter kirankumarmanku/Qwen3.5-4B-sepo-kuhn-sft \
    --game kuhn --episodes 8 --temperature 0.0 --max-tokens 512 \
    --lambda-e 1.5 --lambda-c 2.4 --lambda-x 0.0 \
    --label "sft" --output-dir eval_results/qwen_kuhn_sft \
    2>&1 | tee logs/eval_qwen_kuhn_sft.log

# GRPO checkpoints
for STEP in step_0025 step_0050 step_0075 final; do
    if [ -d "grpo_qwen_kuhn/$STEP" ]; then
        CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
            --model Qwen/Qwen3.5-4B \
            --sft-adapter kirankumarmanku/Qwen3.5-4B-sepo-kuhn-sft \
            --adapter "grpo_qwen_kuhn/$STEP" \
            --game kuhn --episodes 8 --temperature 0.0 --max-tokens 512 \
            --lambda-e 1.5 --lambda-c 2.4 --lambda-x 0.0 \
            --label "grpo_$STEP" \
            --output-dir "eval_results/qwen_kuhn_grpo_$STEP" \
            2>&1 | tee "logs/eval_qwen_kuhn_$STEP.log"
    fi
done
```

---

# Gemma 4 E4B-it

Games: IPD, Auction, Negotiation_GT (GTBench variant), Kuhn.

## Gemma 4 LoRA gradient smoke test

Run once before launching SFT:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/test_gemma4_grad.py \
    --model google/gemma-4-E4B-it \
    --target inner
```

Expected: `PASS — gradients flow to LoRA weights.`

## Gemma 4 SFT

### Multi-game SFT (covers IPD, Auction, Negotiation, Resource)

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.sft \
    --model google/gemma-4-E4B-it \
    --data-dir sepo_sft_data_multi \
    --output-dir outputs/sft_gemma4_multi \
    --epochs 3 \
    --lr 2e-5 \
    --lora-rank 32 \
    --max-length 512 \
    --batch-size 2 \
    --token-type-ids \
    --hf-repo kirankumarmanku/gemma-4-E4B-sepo-sft-v2 \
    2>&1 | tee logs/sft_gemma4_multi.log
```

Expected: ~10.5 hr, loss → ~0.09.

### Kuhn-only SFT

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.sft \
    --model google/gemma-4-E4B-it \
    --data-dir sepo_sft_data_kuhn \
    --output-dir outputs/sft_gemma4_kuhn \
    --epochs 3 \
    --lr 2e-5 \
    --lora-rank 32 \
    --max-length 512 \
    --batch-size 2 \
    --token-type-ids \
    --hf-repo kirankumarmanku/gemma-4-E4B-sepo-kuhn-sft \
    2>&1 | tee logs/sft_gemma4_kuhn.log
```

Expected: ~95 min, loss → ~0.13.

### Negotiation_GT SFT

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.sft \
    --model google/gemma-4-E4B-it \
    --data-dir sepo_sft_neg_gtbench \
    --output-dir outputs/sft_gemma4_neg_gt \
    --epochs 3 \
    --lr 2e-5 \
    --lora-rank 32 \
    --max-length 512 \
    --batch-size 2 \
    --token-type-ids \
    --hf-repo kirankumarmanku/gemma-4-E4B-sepo-neg-gtbench-sft \
    2>&1 | tee logs/sft_gemma4_neg_gt.log
```

Expected: ~3 hr.

## Gemma 4 GRPO

### IPD GRPO (per-game tuned lambdas)

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.grpo \
    --model kirankumarmanku/gemma-4-E4B-sepo-sft-v2 \
    --base-model google/gemma-4-E4B-it \
    --game ipd \
    --lora --lora-rank 16 --ref-4bit \
    --iters 100 \
    --n-rollouts 4 --temperature 0.8 --max-new-tokens 512 \
    --lr 1e-5 --beta 0.1 \
    --lambda-e 2.4 --lambda-c 1.0 --lambda-x 1.8 \
    --token-type-ids \
    --save-every 25 --log-every 5 \
    --output-dir grpo_gemma4_ipd \
    2>&1 | tee logs/grpo_gemma4_ipd.log
```

λc=1.0 and λx=1.8 — Gemma 4 IPD SFT is already cooperative, lower collusion/externality weights maintain that. Best checkpoint typically step 25.

Expected: ~5-7 hr.

### Auction GRPO

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.grpo \
    --model kirankumarmanku/gemma-4-E4B-sepo-sft-v2 \
    --base-model google/gemma-4-E4B-it \
    --game auction \
    --lora --lora-rank 16 --ref-4bit \
    --iters 100 \
    --n-rollouts 4 --temperature 0.8 --max-new-tokens 512 \
    --lr 1e-5 --beta 0.1 \
    --lambda-e 2.4 --lambda-c 2.4 --lambda-x 2.4 \
    --token-type-ids \
    --save-every 25 --log-every 5 \
    --output-dir grpo_gemma4_auction \
    2>&1 | tee logs/grpo_gemma4_auction.log
```

Best checkpoint typically step 75 (final regresses — known KL drift issue).

Expected: ~10-12 hr.

### Negotiation_GT GRPO

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.grpo \
    --model kirankumarmanku/gemma-4-E4B-sepo-neg-gtbench-sft \
    --base-model google/gemma-4-E4B-it \
    --game negotiation_gt \
    --lora --lora-rank 16 --ref-4bit \
    --iters 100 \
    --n-rollouts 4 --temperature 0.8 --max-new-tokens 512 \
    --lr 1e-5 --beta 0.1 \
    --lambda-e 2.4 --lambda-c 2.4 --lambda-x 2.4 \
    --token-type-ids \
    --save-every 25 --log-every 5 \
    --output-dir grpo_gemma4_neg_gt \
    2>&1 | tee logs/grpo_gemma4_neg_gt.log
```

Best checkpoint step 75 (safety +2.187 — strongest positive-safety result).

Expected: ~18-20 hr.

### Kuhn GRPO (safer hyperparameters)

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.grpo \
    --model kirankumarmanku/gemma-4-E4B-sepo-kuhn-sft \
    --base-model google/gemma-4-E4B-it \
    --game kuhn \
    --lora --lora-rank 16 --ref-4bit \
    --iters 100 \
    --n-rollouts 4 --temperature 0.8 --max-new-tokens 512 \
    --lr 3e-6 --beta 0.2 \
    --lambda-e 1.5 --lambda-c 2.4 --lambda-x 0.0 \
    --token-type-ids \
    --save-every 25 --log-every 5 \
    --output-dir grpo_gemma4_kuhn \
    2>&1 | tee logs/grpo_gemma4_kuhn.log
```

Best checkpoint typically final (monotonic improvement, no late regression).

Expected: ~12-15 hr.

## Gemma 4 Evaluation

### IPD

```bash
# Base
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model google/gemma-4-E4B-it \
    --game ipd --episodes 8 --temperature 0.0 --max-tokens 512 \
    --lambda-e 2.4 --lambda-c 1.0 --lambda-x 1.8 \
    --token-type-ids \
    --label "base" --output-dir eval_results/gemma4_ipd_base \
    2>&1 | tee logs/eval_gemma4_ipd_base.log

# SFT
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model google/gemma-4-E4B-it \
    --adapter kirankumarmanku/gemma-4-E4B-sepo-sft-v2 \
    --game ipd --episodes 8 --temperature 0.0 --max-tokens 512 \
    --lambda-e 2.4 --lambda-c 1.0 --lambda-x 1.8 \
    --token-type-ids \
    --label "sft" --output-dir eval_results/gemma4_ipd_sft \
    2>&1 | tee logs/eval_gemma4_ipd_sft.log

# GRPO checkpoints
for STEP in step_0025 step_0050 step_0075 final; do
    if [ -d "grpo_gemma4_ipd/$STEP" ]; then
        CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
            --model google/gemma-4-E4B-it \
            --sft-adapter kirankumarmanku/gemma-4-E4B-sepo-sft-v2 \
            --adapter "grpo_gemma4_ipd/$STEP" \
            --game ipd --episodes 8 --temperature 0.0 --max-tokens 512 \
            --lambda-e 2.4 --lambda-c 1.0 --lambda-x 1.8 \
            --token-type-ids \
            --label "grpo_$STEP" \
            --output-dir "eval_results/gemma4_ipd_grpo_$STEP" \
            2>&1 | tee "logs/eval_gemma4_ipd_$STEP.log"
    fi
done
```

### Auction

```bash
# Base
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model google/gemma-4-E4B-it \
    --game auction --episodes 8 --temperature 0.0 --max-tokens 512 \
    --lambda-e 2.4 --lambda-c 2.4 --lambda-x 2.4 \
    --token-type-ids \
    --label "base" --output-dir eval_results/gemma4_auction_base \
    2>&1 | tee logs/eval_gemma4_auction_base.log

# SFT
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model google/gemma-4-E4B-it \
    --adapter kirankumarmanku/gemma-4-E4B-sepo-sft-v2 \
    --game auction --episodes 8 --temperature 0.0 --max-tokens 512 \
    --lambda-e 2.4 --lambda-c 2.4 --lambda-x 2.4 \
    --token-type-ids \
    --label "sft" --output-dir eval_results/gemma4_auction_sft \
    2>&1 | tee logs/eval_gemma4_auction_sft.log

# GRPO checkpoints
for STEP in step_0025 step_0050 step_0075 final; do
    if [ -d "grpo_gemma4_auction/$STEP" ]; then
        CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
            --model google/gemma-4-E4B-it \
            --sft-adapter kirankumarmanku/gemma-4-E4B-sepo-sft-v2 \
            --adapter "grpo_gemma4_auction/$STEP" \
            --game auction --episodes 8 --temperature 0.0 --max-tokens 512 \
            --lambda-e 2.4 --lambda-c 2.4 --lambda-x 2.4 \
            --token-type-ids \
            --label "grpo_$STEP" \
            --output-dir "eval_results/gemma4_auction_grpo_$STEP" \
            2>&1 | tee "logs/eval_gemma4_auction_$STEP.log"
    fi
done
```

### Negotiation_GT

```bash
# Base
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model google/gemma-4-E4B-it \
    --game negotiation_gt --episodes 8 --temperature 0.0 --max-tokens 512 \
    --lambda-e 2.4 --lambda-c 2.4 --lambda-x 2.4 \
    --token-type-ids \
    --label "base" --output-dir eval_results/gemma4_neg_gt_base \
    2>&1 | tee logs/eval_gemma4_neg_gt_base.log

# SFT
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model google/gemma-4-E4B-it \
    --adapter kirankumarmanku/gemma-4-E4B-sepo-neg-gtbench-sft \
    --game negotiation_gt --episodes 8 --temperature 0.0 --max-tokens 512 \
    --lambda-e 2.4 --lambda-c 2.4 --lambda-x 2.4 \
    --token-type-ids \
    --label "sft" --output-dir eval_results/gemma4_neg_gt_sft \
    2>&1 | tee logs/eval_gemma4_neg_gt_sft.log

# GRPO checkpoints
for STEP in step_0025 step_0050 step_0075 final; do
    if [ -d "grpo_gemma4_neg_gt/$STEP" ]; then
        CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
            --model google/gemma-4-E4B-it \
            --sft-adapter kirankumarmanku/gemma-4-E4B-sepo-neg-gtbench-sft \
            --adapter "grpo_gemma4_neg_gt/$STEP" \
            --game negotiation_gt --episodes 8 --temperature 0.0 --max-tokens 512 \
            --lambda-e 2.4 --lambda-c 2.4 --lambda-x 2.4 \
            --token-type-ids \
            --label "grpo_$STEP" \
            --output-dir "eval_results/gemma4_neg_gt_grpo_$STEP" \
            2>&1 | tee "logs/eval_gemma4_neg_gt_$STEP.log"
    fi
done
```

### Kuhn

```bash
# Base
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model google/gemma-4-E4B-it \
    --game kuhn --episodes 8 --temperature 0.0 --max-tokens 512 \
    --lambda-e 1.5 --lambda-c 2.4 --lambda-x 0.0 \
    --token-type-ids \
    --label "base" --output-dir eval_results/gemma4_kuhn_base \
    2>&1 | tee logs/eval_gemma4_kuhn_base.log

# SFT
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model google/gemma-4-E4B-it \
    --adapter kirankumarmanku/gemma-4-E4B-sepo-kuhn-sft \
    --game kuhn --episodes 8 --temperature 0.0 --max-tokens 512 \
    --lambda-e 1.5 --lambda-c 2.4 --lambda-x 0.0 \
    --token-type-ids \
    --label "sft" --output-dir eval_results/gemma4_kuhn_sft \
    2>&1 | tee logs/eval_gemma4_kuhn_sft.log

# GRPO checkpoints
for STEP in step_0025 step_0050 step_0075 final; do
    if [ -d "grpo_gemma4_kuhn/$STEP" ]; then
        CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
            --model google/gemma-4-E4B-it \
            --sft-adapter kirankumarmanku/gemma-4-E4B-sepo-kuhn-sft \
            --adapter "grpo_gemma4_kuhn/$STEP" \
            --game kuhn --episodes 8 --temperature 0.0 --max-tokens 512 \
            --lambda-e 1.5 --lambda-c 2.4 --lambda-x 0.0 \
            --token-type-ids \
            --label "grpo_$STEP" \
            --output-dir "eval_results/gemma4_kuhn_grpo_$STEP" \
            2>&1 | tee "logs/eval_gemma4_kuhn_$STEP.log"
    fi
done
```

---

# Reference

## HuggingFace repos

| Model | Game | Repo |
|---|---|---|
| Qwen3.5-4B | IPD/Auction/Neg/Res | `kirankumarmanku/Qwen3.5-4B-sepo-sft-v2` |
| Qwen3.5-4B | Kuhn | `kirankumarmanku/Qwen3.5-4B-sepo-kuhn-sft` |
| Gemma 4 E4B-it | IPD/Auction/Neg/Res | `kirankumarmanku/gemma-4-E4B-sepo-sft-v2` |
| Gemma 4 E4B-it | Kuhn | `kirankumarmanku/gemma-4-E4B-sepo-kuhn-sft` |
| Gemma 4 E4B-it | Negotiation_GT | `kirankumarmanku/gemma-4-E4B-sepo-neg-gtbench-sft` |

## Per-game hyperparameters

| Game | lr | beta | λe | λc | λx |
|---|---|---|---|---|---|
| IPD (Gemma 4) | 1e-5 | 0.1 | 2.4 | **1.0** | **1.8** |
| IPD (Qwen, joint) | 1e-5 | 0.1 | 2.4 | 2.4 | 2.4 |
| Auction | 1e-5 | 0.1 | 2.4 | 2.4 | 2.4 |
| Negotiation (original) | 1e-5 | 0.1 | 2.4 | 2.4 | 2.4 |
| Negotiation_GT | 1e-5 | 0.1 | 2.4 | 2.4 | 2.4 |
| Kuhn | **3e-6** | **0.2** | **1.5** | 2.4 | **0.0** |

## Best checkpoints per game

| Game | Qwen best | Gemma 4 best |
|---|---|---|
| IPD | step 100 | **step 25** (with tuned λ) |
| Auction | step 100 | **step 75** |
| Negotiation (original) | step 100 | not run |
| Negotiation_GT | not run | **step 75** |
| Kuhn | step 75 | **final** |

## Total runtime estimates

| Stage | Qwen | Gemma 4 |
|---|---|---|
| Data generation | ~10 min | ~10 min (shared) |
| SFT (all variants) | ~9 hr | ~14 hr |
| GRPO (all games) | ~35-40 hr | ~45-55 hr |
| All evals (parallel GPU 1) | ~4 hr | ~6 hr |
| **Total wall time** | **~50 hr** | **~70 hr** |

## Monitoring

```bash
grep "Step " logs/grpo_<run>.log | tail -5
grep -oP "kl=\d+\.\d+" logs/grpo_<run>.log | tail -10
ls grpo_<run>/
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```
