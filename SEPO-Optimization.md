# SEPO Optimization — Experiment Log

**Model:** Gemma 3 4B (`google/gemma-3-4b-it`)  
**Game:** Iterated Prisoner's Dilemma (IPD), GTBench-style  
**Objective:** Safe Equilibrium Policy Optimization — `J(π) = u(π) − λe·e(π) − λc·c(π) − λx·x(π)`  
**Strategic safety index:** `S(π) = u − 3e − 4c − 2x`  
**Payoffs:** T=5, R=3, P=1, S=0 | 8 rounds/episode | 20 episodes/opponent  

---

## Baselines (reference numbers from paper)

| Objective | Payoff ↑ | Welfare ↑ | Exploit ↓ | Robust ↑ | Externality ↓ | Safety ↑ |
|---|---|---|---|---|---|---|
| TFT (GTBench conventional) | 18.333 | 38.333 | 5.000 | 24.000 | 0.322 | 2.689 |
| Reward-only | 18.367 | 38.300 | 5.000 | 24.000 | 0.323 | 2.721 |
| Welfare-only | 16.450 | 43.767 | 32.500 | 23.750 | 0.140 | -81.330 |
| SEPO (full) | 18.367 | 38.050 | 5.250 | 23.838 | 0.325 | 1.966 |

> These are LLM optimizer results (no fine-tuning), computed by `llm_sepo_optimizer.py`.

---

## Gemma 3 4B — Base Model Evals

### Eval 1: Base model, prompt reasoning (no fine-tuning)

**Setup:** `gemma3:4b` via Ollama, `--reasoning prompt`, temperature=0.0  

| Payoff | Welfare | Exploit | Robust | Externality | Safety | Coop rate |
|---|---|---|---|---|---|---|
| 14.317 | 41.050 | 40.000 | 24.500 | 0.151 | -105.986 | 0.815 |

**Observations:**
- Model cooperates 81.5% of the time — excessively cooperative
- Exploitability=40 (catastrophic) — never retaliates against `always-defect`
- Gets 0 payoff vs `always-defect` in every episode

### Eval 2: Base model, chain-of-thought reasoning

**Setup:** `gemma3:4b` via Ollama, `--reasoning cot`, temperature=0.0  

| Payoff | Welfare | Exploit | Robust | Externality | Safety | Coop rate |
|---|---|---|---|---|---|---|
| 17.733 | 34.883 | 0.000 | 18.500 | 0.391 | 16.951 | 0.506 |

**Observations:**
- CoT accidentally discovers a near-defect strategy (50% coop)
- Exploitability=0 — best safety index across all evals
- Low robustness (18.5) and high externality (0.391) are the tradeoff
- CoT takes 8066s (~2.2hrs) vs 1968s for prompt — not practical for training

---

## Stage 1 — SFT Warm Start

### Setup

- **Base model:** `google/gemma-3-4b-it`
- **Training:** LoRA fine-tune on SEPO IPD demonstration traces
- **Data:** 6400 train / 1600 valid chat JSONL (`sepo_sft_data/`)
- **Strategy distribution in data:** TFT 85.2%, Grim 11.5%, AlwaysDefect 2.0%, GenTFT 0.8%, AlwaysCooperate 0.5%
- **LoRA config:** rank=8, alpha=16, target=`[q_proj, v_proj]`, no quantization
- **Training:** 1 epoch, lr=1e-5, cosine schedule, gradient checkpointing, bf16
- **Convergence:** Best checkpoint at iter 200, val loss=0.012
- **Artifact:** `kartiinx/gemma-3-4b-sepo-sft-hf` (HF Hub, private, LoRA adapter only)

### SFT Eval Results

**Setup:** `gemma_ipd_baseline.py --backend peft --model kartiinx/gemma-3-4b-sepo-sft-hf`, temperature=0.0  

| Payoff | Welfare | Exploit | Robust | Externality | Safety | Coop rate |
|---|---|---|---|---|---|---|
| 18.000 | 39.333 | 10.000 | 24.000 | 0.289 | -12.578 | 0.750 |

**Comparison with base model:**

| Metric | Base (prompt) | Base (CoT) | SFT |
|---|---|---|---|
| Payoff ↑ | 14.317 | 17.733 | **18.000** |
| Welfare ↑ | 41.050 | 34.883 | 39.333 |
| Exploitability ↓ | 40.000 | **0.000** | 10.000 |
| Robustness ↑ | 24.500 | 18.500 | **24.000** |
| Externality ↓ | 0.151 | 0.391 | **0.289** |
| Safety ↑ | -105.986 | **16.951** | -12.578 |

**Observations:**
- SFT achieves highest payoff (18.0) and best robustness (24.0)
- Exploitability drops from 40 → 10 — SFT learned reciprocity but imperfectly
- CoT base still wins on safety (16.951) due to its near-defect strategy having exploitability=0
- Coop rate 0.750 — SFT learned TFT-like behavior from data but cooperates too unconditionally against defectors
- **Root cause of exploitability=10:** SFT is behavioral cloning — it imitates TFT patterns but doesn't learn the strategic *reason* to punish defectors. The SEPO exploitability penalty is needed to close this gap.
- SFT is a good warm start: cooperative prior established, Stage 2 GRPO needs to bring exploitability 10 → ~5

---

## Stage 2 — GRPO with SEPO Objective

### Setup

- **Starting checkpoint:** `kartiinx/gemma-3-4b-sepo-sft-hf` (LoRA adapter, loaded via PEFT merge)
- **Framework:** Custom GRPO loop in `grpo_sepo.py`
- **SEPO weights:** λe=3.6, λc=3.2, λx=2.4
- **LoRA for GRPO policy:** rank=16, alpha=32, target=`[q_proj, v_proj]`
- **Reference model:** frozen SFT checkpoint, loaded in 4-bit to save VRAM
- **Hardware:** RTX 4090 24GB (RunPod)
- **Optimizer:** AdamW, lr=1e-5, grad clip=1.0
- **Loss:** Clipped surrogate (DeepSeek-R1 / GRPO paper) + KL penalty

```
L = -mean[ min(r·A, clip(r, 1-ε, 1+ε)·A) ] + β·KL(π || π_ref)
where r = exp(log π_new - log π_old),  ε=0.2,  β=0.01
```

### Attempt 1 — Episode-level reward (broken)

**Config:** n_rollouts=2→4, reward = single SEPO scalar per full episode  

**Training log (96 steps):**
```
Step  0 | loss=0.0000 | u=2.000 | e=5.000 | c=1.000 | x=0.083 | kl=0.0000
Step 16 | loss=0.0010 | u=2.000 | e=5.000 | c=1.000 | x=0.083 | kl=0.1016
Step 36 | loss=0.0000 | u=2.000 | e=5.000 | c=1.000 | x=0.083 | kl=0.0000
Step 64 | loss=0.0000 | u=2.000 | e=5.000 | c=1.000 | x=0.083 | kl=0.0000
```

**Problem:** Metrics completely frozen. Loss → 0.

**Root cause:** The SFT model is near-deterministic at temperature=0.8 — it almost always outputs `<SILENT>`. With episode-level reward, all n_rollouts produce identical action sequences → identical SEPO scalars → std=0 → all advantages=0 → no gradient signal. Additionally, the KL term went negative in early runs (policy drifting away from reference) before the `clamp(min=0)` fix was applied.

**Step_0016 eval** (from this broken run):

| Payoff | Welfare | Exploit | Robust | Externality | Safety | Coop rate |
|---|---|---|---|---|---|---|
| 16.000 | 45.333 | 40.000 | 24.000 | 0.089 | -104.178 | **1.000** |

Model regressed — coop rate went 0.75 → 1.00, exploitability 10 → 40. The zero-gradient run pushed the model toward always cooperating due to the unclamped negative KL dominating early steps.

### Attempt 2 — Per-round reward, temp=0.8 (abandoned at step 12)

**Key insight:** The SEPO reward aggregates 8 rounds into one scalar per episode, masking within-episode variance. Even if 2 of 8 rounds differ between rollouts, the aggregate hides it.

**Fix:** Compute advantage **per round** across n_rollouts, not per episode:

```
reward_t_r = payoff_t_r − SEPO_penalty_r
```

Where:
- `payoff_t_r` = immediate payoff at round t for rollout r (0, 1, 3, or 5)
- `SEPO_penalty_r` = λe·e + λc·c + λx·x computed for the full episode (shared across all rounds in that rollout)
- Advantage normalised across n_rollouts at each round t independently

**Grouping:** One group per train-pool opponent (AlwaysDefect, TFT, GenerousTFT). Advantages normalized within each group independently.

**Training log:**
```
Step  0 | loss=0.0199 | u=2.000 | e=5.000 | c=1.000 | x=0.111 | kl=1.9922
Step  4 | loss=0.0167 | u=2.000 | e=5.000 | c=1.000 | x=0.111 | kl=1.6719
Step  8 | loss=0.0040 | u=2.000 | e=5.000 | c=1.000 | x=0.111 | kl=0.4004
Step 12 | loss=0.0014 | u=2.000 | e=5.000 | c=1.000 | x=0.111 | kl=0.1357
```

**Problem:** Loss (0.0199→0.0014) and KL (2.0→0.14) both collapsed to near-zero by step 12. Metrics completely frozen throughout.

**Root cause:** Per-round reward is an improvement over episode-level reward — loss was non-zero at step 0 confirming some variance exists. However, the SFT model is still too deterministic at temperature=0.8. Almost all rollouts output `<SILENT>` every round → per-round payoffs are identical across rollouts → std≈0 → advantages≈0 → no gradient signal. The small initial loss came from rare temperature-induced defections but was insufficient to sustain training.

**Note on SEPO penalty:** The penalty (λe·e + λc·c + λx·x ≈ 21.5) is constant across rollouts when all rollouts behave identically — it cancels out in advantage normalization and has no effect on variance. Tuning λ weights will not help until behavioral diversity between rollouts is established first.

**Config:**
```bash
python grpo_sepo.py --model kartiinx/gemma-3-4b-sepo-sft-hf --base-model google/gemma-3-4b-it \
  --game ipd --lora --ref-4bit --n-rollouts 8 --iters 96 \
  --log-every 4 --save-every 16 --temperature 0.8 --output-dir grpo_gemma3_ipd_v2
```

---

### Attempt 3 — Per-round reward, temp=1.2 (running)

**Hypothesis:** Raising temperature to 1.2 forces the model to explore defection more frequently, creating non-zero per-round variance between rollouts even for a strongly cooperative SFT prior.

**Config:**
```bash
python grpo_sepo.py --model kartiinx/gemma-3-4b-sepo-sft-hf --base-model google/gemma-3-4b-it \
  --game ipd --lora --ref-4bit --n-rollouts 8 --iters 96 \
  --log-every 4 --save-every 16 --temperature 1.2 --output-dir grpo_gemma3_ipd_v3
```

**Watch for:** Loss staying above 0.01 past step 12, and `c` or `e` beginning to shift by step 20-40.

**Status:** Running.

---

## Experiment Tracker

| # | Phase | Label | Config | Issue | Loss/KL | Exploit | Safety | Coop | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Pre-train | Base prompt | gemma3:4b, temp=0.0, prompt reasoning | Excessively cooperative, never retaliates | — | 40.0 | -105.9 | 0.815 | Done |
| 2 | Pre-train | Base CoT | gemma3:4b, temp=0.0, CoT reasoning | Near-defect by accident, low robustness, 8066s runtime | — | 0.0 | +16.9 | 0.506 | Done |
| 3 | SFT | SFT warm start | LoRA r=8, lr=1e-5, 1 epoch, bf16, no quant | Behavioral cloning — imitates TFT but cooperates unconditionally vs defectors | val_loss=0.012 | 10.0 | -12.6 | 0.750 | Done |
| 4 | GRPO | Ep-reward, n_rollouts=2 | Episode-level SEPO scalar, 2 rollouts | KL went negative (−55 at step 20), optimizer maximized negative KL → always cooperate | loss→0, kl→−55 | — | — | — | Abandoned |
| 5 | GRPO | Ep-reward, n_rollouts=4, KL fix | Episode-level reward, clamp(kl≥0), 4 rollouts | Metrics completely frozen (u=2, e=5, c=1 every step), loss→0. SFT model too deterministic → all rollouts identical → std=0 → zero advantages | loss→0, kl→0 | 40.0 | -104.2 | 1.000 | Abandoned (step_0016 eval showed regression) |
| 6 | GRPO | Per-round reward, n_rollouts=8, temp=0.8 | Per-round advantage normalisation, clipped surrogate ε=0.2, per-opponent groups | Loss and KL collapsed to ~0 by step 12 (loss 0.0199→0.0014, kl 2.0→0.14 in 12 steps). Metrics frozen throughout (u=2, e=5, c=1). Model too deterministic at temp=0.8 — all rollouts output `<SILENT>` every round → zero variance even per-round | loss 0.02→0.001, kl 2.0→0.14 | — | — | 1.000 | Abandoned at step 12 |
| 7 | GRPO | Per-round reward, temp=1.2 | Same as #6, temperature raised to 1.2, output dir grpo_gemma3_ipd_v3 | Planned — higher temperature forces action diversity between rollouts, creating non-zero per-round variance | TBD | TBD | TBD | TBD | Running |
| 8 | GRPO | Multi-game joint | All 10 games, per-round reward, temp=1.2 | Planned — cross-game gradient diversity breaks IPD determinism indirectly | TBD | TBD | TBD | TBD | Phase 2 |
| — | Target | SEPO paper | LLM optimizer (no fine-tuning) | Reference | — | 5.25 | +1.97 | 0.852 | Reference |

---

## Summary Table

| Stage | Exploit ↓ | Safety ↑ | Coop rate | Notes |
|---|---|---|---|---|
| Base (prompt) | 40.0 | -105.9 | 0.815 | Always cooperative |
| Base (CoT) | 0.0 | +16.9 | 0.506 | Near-defect, low robustness |
| SFT warm start | 10.0 | -12.6 | 0.750 | Good prior, needs RL |
| GRPO attempt 1 (broken) | 40.0 | -104.2 | 1.000 | Zero variance collapse |
| GRPO attempt 2 (per-round) | TBD | TBD | TBD | Running |
| Target (SEPO paper) | 5.25 | +1.97 | ~0.852 | TFT-dominant strategy |

---

## Key Engineering Notes

- `kartiinx/gemma-3-4b-sepo-sft-hf` is a **LoRA adapter repo**, not a full model. `grpo_sepo.py` auto-detects this and calls `merge_and_unload()` before adding the GRPO LoRA on top.
- Gemma 3 requires `token_type_ids=zeros` in training mode (`modeling_gemma3.py` enforces this).
- PEFT + Gemma 3 requires `autocast_adapter_dtype=False` to avoid `float8_e8m0fnu` error on older PyTorch.
- Reference model loaded in 4-bit (`BitsAndBytesConfig`) to fit policy + ref in 24GB VRAM.
- Gradient checkpointing enabled on policy model.
- KL penalty clamped `≥ 0` to prevent optimizer from driving policy away from reference.
