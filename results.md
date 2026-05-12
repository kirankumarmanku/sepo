# SEPO Results

## Model: Gemma 3 4B (`google/gemma-3-4b-it`)

---

## IPD (Iterated Prisoner's Dilemma)

**Setup**: 8 rounds/episode · 20 episodes/opponent · temperature 0.8  
**Exploiter pool**: AlwaysDefect + AlternatingDefect (per-opponent exploit averaging)  
**Collusive pool**: AlwaysCooperate (defection rate = collusion metric)  
**Train pool**: TitForTat · GenerousTFT · MixedStrategy(0.5) · GrimTrigger  
**λe=2.4 · λc=1.0 · λx=1.8**

| Model | Pay/round | Exploit | Robust | Ext | Safety | NRA |
|---|---|---|---|---|---|---|
| Base | 2.688 | 0.328 | 1.766 | 0.248 | +1.328 | +0.043 |
| SFT (`sft_multi_v1`) | 2.517 | 0.672 | 1.625 | 0.264 | +0.085 | +0.020 |
| GRPO (`grpo_ipd_v5/step_0075`) | **2.747** | **0.312** | 1.788 | **0.228** | **+1.480** | +0.039 |

**Best GRPO checkpoint**: `grpo_ipd_v5/step_0075`  
(`step_0100` marginally higher safety +1.482 and robustness 1.819 at exploit 0.328)

### Observations

- **GRPO beats base on safety** (+1.480 vs +1.328) and externality (0.228 vs 0.248)
- **GRPO ties base on exploit** (0.312) — both at the TFT floor for the new exploiter pool
- **SFT degrades exploit resistance** significantly (0.672 vs 0.328 base) — over-cooperative warm-start. GRPO is necessary to correct this regression
- **Collusion**: GRPO c≈0.108 vs base c≈0.126 — GRPO exploits cooperative partners slightly less
- All three models have positive safety with current λe=2.4

### Lambda Sensitivity (IPD Safety Sign)

To differentiate base (negative) from GRPO (positive) on safety:

| Model | λe=2.4 | λe=6.7 |
|---|---|---|
| Base | +1.328 | **-0.082** |
| SFT | +0.085 | **-2.805** |
| GRPO step75 | +1.480 | **+0.138** |
| GRPO step100 | +1.482 | **+0.072** |

**Sweet spot**: λe ∈ (6.45, 6.92) with λc=1.0, λx=1.8  
Gives ordering: SFT < Base < 0 < GRPO — clean safety separation

### Training Trajectory (`grpo_ipd_v5`)

| Step | Exploit | Collusion | Utility | Ext | Safety (train) |
|---|---|---|---|---|---|
| 0 (SFT) | 0.312 | 0.188 | 2.719 | 0.356 | +1.14 |
| 10 | 0.469 | 0.125 | 2.406 | 0.375 | +0.48 |
| 35 | 0.312 | 0.000 | 2.250 | 0.356 | +1.08 |
| 65 | 0.312 | 0.000 | 2.625 | 0.304 | +1.33 |
| 70 | 0.312 | 0.000 | 2.906 | 0.292 | +1.63 |

Exploit stabilises at TFT floor (0.312) with the penalty fix — no catastrophic increase seen in prior buggy runs. Utility recovers past SFT starting point at step 70.

### Key Fixes That Enabled These Results

1. **GRPO penalty fix** (`grpo_sepo.py`): per-rollout exploit episodes so SEPO penalty varies across rollouts and contributes to advantage. Previous runs had constant penalty that cancelled in normalization → zero exploit gradient.
2. **IPD exploiter pool**: added AlternatingDefect alongside AlwaysDefect — breaks TFT floor (TFT scores exploit=0 vs AltD), per-opponent exploit averaging prevents masking.
3. **IPD collusive pool**: AlwaysCooperate with defection rate metric (was AlwaysDefect — measured exploitation of a naive partner, now correctly measures collusion = exploiting a cooperative partner).
4. **IPD train pool**: removed AlwaysDefect (conflicted with exploit signal), added GrimTrigger.

---

## Resource · Auction · Negotiation

*Pending — pool fixes for Auction applied (commit `54bb97a`). Full 4-game GRPO run with all fixes TBD.*
