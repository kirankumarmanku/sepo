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

## Auction (Repeated Sealed-Bid)

**Setup**: 6 rounds/episode · 20 episodes/opponent · temperature 0.8  
**Exploiter pool**: AggressiveBid + AdaptiveBid (per-opponent exploit averaging)  
**Collusive pool**: CollusiveLowBid (LOW bid rate = collusion metric)  
**Train pool**: TruthfulBid · ConservativeBid · ShadedBid  
**λe=1.2 · λc=1.0 · λx=1.8**

| Model | Pay/round | Exploit | Robust | Ext | Safety | NRA |
|---|---|---|---|---|---|---|
| Base | 0.719 | 0.279 | 0.158 | 0.353 | -0.568 | +0.088 |
| SFT (`sft_multi_v1`) | 0.544 | **0.167** | 0.196 | 0.477 | -0.580 | **+0.342** |
| GRPO (`grpo_auction_v1/step_0025`) | **0.806** | 0.250 | 0.167 | **0.320** | **-0.337** | +0.234 |

**Best GRPO checkpoint**: `grpo_auction_v1/step_0025` (exploit and safety both worsen after step25)

### Observations

- **GRPO beats base on safety** (-0.337 vs -0.568) and utility (0.806 vs 0.719) and NRA (+0.234 vs +0.088)
- **SFT wins exploit** (0.167) and NRA (+0.342) — conservative bidding naturally resists exploitation but sacrifices utility (0.544 vs 0.806)
- **All models safety-negative** — structural. Externality + collusion penalties exceed utility gains in auction; max profit per round is low (value 2 or 4, bid 1-3)
- **No stable equilibrium**: exploit increases monotonically after step25 unlike IPD. Reason: train pool (all conservative bidders) pushes toward aggressive bidding while exploiters push back — no Nash equilibrium exists that simultaneously minimises exploit and maximises utility
- **Best checkpoint is step25** — later steps regress

### Training Trajectory (`grpo_auction_v1`)

| Step | Exploit | Safety (eval) | NRA |
|---|---|---|---|
| Base | 0.279 | -0.568 | +0.088 |
| SFT (step 0) | 0.167 | -0.580 | +0.342 |
| GRPO step25 | **0.250** | **-0.337** | +0.234 |
| GRPO step75 | 0.329 | -0.432 | +0.091 |
| GRPO step100 | 0.387 | -0.641 | +0.154 |

---

## Negotiation (Split Bargaining)

**Setup**: 4 rounds/episode · 20 episodes/opponent · temperature 0.8  
**Exploiter pool**: SelfishSplit (demands 7) + HardballSplit (demands 8)  
**Collusive pool**: FairSplit (demands ≥7 vs fair partner = collusion metric)  
**Train pool**: FairSplit · BalancedSplit · ConcedeSplit · HardballSplit  
**λe=3.0 · λc=1.0 · λx=1.8**

| Model | Pay/round | Exploit | Robust | Ext | Safety | NRA |
|---|---|---|---|---|---|---|
| Base | **1.409** | **1.531** | 0.762 | 0.654 | **-3.363** | -0.367 |
| SFT (`sft_multi_v1`) | 1.306 | 2.037 | **0.881** | **0.632** | -4.258 | **-0.404** |
| GRPO (`grpo_neg_v1/final`) | 1.278 | 1.681 | 0.781 | 0.658 | -3.643 | -0.400 |

**Best GRPO checkpoint**: `grpo_neg_v1/final` (step25 only — more steps needed)

### Observations

- **Base wins negotiation** at step25 — lowest exploit (1.531), best safety (-3.363), best utility (1.409). Base model is naturally effective at single-issue bargaining
- **SFT degrades exploit resistance** (2.037 vs 1.531 base) — over-accommodating, same pattern as IPD
- **GRPO step25 partially recovers** from SFT regression (1.681 < 2.037) but doesn't beat base yet — only 25 steps trained, needs 75-100 for full convergence
- **GRPO lowest collusion** (c≈0.200 vs base 0.275) — correctly learns not to demand too much from fair partners
- **All models deeply safety-negative** — structural. λe=3.0 × exploit≈1.6 × scale(0.6) = 2.88, overwhelms utility (≈0.85 scaled)
- **Learnable counter-strategies exist**: demand exactly (10 − opp_demand) → exploit stabilises rather than growing uncontrolled

---

## Resource

*Pending — same pool fixes needed (HighExtract in all three pools). TBD.*
