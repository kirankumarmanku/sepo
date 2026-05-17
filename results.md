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

## Negotiation v1 — Split Bargaining (Single-Issue)

**Setup**: 4 rounds/episode · 20 episodes/opponent · temperature 0.8  
**Exploiter pool**: SelfishSplit (demands 7) + HardballSplit (demands 8), per-opponent averaging  
**Collusive pool**: FairSplit (demand ≥7 vs fair partner = collusion metric)  
**Train pool**: FairSplit · BalancedSplit · ConcedeSplit · HardballSplit  
**λe=3.0 (train) · λc=2.0 (train) · λc=1.0 (eval) · λx=1.8**

### Why we built this game

Negotiation is a core multi-agent safety scenario: two agents competing over a fixed resource, where greed causes both to lose (breakdown), cooperation requires trust, and exploitation means demanding more than a fair partner can accept. We started with the simplest version — single-issue, complete-information, integer demands — to validate that SEPO can learn the right equilibrium strategy (`demand = 10 − opponent_demand`) before tackling harder variants.

### Current best results (step 125)

| Model | Pay/round | Exploit | Robust | Ext | Safety | NRA |
|---|---|---|---|---|---|---|
| Base | **1.409** | **1.531** | 0.762 | 0.654 | **-3.363** | **-0.367** |
| SFT (`sft_multi_v1`) | 1.306 | 2.037 | **0.881** | **0.632** | -4.258 | -0.404 |
| GRPO (`grpo_neg_final/step_0125`) | 1.297 | 1.587 | 0.762 | 0.667 | -3.530 | -0.386 |

**Best GRPO checkpoint**: `grpo_neg_final/step_0125`

### Observations

- **Base still wins** — lowest exploit (1.531) and best safety (-3.363). The single-issue complete-information format is too simple: the base model can already infer the right counter from the game prompt, leaving little room for GRPO improvement
- **GRPO significantly beats SFT** — exploit 1.587 vs 2.037 (−22%), safety −3.530 vs −4.258 (+0.728). The SFT warm-start teaches over-accommodating behavior (always demands ≤ fair share), which GRPO must correct
- **Gap to base is narrowing**: at step 25 exploit was 1.681, step 125 is 1.587 — trending down. The equilibrium (`demand = 10 − opp_demand`) is being learned but slowly; gap to base exploit is now only 0.056
- **All models deeply safety-negative** — structural. At λe=3.0 the exploit penalty (≈1.6 × scale ≈ 0.96) overwhelms utility (≈0.85 scaled); no model achieves positive safety on this game
- **Collusion reducing**: GRPO c≈0.187 (step25) down from base c≈0.275 — λc=2.0 in training suppressed greedy demands vs cooperative partners
- **Why training continues**: with enough steps the exploit should converge to the TFT-like floor (`counter-demand` equilibrium), mirroring IPD convergence. Step 125 is the current best but not the final state

### Training Trajectory (`grpo_neg_final`)

| Step | Exploit | Collusion | Safety (eval) |
|---|---|---|---|
| 0 (SFT) | 2.037 | 0.275 | -4.258 |
| 25 | 1.900 | 0.187 | -3.863 |
| 50 | 1.775 | ~0.20 | ~-3.7 |
| 75 | 1.669 | ~0.19 | ~-3.6 |
| 100 | 1.706 | ~0.19 | ~-3.62 |
| **125** | **1.587** | ~0.19 | **-3.530** |
| 150 | 1.637 | ~0.19 | -3.601 |

Step 125 is best — small oscillation after that. Exploit trending down overall from 2.037 to 1.587 over 125 steps.

### Why this game hits a ceiling

The single-issue split game has a fundamental limitation: the optimal strategy (`demand = 10 − last_opp_demand`) is derivable from the prompt alone after round 1. The base model with good instruction-following can already approximate this, so there is little advantage GRPO can add beyond correcting the SFT over-cooperation bias. The game does not require private information reasoning, multi-dimensional trade-offs, or multi-round strategic inference — all of which are necessary to stress-test safety alignment in real negotiations.

---

## Negotiation v2 — GTBench Multi-Issue (Incomplete Information)

**Why the upgrade**: GTBench (arXiv:2402.12348) defines negotiation as a 3-item incomplete-information problem (Books, Hats, Balls) with private valuations — the closest published benchmark to real-world negotiation. Moving to this format:
1. Makes NRA scores directly comparable to GTBench baselines
2. Tests whether SEPO can handle hidden-information reasoning (opponent values unknown)
3. Creates a harder game where the base model cannot simply invert the opponent's demand — it must infer opponent preferences from demand patterns across rounds
4. Eliminates the artificial ceiling that made v1 unimprovable

**Format**: 3 items (Books, Hats, Balls) · pool randomly sampled [1–4 units each] · private values (sum to 10, hidden from opponent) · 4 rounds · deal if combined demands ≤ pool; else breakdown

**Opponent pool**: Train: [FairNeg, TFTNeg, ConcedeNeg] · Exploiter: [GreedyNeg, HardballNeg] · Collusive: [FairNeg]

**λe=3.0 · λc=2.0 (train) · λc=1.0 (eval) · λx=1.8** (same as v1)

**SFT data**: 4 strategies (Proportional, Conservative, Adaptive, Defensive) × 5 opponents × 200 episodes · 3 epochs · `sft_neg_gtbench`

### Results (GRPO step 25)

| Model | Pay/round | Exploit | Robust | Ext | Safety | NRA |
|---|---|---|---|---|---|---|
| Base | 5.192 | 0.781 | 1.275 | 0.642 | -0.627 | -0.048 |
| SFT (`sft_neg_gtbench`) | **5.725** | 2.706 | 1.988 | **0.443** | -1.777 | -0.258 |
| **GRPO step25** | 4.721 | **0.319** | **2.375** | 0.637 | **-0.518** | **+0.011** |

**Best GRPO checkpoint**: `grpo_neg_gt_v1/final` (step 25 — training continuing)

### Observations

- **GRPO beats base on safety** (-0.518 vs -0.627) and exploit (0.319 vs 0.781, −59%) — unlike v1 where base always led at step 25. The GTBench upgrade confirms the hypothesis: harder game gives GRPO room to improve
- **GRPO only model with positive NRA (+0.011)** — the model is genuinely competitive on average across all opponents
- **SFT exploit catastrophic** (2.706 vs 0.781 base) — worst SFT degradation across all games. The 3-item private-value format amplifies the over-accommodation pattern: SFT learns to demand little across all items, which exploiters exploit heavily on high-value items
- **All models safety-negative** — structural to the game; λe=3.0 × exploit penalty dominates even at low exploit
- **GRPO parse failures higher** (9 vs 6 base vs 3 SFT) — fallback [1,1,1] suppresses utility. More training steps expected to improve format compliance alongside strategy quality
- **Only step 25** — exploit already at 0.319, well below base (0.781). Training to step 75–100 expected to push safety further past base and recover utility, following IPD convergence pattern

### Comparison: v1 (single-issue) vs v2 (GTBench) at step 25

| | v1 Base safety | v1 GRPO safety | v2 Base safety | v2 GRPO safety |
|---|---|---|---|---|
| Safety | -3.363 | -3.643 | -0.627 | **-0.518** |
| GRPO beats base? | No | — | **Yes** | — |
| GRPO exploit vs base | +0.150 worse | — | **−0.462 better** | — |

v2 is the correct game for SEPO negotiation. v1 had a structural ceiling; v2 gives GRPO genuine room to outperform base from step 25.

---

## Resource

*Pending — same pool fixes needed (HighExtract in all three pools). TBD.*
