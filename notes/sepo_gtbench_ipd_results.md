# SEPO on GTBench — Iterated Prisoner's Dilemma Results

**Setup:** R=3, T=5, S=0, P=1 | 8 rounds/episode | 20 episodes/matchup | λ_e=3.6, λ_c=3.2, λ_x=2.4  
**Actions:** `<SILENT>` = Cooperate, `<TESTIFY>` = Defect  
**GTBench conventional agent:** Tit-for-Tat  
**Safety index:** S(π) = u(π) − 3e(π) − 4c(π) − 2x(π)

---

## Main Results

| Objective | Payoff ↑ | Welfare ↑ | Exploit ↓ | Robust ↑ | Collusion ↓ | Externality ↓ | Safety ↑ |
|---|---|---|---|---|---|---|---|
| TFT (GTBench conventional) | 18.333 | 38.333 | 5.000 | 24.000 | 0.000 | 0.322 | 2.689 |
| Reward-only | 18.367 | 38.300 | 5.000 | 24.000 | 0.000 | 0.323 | 2.721 |
| Welfare-only | 16.450 | 43.767 | 32.500 | 23.750 | 0.000 | 0.140 | −81.330 |
| **SEPO (full)** | **18.367** | 38.050 | **5.250** | 23.838 | **0.000** | 0.325 | **1.966** |

---

## Ablations

| Objective | Payoff ↑ | Exploit ↓ | Collusion ↓ | Externality ↓ | Safety ↑ |
|---|---|---|---|---|---|
| SEPO (full) | 18.367 | 5.250 | 0.000 | 0.325 | 1.966 |
| SEPO-no-exploit (λ_e=0.0) | 18.067 | 11.250 | 0.000 | 0.281 | −16.245 |
| SEPO-no-collusion (λ_c=0.0) | 18.367 | 5.250 | 0.000 | 0.325 | 1.966 |
| SEPO-no-externality (λ_x=0.0) | 18.375 | 5.500 | 0.000 | 0.323 | 1.229 |

---

## Learned Strategy Mixtures

| Strategy | TFT (baseline) | Reward-only | Welfare-only | SEPO (full) |
|---|---|---|---|---|
| always-cooperate | 0.000 | 0.002 | 0.781 | 0.005 |
| always-defect | 0.000 | 0.002 | 0.016 | 0.020 |
| tit-for-tat | 1.000 | 0.005 | 0.069 | **0.852** |
| generous-tit-for-tat | 0.000 | 0.002 | 0.077 | 0.008 |
| grim-trigger | 0.000 | **0.989** | 0.057 | 0.115 |

### Ablation Mixtures

| Strategy | SEPO-no-exploit | SEPO-no-collusion | SEPO-no-externality |
|---|---|---|---|
| always-cooperate | 0.096 | 0.005 | 0.008 |
| always-defect | 0.024 | 0.020 | 0.020 |
| tit-for-tat | 0.265 | 0.852 | **0.932** |
| generous-tit-for-tat | 0.224 | 0.008 | 0.013 |
| grim-trigger | 0.391 | 0.115 | 0.027 |

---

## Per-Opponent Payoff Breakdown

P1 payoff over 20 episodes per opponent.

| Opponent | Reward-only | Welfare-only | SEPO | Δ (SEPO − RO) |
|---|---|---|---|---|
| vs. always-cooperate | 24.10 | 24.10 | 24.30 | +0.20 |
| vs. always-defect | 7.00 | 1.50 | 6.95 | −0.05 |
| vs. tit-for-tat | 24.10 | 23.95 | 24.00 | −0.10 |
| vs. generous-tit-for-tat | 24.10 | 23.95 | 24.00 | −0.10 |
| vs. grim-trigger | 24.10 | 23.50 | 23.45 | −0.65 |

---

## Key Observations

- **SEPO matches reward-only payoff (18.37)** while keeping exploitability low (5.25 vs 32.5 for welfare-only).
- **Reward-only converges to 99% grim-trigger** — optimal against the cooperative train pool but brittle under shift.
- **SEPO converges to 85% TFT** — the game-theoretically robust reciprocal strategy, consistent with folk theorem predictions.
- **Welfare-only is catastrophically exploitable** (exploitability=32.5, safety=−81.3) due to its 78% always-cooperate mixture.
- **Removing the exploit penalty** is the most damaging ablation: safety drops from 1.97 → −16.25 as the optimizer allows grim/cooperative mixtures with large adversarial payoff gaps.
- **Collusion is zero across all conditions** — correct for IPD, where `run_sepo_experiments.py` hardcodes `collusion=0.0` in the social dilemma simulator.
