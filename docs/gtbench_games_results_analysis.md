# Gemma-3-4B on GTBench — Prompt vs CoT Analysis

**Model:** google/gemma-3-4b-it | **Backend:** Ollama | **Games:** 5 | **Reasoning:** Prompt & CoT

---

## 1. Summary — NRA across all games

Normalised Relative Advantage (NRA) is the primary GTBench metric. Range: −1 (dominated) to +1 (dominates). CoT improves average NRA from −0.144 to −0.068, a net gain of +0.076.

| Game | Prompt NRA | CoT NRA | Δ CoT−Prompt | Verdict |
|---|---|---|---|---|
| IPD | −0.479 | +0.193 | +0.672 | CoT better |
| Kuhn Poker | +1.000 | +1.000 | 0.000 | Tie |
| Blind Auction | −0.725 | −1.000 | −0.275 | CoT worse |
| Negotiation | −0.134 | +0.017 | +0.151 | CoT better |
| Pig | −0.383 | −0.550 | −0.167 | CoT worse |
| **Average** | **−0.144** | **−0.068** | **+0.076** | **CoT net better** |

- CoT improves in 3 of 5 games: IPD (+0.672), Negotiation (+0.151), and Kuhn Poker (profit improves despite equal NRA).
- CoT hurts in 2 games: Blind Auction (−0.275) and Pig (−0.167).
- Kuhn Poker is the only game where both modes fully dominate the opponent (NRA = +1.000).
- Blind Auction is the weakest game for both modes — both fail to grasp bid shading below private value.

---

## 2. Iterated Prisoner's Dilemma (IPD)

**Setup:** 8 rounds/episode, 20 episodes/opponent. Payoffs: T=5, R=3, P=1, S=0. Conventional agent: Tit-for-Tat. Safety index: S(π) = u(π) − 3e(π) − 4c(π) − 2x(π).

| Model | Payoff ↑ | Exploit ↓ | Robust ↑ | Safety ↑ | Coop rate |
|---|---|---|---|---|---|
| Prompt | 14.317 | 40.000 | 24.500 | −105.986 | 81.5% |
| CoT | 17.733 | 0.000 | 18.500 | +16.951 | 50.6% |
| Δ CoT−Prompt | +3.416 | −40.000 | −6.000 | +122.937 | −30.9pp |

- CoT eliminates exploitability entirely (40.0 → 0.0) by learning to retaliate against defectors — the core TFT insight — without any strategic training.
- Safety index swings from −105.99 to +16.951, the single largest CoT improvement across all 5 games and the best safety score of any condition including rule-based SEPO policies.
- Prompt Gemma's 81.5% cooperation rate confirms near-always-cooperate behaviour, making it the easiest possible target for an exploiter (earns 0 pts vs always-defect).
- CoT robustness (18.5) drops below prompt (24.5), indicating it generalises less cleanly to held-out opponents in the shift pool — the gap SEPO training would be expected to close.
- IPD is the game where CoT reasoning is most impactful because the key insight (reciprocity) is qualitative and well-represented in language model training data.

---

## 3. Kuhn Poker

**Setup:** 3 cards (Jack, Queen, King). 20 games per opponent, both positions (80 total). Opponents: Nash equilibrium and random. Nash optimal bluff rate with Jack = 1/3 ≈ 0.333.

| Model | Win rate ↑ | Avg profit ↑ | NRA ↑ | Bluff rate | vs Nash (0.333) |
|---|---|---|---|---|---|
| Prompt | 0.600 | 0.150 | +1.000 | 0.237 | −0.096 off |
| CoT | 0.675 | 0.263 | +1.000 | 0.325 | −0.008 off |
| Δ CoT−Prompt | +0.075 | +0.113 | 0.000 | +0.088 | — |

- Both prompt and CoT achieve NRA +1.000 — Gemma dominates all opponents regardless of reasoning mode, making this its strongest game overall.
- CoT bluff rate (0.325) lands almost exactly on the Nash optimal (0.333), suggesting chain-of-thought reasoning independently rediscovers the mixed-strategy equilibrium without being given it explicitly.
- Prompt underbluffs at 0.237, leaving value on the table by being too predictable when holding a Jack — opponents can exploit this by folding more to any bet.
- Neither mode ever folds (fold rate 0.000), which is correct at Nash equilibrium for this simple card structure — zero fold rate is a sign of good play, not passivity.
- The shallow decision tree (at most 3 sequential actions per game) likely explains why even prompt mode does well — there is not enough complexity to expose deep reasoning gaps.

---

## 4. Blind Auction

**Setup:** First-price sealed-bid auction. Private values uniform [1, 10]. 6 rounds/episode, 20 episodes/opponent (Nash, truthful, random). Nash optimal bid = value / 2.

| Model | Avg profit ↑ | Win rate ↑ | NRA ↑ | Bid shading ↑ | Overbid rate ↓ | Parse fails |
|---|---|---|---|---|---|---|
| Nash optimal | ~0.50 | ~0.50 | 0.000 | 0.500 | 0.000 | — |
| Prompt | −1.150 | 0.756 | −0.725 | −0.997 | 0.528 | 0.000 |
| CoT | −0.261 | 0.411 | −1.000 | +0.054 | 0.225 | 0.031 |
| Δ CoT−Prompt | +0.889 | −0.345 | −0.275 | +1.051 | −0.303 | +0.031 |

- Prompt overbids 52.8% of rounds (shading ratio −0.997) — it systematically bids above private value, guaranteeing negative expected profit (−1.150 per round).
- CoT partially corrects overbidding: rate drops to 22.5% and shading turns slightly positive (+0.054), but Nash optimal is 0.500 — CoT still bids far too high.
- Prompt's higher win rate (0.756 vs 0.411) is misleading — it wins more auctions precisely because it overbids, but at a steep profit loss. Win rate is not a reliable metric here.
- CoT NRA (−1.000) is worse than prompt (−0.725) despite better shading, because both agent and opponent profits compress near zero, making relative advantage floor out.
- This is the clearest case where CoT reasoning is insufficient — the correct strategy requires computing expected profit across all possible opponent bids, a quantitative calculation Gemma cannot reliably execute.

---

## 5. Negotiation

**Setup:** Multi-turn bargaining over 3 books, 2 hats, 2 balls with private item values. Up to 5 rounds per game, 20 episodes per opponent (greedy and fair). Disagreement = 0 points for both.

| Model | Avg score ↑ | Deal rate ↑ | NRA ↑ | Fairness ↑ | Parse fails | n games |
|---|---|---|---|---|---|---|
| Prompt | 4.925 | 1.000 | −0.134 | 0.664 | 0.000 | 40 |
| CoT | 6.100 | 0.975 | +0.017 | 0.691 | 0.000 | 40 |
| Δ CoT−Prompt | +1.175 | −0.025 | +0.151 | +0.027 | 0.000 | — |

- CoT is the clear winner: score improves 24% (4.925 → 6.100) and NRA flips from negative (−0.134) to slightly positive (+0.017).
- Deal rate stays near-perfect for both (100% prompt, 97.5% CoT) — Gemma never walks away from a deal, which is correct when disagreement yields zero for both parties.
- Near-zero CoT NRA (+0.017) means Gemma negotiates approximately equal splits — cooperative but not extracting surplus even from the greedy opponent.
- Fairness index improves marginally with CoT (0.664 → 0.691), suggesting CoT makes more balanced counter-proposals rather than anchoring on its own values.
- Multi-turn dialogue is where CoT adds the most value — the model can reason about what items the opponent likely values before proposing, which is qualitative opponent modelling rather than precise calculation.

---

## 6. Pig (dice game)

**Setup:** Roll to accumulate, bank with HOLD, lose turn on a 1. First to 100 wins. 20 games/opponent, both positions (120 total). Opponents: hold-at-20, hold-at-25, random. Optimal hold threshold ≈ 18.

| Model | Win rate ↑ | Avg score ↑ | NRA ↑ | Avg turn score | Parse fails | n games |
|---|---|---|---|---|---|---|
| Hold-at-20 (optimal) | ~0.50 | — | 0.000 | ~18 | — | — |
| Prompt | 0.308 | 48.9 | −0.383 | 40.469 | 0.000 | 120 |
| CoT | 0.225 | 56.6 | −0.550 | 14.895 | 0.000 | 120 |
| Δ CoT−Prompt | −0.083 | +7.7 | −0.167 | −25.574 | 0.000 | — |

- Prompt rolls far too aggressively — avg turn score 40.469 is more than double the optimal ~18, essentially playing always-roll and getting pigged repeatedly (win rate 30.8%).
- CoT overcorrects in the opposite direction — avg turn score 14.895 means holding too early, never accumulating enough, and win rate drops further to 22.5% (NRA −0.550).
- Both modes are wrong but in opposite directions: prompt has no risk aversion, CoT has too much.
- The 7× gap between prompt and CoT turn scores (40.5 vs 14.9) is the largest behavioural divergence across any single metric in the entire experiment.
- Pig requires precise multi-variable expected-value reasoning (current score, opponent score, roll probability) — exactly the type of quantitative calculation where CoT reasoning is insufficient.
- CoT avg score per game is higher (56.6 vs 48.9) because it accumulates score safely per turn, but cannot overcome opponents holding at more aggressive thresholds.

---

## 7. Overall conclusions

CoT improves average NRA from −0.144 to −0.068 (+0.076) but gains are highly game-dependent. Results split cleanly by reasoning type required:

### CoT helps: qualitative strategic reasoning

- **IPD:** Reciprocity (retaliate after defection) is a qualitative insight well-represented in language training data. CoT recovers near-TFT behaviour with zero exploitability.
- **Kuhn Poker:** Bluff calibration is directional ("bet more with strong hands, occasionally bluff weak ones"). CoT finds near-Nash bluff rate independently.
- **Negotiation:** Opponent modelling ("what does the other side likely value?") benefits from explicit reasoning steps before proposing.

### CoT hurts: precise quantitative calculation

- **Blind Auction:** Expected profit maximisation requires integrating over all possible opponent bids — a calculation that CoT attempts but executes incorrectly, producing worse NRA than prompt.
- **Pig:** Optimal hold threshold is a multi-variable risk calculation that CoT over-solves, inducing excessive caution and lower win rate than prompt.

### Key finding

IPD CoT safety index (+16.951) exceeds every rule-based SEPO condition (best: TFT at +2.689) without any strategic training. This demonstrates that reasoning alone can recover near-optimal reciprocal behaviour in repeated social dilemmas — directly supporting the SEPO paper's argument that richer training objectives would further improve LLM strategic behaviour.
