# Opponent World Models: Small LLMs as Learned Dynamics for Strategic Planning

Proposal and proof-of-concept results.
Follow-on to SEPO (Safe Equilibrium Policy Optimization).
Target: ML conference submission (NeurIPS / ICML / ICLR), with AAMAS as the better-matched alternative.

---

## 1. Abstract

**Situation.**
World models let an agent plan by imagining futures before acting, and they have driven the strongest sample-efficient RL of the last few years (Dreamer, IRIS, DIAMOND).
Essentially all of them model a single-agent, stationary environment: the dynamics are fixed physics, and the model's job is to predict what the world does next.

**Problem.**
In strategic multi-agent settings the environment's dynamics *are other adaptive agents*.
A single-agent world model treats opponents as fixed environment noise, so it cannot anticipate best-responses, retaliation, or coordination, which are the exact behaviors that make reward-optimal policies exploitable.
The hard prediction is what the other player does next, and it is non-stationary by construction because the opponent is reacting to you.

**Solution.**
Learn a small-LLM opponent world model `M` whose only job is to predict the opponent's next action and resulting state from language-anchored interaction history, then plan against rollouts of `M` before committing a real action.
This converts a hard adversarial problem ("be robust to any opponent") into a learnable predictive one ("predict *this* opponent and best-respond").
Because the model predicts agents rather than pixels, it is cheap, language-conditioned, generalizes across opponent types, and trains on a single GPU.

**The twist that makes it a paper.**
Our proof of concept shows that a planner with a *perfect* opponent model becomes dramatically **more** exploitable, not less, when it plans for payoff.
Adaptivity is a capability, not a safety property.
Strategic safety is a property of the objective you plan against, and that is the claim this project is built to defend.

---

## 2. Why this follows from SEPO, empirically

SEPO optimizes `J(π) = u(π) − λe·e(π) − λc·c(π) − λx·x(π)` via GRPO.
It works: collusion and externality fall, safety rises across five games.
But its own results show exactly where it stops, and the pattern is structural rather than a tuning failure.

| Game | Equilibrium | SEPO exploitability outcome |
|---|---|---|
| IPD | Yes, TFT | Stable at 0.312 from step 35. GRPO **ties** base. Both at the TFT floor. |
| Kuhn Poker | Yes, Nash mixed | Reaches 0.000. Nothing left to win. |
| Negotiation v1 | Partial | Structural ceiling. Base already near-optimal. |
| Negotiation v2 | Converging | 0.319 vs base 0.781. SEPO works here. |
| **Auction** | **No** | **Exploit climbs monotonically after step 25. No single strategy simultaneously minimises exploit vs AggressiveBid and maximises utility.** |

The diagnosis: **policy optimization returns the best fixed strategy, so its exploitability floor is the floor of that fixed strategy.**
Where a learnable equilibrium exists, SEPO reaches it and there is nothing further to gain.
Where no equilibrium exists, as in the auction, no fixed policy can be both non-exploitable and useful, and training cannot fix it at all.

Opponent-model planning returns something strictly larger than a fixed strategy: a *mapping* from inferred opponent to response.
That is the class of solutions the auction result says is required and policy optimization cannot express.

---

## 3. Approach

Two components: a policy that plans, and an opponent world model `M` mapping interaction history to a distribution over the opponent's next action and next state.
At each decision the agent rolls `M` forward `k` steps to score candidate actions, selects the best-responding action, and only then acts for real.

The capability objection that defeats naive robustness ("you cannot be robust to an opponent you cannot out-think") does not apply the same way, because the agent predicts the specific opponent it faces from observed behavior rather than defending against a worst case.
The gap returns through two specific doors, which become the study's measured failure axes rather than fatal flaws:

1. **Opponent-model error**, when the opponent is off-distribution or richer than the model can capture.
2. **Non-stationarity and co-adaptation**, when the opponent adapts at a comparable rate and the model chases a moving target.

---

## 4. What the proof of concept already establishes

`owm_integrated.py` runs an opponent-modeling planner inside the real SEPO simulator, on the same opponent pools, with the same exploitability metric, against policies produced by the actual SEPO optimizer.
Deliberately no LLM and no GRPO, so the mechanism is isolated.
Two planners are evaluated that differ **only** in the objective they plan against, with identical opponent model, belief update, horizon, pools and seeds.

| game | agent | pay/rnd | exploit |
|---|---|---|---|
| IPD | SEPO reward-only | 2.57 | 0.124 |
| IPD | SEPO welfare-only | 2.41 | 0.952 |
| IPD | SEPO (full) | 2.57 | 0.129 |
| IPD | OWM planner (payoff) | **2.77** | 0.125 |
| IPD | OWM planner (SEPO objective) | **2.77** | 0.125 |
| Negotiation | SEPO (full) | 1.73 | **0.000** |
| Negotiation | OWM planner (payoff) | **3.05** | **1.850** |
| Negotiation | OWM planner (SEPO objective) | 1.75 | **0.000** |

Three findings, in order of how much weight they can carry.

**Finding 1 (strong): adaptivity is not safety.**
A planner with an *exact* opponent model, planning for payoff, drives negotiation exploitability from 0.000 to 1.850 while nearly doubling payoff.
It concedes in order to close deals with greedy opponents, earning more while the opponent earns much more.
The effect is large, unambiguous, and objective-controlled.

**Finding 2 (strong): the planning objective is the fix.**
Swapping the planning objective to SEPO's returns exploitability to 0.000 at 1.75 payoff per round, marginally ahead of trained SEPO at 1.73.
Only the objective changed.
This is the cleanest available evidence that safety here is a property of what you plan against, not of the policy you trained.

**Finding 3 (weak, and it reshapes C1): there was no exploitability headroom to win.**
In IPD the planner gains payoff (2.77 vs 2.57) at statistically indistinguishable exploitability (0.125 vs 0.129).
In negotiation the model-free baseline is already at exactly 0.000.
**The PoC therefore does not yet support C1 as originally stated.**
This is consistent with SEPO's own IPD result, where GRPO ties base at the TFT floor, and it is why the auction domain moves to the front of the program.

### What the PoC does not establish

- The opponent model is **exact** (known finite type library), which assumes away prediction error, compounding, and non-stationarity, that is, the entire content of Tier 2.
- The negotiation return to 0.000 is achieved largely by **walking away**: against balanced, selfish and hardball opponents the planner deliberately offers a sum over 10 so no deal closes, and it only trades with fair and concede. That is optimal under the stated objective but it is "refuse to trade", not "negotiate better", and it is driven by a metric artifact where accepting a rational unequal split scores as being exploited.
- Single seed, no confidence intervals. The exploitability differences of 0.001 to 0.005 are almost certainly below seed variance.
- Tabular, no LLM, no GRPO.

---

## 5. Claims (falsifiable)

**C1 (revised).** Exploitability reduction from opponent-model planning is available exactly when no single fixed strategy is simultaneously non-exploitable and high-utility.
Where a learnable equilibrium exists (IPD to TFT, Kuhn to Nash), model-free SEPO already reaches the floor and planning yields no exploitability gain, though it may still yield payoff gain at equal exploitability.
Where no such equilibrium exists (auction), planning reduces exploitability below the model-free floor.
IPD and Kuhn are the negative controls, not gaps.

**C2 (headline).** With a learned, imperfect opponent model the gain persists up to a planning horizon `k` where compounding prediction error overtakes it, defining an optimal `k`.
The error-versus-horizon curve is itself the result, and `k` should scale down with opponent complexity and with model error rate.

**C3 (boundary).** The gain collapses, or inverts below model-free, under co-adaptation, when the opponent adapts at a rate comparable to the agent.
This quantifies the non-stationarity limit and marks the boundary with the multi-agent learning dynamics literature.

**C4 (safety, already supported).** Opponent-model planning is not safety-neutral.
Under a payoff objective it increases exploitability; under the SEPO objective it does not.
Safety in strategic settings is a property of the planning objective.

### Conditions

Model-free GRPO (`k`=0 baseline), fixed opponent-type classifier plus best-response (static-belief baseline), learned opponent-model planner (realistic), and an oracle planner with the true opponent policy (upper bound).
Sweep planning horizon `k`, opponent history-dependence, opponent-model error rate, and stationary versus co-adapting opponents.

---

## 6. Program to submission

### T0. Tabular, no GPU. Two to three weeks.

1. **Auction domain in the PoC.** The decisive C1 test, in the one game where the headroom is structurally unavailable to any fixed policy. The tabular substrate already has the simulator, strategy library and shift pools, so this is a small extension rather than a new build.
2. **Opponent-model error sweep.** Put a tunable error rate on the tabular opponent model and sweep planning depth against it. This produces the C2 curve without a GPU, and converts the PoC's weakest assumption (exact model) into its headline characterization result.
3. **Seeds and confidence intervals** on both axes, since the current IPD claim rests on differences below noise.
4. **Negotiation metric fix** so that accepting a rational unequal split is not scored as exploitation, then re-run. If the safety planner still walks away, that is a real result. If it starts closing unequal but positive deals, the current negotiation panel overstates the fix.

T0 is cheap and it decides whether the project has a headline. If auction shows no headroom and the error sweep shows optimal `k`=1, the paper is thin and worth reconsidering before any GPU time is spent.

### T1. Learned opponent predictor plus GRPO, single 4090. Six to eight weeks.

Small-LLM opponent predictor trained on language-anchored interaction history, planning loop inside the existing GRPO pipeline, C1 and C2 re-measured under real prediction error.
Compute risk is real and is named in section 8.

### T2. LLM opponents and co-adaptation. Four weeks plus writing.

Full GTBench or LLM-versus-LLM negotiation where the opponent is not in any finite library.
C3 measured directly.

Roughly four to five months to a submission-ready paper, which fits one conference cycle with margin.

### Venue note

In this space the top venues are conferences rather than journals: NeurIPS, ICML, ICLR.
AAMAS is the better-matched venue for opponent modeling and exploitability, because reviewers already know LOLA, I-POMDPs and ToMnet and the novelty argument becomes specific rather than defensive.
TMLR is a strong journal fallback with rolling submission and no page limit, and it judges correctness and evidence rather than novelty, which suits a characterization result like C2.
A main-track ML conference paper needs C1 and C2 complete with T2 at least started.

---

## 7. Positioning

Every neighbor misses a different piece.
Single-agent world models (Dreamer, IRIS, DIAMOND) treat other agents as static environment.
MuZero plans with a learned model but via tree search over its own value, not a generative opponent model.
LOLA incorporates opponent learning but assumes access to opponent gradients or parameters rather than a model learned from behavior alone.
ToMnet is the closest neighbor and is prediction-only: it infers latent types from trajectories but does not roll the agent-model forward to plan a best response, is not a language-conditioned generative model over structured game state, and is not evaluated on exploitability reduction.

None of them (i) uses a small LLM as a language-conditioned generative opponent-dynamics model, (ii) uses it for multi-step best-response planning aimed at reducing exploitability, or (iii) characterizes the co-adaptation breakdown as a measured failure axis.
That intersection is the space this proposal occupies.

The honest framing for reviewers: opponent modeling itself is old.
The contributions are the safety result (C4), the structural account of when modeling can and cannot help (C1), and the horizon-versus-model-error characterization (C2).

---

## 8. Risks and kill criteria

| Risk | Detection | Response |
|---|---|---|
| No exploitability headroom anywhere, including auction | T0 step 1 | C1 dies. The payoff-at-equal-exploitability result and C4 survive as a smaller paper. Reconsider before T1. |
| Learned predictor too weak, optimal `k`=1 | T0 step 2, confirmed T1 | C2 becomes a thin result. Pivot the headline to C4. |
| Compute blowup from nested rollouts | T1 start | GRPO already runs fresh exploit and collusive episodes per rollout. Adding `k` planning rollouts per real step multiplies that. Mitigate with small `k`, cached rollouts, and keeping the horizon sweep tabular. |
| "Opponent modeling is old" | Review | Lead with C4 and C2, not C1. |
| Negotiation result is a metric artifact | T0 step 4 | Fix the metric or drop the panel. |

---

## 9. Open questions

- How far can a learned opponent model see before compounding error dominates, and how does optimal `k` scale with opponent complexity and partial observability?
- Does the gain transfer from finite-library opponents to LLM opponents?
- Can planning stay stable under co-adaptation, or is this fundamentally limited to slowly-adapting opponents?
- What must be predicted: opponent action, opponent latent type, or full next state? In hidden-type games such as Kuhn poker, does partial observability degrade gracefully or sharply?
- Cost versus benefit. Planning costs `k` opponent-model rollouts per real step. Is the exploitability gain worth the inference overhead relative to training a more robust model-free policy against a larger opponent pool?

---

## 10. Dual-use note

A strong generative opponent model that best-responds is, by symmetry, a manipulation tool.
The same machinery that reduces one's own exploitability can be pointed at a target to exploit its predictable behavior, which is directly relevant to the LLM-collusion and market-manipulation setting.
The deliverable must be the defensive framing: exploitability reduction, and the measurement of where opponent-model planning fails.
No manipulation-optimized policy is released, and any exploiter component stays small and closed.

---

## 11. References

Citations below are carried over from the earlier draft and **have not been verified against the literature in this session**.
Confirm every arXiv ID and venue before submission, and run a fresh search for 2025 to 2026 LLM-agent world-model work, which is moving quickly and is not covered here.

- Ha & Schmidhuber. *World Models.* arXiv:1803.10122 (2018). Origin of learned-dynamics-for-planning, the single-agent template this extends.
- Hafner et al. *Mastering Diverse Domains through World Models* (DreamerV3). arXiv:2301.04104 (2023).
- Schrittwieser et al. *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero). Nature (2020), arXiv:1911.08265.
- Micheli et al. *Transformers are Sample-Efficient World Models* (IRIS). ICLR 2023, arXiv:2209.00588. The single-GPU precedent for tiny world models.
- Rabinowitz et al. *Machine Theory of Mind* (ToMnet). ICML 2018, arXiv:1802.07740. Closest neighbor.
- Foerster et al. *Learning with Opponent-Learning Awareness* (LOLA). AAMAS 2018, arXiv:1709.04326.
- Chae et al. *Web Agents with World Models.* ICLR 2025. (verify ID)
- Ma et al. *LLM-Based World Models Can Make Decisions Solely, But Rigorous Evaluations Are Needed.* arXiv:2411.08794 (2024). (verify)
- Lin, Ojha, Cai & Chen. *Strategic Collusion of LLM Agents.* arXiv:2410.00031 (2024). (verify)
- Quevedo et al. *WorldGym: World Models as Environments for Policy Evaluation* (2025). (verify ID)
- SEPO (this group, prior work). Supplies the exploitability metric, opponent pools and strategy libraries reused as the Tier-1 substrate.
