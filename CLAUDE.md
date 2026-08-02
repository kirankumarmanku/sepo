# CLAUDE.md — Opponent World Models (SEPO follow-on)

This file orients a terminal session. It summarizes an ongoing research thread
so work can continue without the original chat transcript.

## What this project is

A next research direction building on **SEPO** (Safe Equilibrium Policy
Optimization). SEPO augments task reward with penalties for exploitability,
collusion risk, and externality cost, trained via GRPO on small (~4B) LLMs in
strategic games (IPD, auction, negotiation, Kuhn poker, resource). SEPO's honest
open limitation: **exploitability** is the axis it does not solve.

**The new idea — Opponent World Models (OWM).** An agent that learns a model of
its opponent(s) from interaction history and *plans* a best response against
model rollouts, targeting exploitability directly. This is the multi-agent
version of a world model: in strategic settings the "environment dynamics" are
other adaptive agents, and predicting what they do next is the hard part.
Proposal: `owm_proposal.md` (full pitch, PoC results, program to an ML conference
submission). Note the earlier `opponent_world_models_onepager.md` referenced here
never existed in the repo.

## Current state — the PoC (done)

A pitch-stage proof of concept, deliberately with **no LLM and no GRPO** — it
isolates the *mechanism* on the tabular SEPO simulator so the argument rests on
one clean figure.

- `owm_poc.py` — standalone minimal PoC (IPD), fixed baselines vs planner.
- `owm_integrated.py` — **main artifact.** Imports `run_sepo_experiments.py`,
  runs the real optimizer to get trained reward-only / welfare-only / SEPO
  policies, and evaluates an opponent-modeling planner against them **inside the
  SEPO simulator, on the same opponent pools, with the same exploitability
  metric**, across two games (IPD + Negotiation).

Run: `python owm_integrated.py` → `owm_integrated.png` + printed table.

Two planners are evaluated, differing ONLY in the objective they plan against:
`payoff` (raw own payoff) and `sepo` (payoff − λe·exploit − λc·collusion −
λx·externality).
The λ's are recovered by probing `sepo.objective_value` rather than restated, so
the planner and the SEPO optimizer cannot drift apart.
Everything else (opponent model, belief update, horizon, pools, seeds) is
identical, so the gap between the two points isolates the objective.

### Key finding (this is the pitch)

- **IPD:** planner Pareto-dominates the trained SEPO policies — higher payoff
  (~2.77 vs ~2.57/round) at equal exploitability (~0.125). Earned against a
  strong baseline (SEPO reward-only is ~93% tit-for-tat).
  The safety-aware planner lands on the *same point*: the payoff-greedy plan is
  already unexploitable here, so the objective swap costs nothing.
- **Negotiation:** payoff-greedy planner nearly doubles payoff (~3.05 vs ~1.73)
  but exploitability spikes (~1.85 vs 0.0). A *payoff-greedy* planner concedes to
  close deals with greedy opponents — more payoff, but the opponent out-earns it.
  The safety-aware planner returns to **0.0 exploitability at ~1.75/round**,
  marginally ahead of trained SEPO (~1.73) on the same axis.
- **Takeaway:** opponent modeling gives adaptivity; it does NOT automatically
  give safety. It is **complementary to SEPO, not a replacement**. Modeling +
  the SEPO objective is the project.

Caveats to keep honest: the negotiation exploitability metric conflates "beaten"
with "accepted a rational unequal split"; and the planner's opponent model here
is *exact* (known type library) — a learned model is where prediction error and
the optimal-horizon question enter.
A third caveat now matters most.
The safety-aware planner reaches 0.0 exploitability in negotiation largely by
**walking away**: against `balanced`/`selfish`/`hardball` it deliberately offers
a sum over 10 so no deal closes (0/0), and it only trades with `fair` and
`concede`.
That is optimal under the stated objective, but it is "refuse to trade", not
"negotiate better", and it is driven by the same metric artifact above.
Worth fixing the metric before leaning on this panel in a pitch.

## Immediate next step

Fix the negotiation exploitability metric so that accepting a rational unequal
split is not scored as being exploited, then re-run.
If the safety-aware planner still walks away under a corrected metric, that is a
real result; if it starts closing unequal-but-positive deals, the current
negotiation panel is overstating the fix.

## Then (the real project, Tier 2)

Replace the exact tabular opponent model with a **small-LLM opponent predictor**
(predict opponent next action from language-anchored history), train the planner
with the existing **GRPO** loop, and measure:
- C1: exploitability reduction vs model-free, growing with opponent history-dependence.
- C2: optimal planning horizon k where compounding model error overtakes the gain.
- C3: degradation under co-adaptation (non-stationary opponents).

## Environment / constraints

- Compute: single RTX 4090 (RunPod). Models: Gemma 3 4B, Qwen3.5-4B.
- Stack: trl, transformers, GRPO custom pipeline, LoRA; GTBench for eval.
- `run_sepo_experiments.py` is the tabular substrate (strategy libraries,
  opponent pools, exploitability = opponent-advantage/round). Reuse it.

## Conventions

- Keep PoC dependencies minimal (numpy, matplotlib); it must stay runnable
  without GPU.
- Prefer results as chart + table together.
- Exploitability metric = `max(0, opp_payoff - own_payoff) / rounds` (SEPO's def).
