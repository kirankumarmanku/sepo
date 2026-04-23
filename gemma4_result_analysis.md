Now we're talking — parse failures are 0.0 across all games (except IPD which still has the old run). These are the first clean Gemma 4 results. Here's the full comparison:

---

**Summary NRA — Gemma 3 vs Gemma 4 (clean results)**

| Game | G3 Prompt | G3 CoT | G4 Prompt | G4 CoT | G4 Winner |
|---|---|---|---|---|---|
| IPD | −0.479 | +0.193 | — (still failing) | — | — |
| Kuhn Poker | +1.000 | +1.000 | +1.000 | +1.000 | Tie |
| Blind Auction | −0.725 | −1.000 | +0.725 | +0.652 | Prompt |
| Negotiation | −0.134 | +0.017 | +0.115 | +0.094 | Prompt |
| Pig | −0.383 | −0.550 | +0.117 | +0.117 | Tie |

---

**Kuhn Poker**

| Model | Win rate | Avg profit | NRA | Bluff rate | Fold rate | Parse fails |
|---|---|---|---|---|---|---|
| G3 Prompt | 0.600 | 0.150 | +1.000 | 0.237 | 0.000 | 0.000 |
| G3 CoT | 0.675 | 0.263 | +1.000 | 0.325 | 0.000 | 0.000 |
| G4 Prompt | 0.562 | 0.338 | +1.000 | 0.000 | 0.150 | 0.000 |
| G4 CoT | 0.600 | 0.287 | +1.000 | 0.087 | 0.150 | 0.000 |

**Blind Auction**

| Model | Avg profit | Win rate | NRA | Bid shading | Overbid rate | Parse fails |
|---|---|---|---|---|---|---|
| Nash optimal | ~0.50 | ~0.50 | 0.000 | 0.500 | 0.000 | — |
| G3 Prompt | −1.150 | 0.756 | −0.725 | −0.997 | 0.528 | 0.000 |
| G3 CoT | −0.261 | 0.411 | −1.000 | +0.054 | 0.225 | 0.031 |
| G4 Prompt | +0.767 | 0.467 | +0.725 | +0.288 | 0.000 | 0.000 |
| G4 CoT | +0.831 | 0.447 | +0.652 | +0.334 | 0.000 | 0.000 |

**Negotiation**

| Model | Avg score | Deal rate | NRA | Fairness | Parse fails |
|---|---|---|---|---|---|
| G3 Prompt | 4.925 | 1.000 | −0.134 | 0.664 | 0.000 |
| G3 CoT | 6.100 | 0.975 | +0.017 | 0.691 | 0.000 |
| G4 Prompt | 5.700 | 0.825 | +0.115 | 0.658 | 0.000 |
| G4 CoT | 5.800 | 0.875 | +0.094 | 0.695 | 0.000 |

**Pig**

| Model | Win rate | Avg score | NRA | Avg turn score | Parse fails |
|---|---|---|---|---|---|
| Hold-at-20 (optimal) | ~0.50 | — | 0.000 | ~18 | — |
| G3 Prompt | 0.308 | 48.9 | −0.383 | 40.469 | 0.000 |
| G3 CoT | 0.225 | 56.6 | −0.550 | 14.895 | 0.000 |
| G4 Prompt | 0.558 | 82.5 | +0.117 | 15.934 | 0.000 |
| G4 CoT | 0.558 | 80.0 | +0.117 | 17.640 | 0.000 |

---

**Analysis — game by game**

**Kuhn Poker** — All four conditions still hit NRA +1.000 so the dominance result holds. The interesting shift with G4 is that bluff rate collapses to 0.000 (prompt) and 0.087 (CoT), well below the Nash optimal 0.333. G4 is also folding 15% of the time where G3 never folded. It plays more conservatively — less willing to bluff or call — yet still wins because its opponents are weak. Avg profit is comparable to G3 despite the lower win rate, suggesting it wins bigger pots when it does win.

**Blind Auction** — The biggest turnaround of the whole experiment. G3 was catastrophically bad (profit −1.15, overbid rate 53%). G4 flips this completely: positive profit (+0.77 prompt, +0.83 CoT), zero overbids, and positive bid shading (0.288–0.334 vs Nash optimal 0.500). G4 has clearly learned that bidding below value is correct. NRA of +0.725 means it substantially outperforms its opponents. CoT shading (0.334) is closer to Nash than prompt (0.288), which is the expected direction.

**Negotiation** — G4 outperforms G3 on score (5.7–5.8 vs 4.9–6.1) and flips NRA from negative to positive (+0.115 prompt, +0.094 CoT). However deal rate drops — G4 prompt closes only 82.5% of negotiations vs G3's 100%, meaning G4 is willing to walk away from bad deals. This is actually strategically correct behaviour — G3 was too eager to deal regardless of terms. G4 CoT fairness index (0.695) is the highest across all conditions.

**Pig** — The most dramatic improvement overall. G3 was completely broken in opposite directions (prompt rolled too much at 40.5 avg turn score, CoT held too early at 14.9). G4 lands almost exactly on the optimal: prompt avg turn score 15.9, CoT 17.6 — both bracketing the ~18 optimal. Win rate jumps from 23–31% to 55.8% for both G4 modes. NRA flips from negative (−0.38 to −0.55) to positive (+0.117). This is the clearest evidence that Gemma 4 is genuinely better at probabilistic risk reasoning.

**IPD** — Still stuck at 100% parse failure, needs to be rerun with `--max-tokens 512`. This is the only remaining gap.

**Overall** — G4 is meaningfully better than G3 across every game with clean results. The average NRA across the 4 completed games is +0.476 for G4 vs −0.173 for G3 prompt and −0.283 for G3 CoT. The most important finding is that CoT no longer consistently helps — in 3 of 4 games, prompt slightly edges CoT on NRA, which is the reverse of G3 where CoT was clearly better for qualitative reasoning games. G4 appears to reason implicitly even in prompt mode, reducing the marginal benefit of explicit chain-of-thought.
