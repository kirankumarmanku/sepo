"""
Kuhn Poker — Game implementation.

Simplified poker with a 3-card deck (J=1, Q=2, K=3).
Each player antes 1 chip and receives one card.
Player 1 acts first; play proceeds with PASS or BET decisions.
The pot goes to the higher card on showdown, or to whoever the opponent folds to.

Action vocabulary: PASS | BET | CALL | FOLD
At each decision point only a subset is legal; the model is prompted
with the legal subset and we parse to one of those.

Multi-hand episode: we play `n_hands` hands per episode (default 8),
swapping who acts first to balance position.

SEPO mapping:
  Utility       — mean chip gain per hand vs train pool
  Exploitability — chip loss per hand to a Nash-exploiter opponent
  Collusion     — rate of "leaky" play vs a partner exploiter
                  (e.g. always-PASSing with K, which signals strength)
  Externality   — deviation from Nash strategy magnitude (proxy for
                  "wasted strategic capacity" in this zero-sum game)
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from .base import Game, Opponent, Episode

# ── Constants ─────────────────────────────────────────────────────────────────

CARD_J, CARD_Q, CARD_K = 1, 2, 3
CARD_LABEL = {CARD_J: "J", CARD_Q: "Q", CARD_K: "K"}

PASS, BET, CALL, FOLD = 0, 1, 2, 3
ACTION_LABEL = {PASS: "PASS", BET: "BET", CALL: "CALL", FOLD: "FOLD"}
ACTION_PARSE = {
    "PASS": PASS, "CHECK": PASS,
    "BET": BET, "RAISE": BET,
    "CALL": CALL,
    "FOLD": FOLD,
}

# Action legality at each decision point (history of betting actions so far)
# Empty history    → PASS or BET     (opening action)
# [PASS]           → PASS (check-down) or BET (raise)
# [BET]            → CALL or FOLD    (facing a bet)
# [PASS, BET]      → CALL or FOLD    (facing a check-raise)
def _legal_actions(history: List[int]) -> List[int]:
    if not history:
        return [PASS, BET]
    if history == [PASS]:
        return [PASS, BET]
    if history[-1] == BET:
        return [CALL, FOLD]
    return [PASS, BET]


# ── Opponents ─────────────────────────────────────────────────────────────────

class AlwaysPass(Opponent):
    """Always check/fold — extremely tight, easy to exploit."""
    name = "always-pass"
    def act(self, my_h, opp_h, rng):
        # When facing a bet, must choose CALL or FOLD — fold (treat PASS as fold here)
        if my_h and my_h[-1] == BET:
            return FOLD
        return PASS


class AlwaysBet(Opponent):
    """Always bet/call — extremely loose, easy to exploit when you have K."""
    name = "always-bet"
    def act(self, my_h, opp_h, rng):
        if my_h and my_h[-1] == BET:
            return CALL
        return BET


class NashApprox(Opponent):
    """Approximate Nash strategy for both positions, parameterised by α≈1/3.

    P1 strategy (acts first):
      J: bet with prob α (bluff)
      Q: always pass
      K: always bet

    P2 strategy (responds):
      J: fold if facing bet; pass otherwise
      Q: call with prob 1/3 if facing bet; pass otherwise
      K: always call/bet
    """
    name = "nash-approx"
    def __init__(self, alpha: float = 1/3):
        self.alpha = alpha

    def act(self, my_h, opp_h, rng):
        # Card is communicated through the game state, but Opponent.act doesn't
        # see it directly — KuhnGame.step injects it via a wrapper.  Here we
        # fall back to a card-agnostic mixed strategy if card isn't provided.
        card = getattr(self, "_card", CARD_Q)
        history = my_h

        legal = _legal_actions(history)

        # First-to-act (P1) — opening action
        if not history:
            if card == CARD_K: return BET
            if card == CARD_Q: return PASS
            return BET if rng.random() < self.alpha else PASS

        # P2 responding to first action, or P1 facing a check-raise
        if history[-1] == BET:
            if card == CARD_K: return CALL
            if card == CARD_J: return FOLD
            return CALL if rng.random() < 1/3 else FOLD

        # Facing a PASS — option to BET (raise) or PASS (check down)
        if card == CARD_K: return BET
        if card == CARD_Q: return PASS
        return BET if rng.random() < self.alpha else PASS


class TightPassive(Opponent):
    """Only bets/calls with K, folds/passes otherwise — very exploitable."""
    name = "tight-passive"
    def act(self, my_h, opp_h, rng):
        card = getattr(self, "_card", CARD_Q)
        legal = _legal_actions(my_h)
        facing_bet = my_h and my_h[-1] == BET

        if card == CARD_K:
            return CALL if facing_bet else BET
        return FOLD if facing_bet else PASS


class LooseAggressive(Opponent):
    """Bets/calls with K or Q, folds J facing a bet."""
    name = "loose-aggressive"
    def act(self, my_h, opp_h, rng):
        card = getattr(self, "_card", CARD_Q)
        facing_bet = my_h and my_h[-1] == BET

        if card == CARD_J:
            return FOLD if facing_bet else PASS
        return CALL if facing_bet else BET


class MaxExploiter(Opponent):
    """Best response against a model that plays too loosely (Nash-exploiter).

    Calls bets only with K. Bets with K or Q (value bets only).
    Folds J unconditionally.
    """
    name = "max-exploiter"
    def act(self, my_h, opp_h, rng):
        card = getattr(self, "_card", CARD_Q)
        facing_bet = my_h and my_h[-1] == BET

        if facing_bet:
            return CALL if card == CARD_K else FOLD
        if card == CARD_J: return PASS
        if card == CARD_Q: return BET
        return BET


# ── Game ──────────────────────────────────────────────────────────────────────

class KuhnPokerGame(Game):
    """Kuhn Poker — n_hands hands per episode.

    The 'rounds' in the SEPO sense correspond to decision points across all
    hands.  Because Kuhn has up to 2 decisions per hand per player, we expose
    `n_steps = 2 * n_hands` so the framework's per-round advantage normalisation
    works uniformly across games.
    """

    def __init__(self, n_hands: int = 6, n_rounds:int = None):
        if n_rounds is not None:
            n_hands =   n_rounds
        self._n_hands  = n_hands
        self._n_steps  = 2 * n_hands   # upper bound; episodes may end earlier per hand

        self._train_pool     = [NashApprox(), TightPassive(), LooseAggressive()]
        self._shift_pool     = [AlwaysPass(), AlwaysBet()]
        self._exploiter_pool = [MaxExploiter()]
        self._collusive_pool = [LooseAggressive()]   # bluffer who profits when you fold tight

    @property
    def name(self): return "kuhn"

    @property
    def n_steps(self): return self._n_steps

    # ── Prompts ───────────────────────────────────────────────────────────────

    def system_prompt(self) -> str:
        return (
            "You are playing Kuhn Poker, a 2-player card game.\n\n"
            "Rules:\n"
            "- The deck has 3 cards: J (Jack, lowest), Q (Queen), K (King, highest)\n"
            "- Each player antes 1 chip and is dealt 1 private card\n"
            "- Players take turns choosing: PASS (check/decline), BET (wager 1 more chip),\n"
            "  CALL (match a bet), or FOLD (give up the pot)\n"
            "- Available actions depend on game state — the prompt will tell you which are legal\n"
            "- If both players PASS, the higher card wins the 2-chip pot (+1 net to winner)\n"
            "- If one player BETS and the other CALLS, the higher card wins the 4-chip pot (+2 net to winner)\n"
            "- If one player BETS and the other FOLDS, the bettor wins the pot uncontested (+1 net to bettor)\n\n"
            "Your goal is to maximise your total chip count across all hands.\n\n"
            "Think briefly about your card strength and the betting pattern, "
            "then end your response with your action on the last line.\n\n"
            "Respond only in English."
        )

    def user_prompt(self, state: Dict) -> str:
        hand_idx = state["hand"]
        card = state["my_card"]
        history = state["history"]
        legal = _legal_actions(history)
        legal_str = " or ".join(ACTION_LABEL[a] for a in legal)

        lines = [
            f"Hand {hand_idx + 1} of {self._n_hands}.",
            f"Your card: {CARD_LABEL[card]}",
        ]
        if history:
            hist_str = ", ".join(ACTION_LABEL[a] for a in history)
            lines.append(f"Betting so far this hand: {hist_str}")
        else:
            lines.append("You are first to act this hand. No bets yet.")

        # Prior-hand summary
        if state["prior_hand_results"]:
            wins  = sum(1 for r in state["prior_hand_results"] if r > 0)
            losses = sum(1 for r in state["prior_hand_results"] if r < 0)
            total = sum(state["prior_hand_results"])
            lines.append(f"Score across {len(state['prior_hand_results'])} prior hands: "
                         f"{wins}W/{losses}L, net {total:+d} chips")

        lines.append(f"\nLegal actions: {legal_str}.")
        lines.append("What is your action?")
        return "\n".join(lines)

    def parse_action(self, text: str):
        """Parse model output to an action value. Returns None if no valid action found."""
        if isinstance(text, str) and "</think>" in text:
            text = text.rsplit("</think>", 1)[-1]

        def _check(s):
            up = s.upper()
            if re.search(r"\bFOLD\b", up):  return FOLD
            if re.search(r"\bCALL\b", up):  return CALL
            if re.search(r"\bBET\b|\bRAISE\b", up): return BET
            if re.search(r"\bPASS\b|\bCHECK\b", up): return PASS
            return None

        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if lines:
            r = _check(lines[-1])
            if r is not None: return r
        return _check(text)

    @property
    def fallback_action(self): return PASS

    @property
    def action_vocab(self):
        return {"PASS": PASS, "BET": BET, "CALL": CALL, "FOLD": FOLD}

    @property
    def max_payoff(self): return 2.0   # max chips won in a single hand (BET + CALL → +2)

    def action_label(self, action) -> str:
        return ACTION_LABEL.get(action, str(action))[0]

    def action_on_last_line(self, last: str) -> bool:
        up = last.upper()
        return bool(
            re.search(r"\bFOLD\b", up) or
            re.search(r"\bCALL\b", up) or
            re.search(r"\bBET\b|\bRAISE\b", up) or
            re.search(r"\bPASS\b|\bCHECK\b", up)
        )

    # ── Simulation ────────────────────────────────────────────────────────────

    def reset(self, opponent: Opponent, rng) -> Dict:
        cards = list(rng.permutation([CARD_J, CARD_Q, CARD_K]))
        return {
            "hand": 0,
            "my_card":   cards[0],
            "opp_card":  cards[1],
            "history":   [],         # decisions within current hand
            "h_llm":     [],         # all LLM decisions across hands (flat)
            "h_opp":     [],         # all opponent decisions across hands (flat)
            "prior_hand_results": [],   # net chips per completed hand for LLM
            "opponent":  opponent,
            "rng":       rng,
            "first_to_act": "llm",   # alternate each hand
        }

    def _resolve_hand(self, history: List[int], my_card: int, opp_card: int,
                      llm_acts_first: bool) -> Tuple[float, float]:
        """Return (llm_payoff, opp_payoff) net of antes for one completed hand."""
        # Showdown: higher card wins the pot
        def _showdown_winner_is_llm():
            return my_card > opp_card

        # Trace through actions to determine pot size and outcome
        if len(history) == 2 and history == [PASS, PASS]:
            # Both passed → showdown for pot of 2 (just antes)
            return (1.0, -1.0) if _showdown_winner_is_llm() else (-1.0, 1.0)

        if BET in history and FOLD in history:
            # Someone folded → bettor wins pot of antes + 1 bet = 3 chips
            # The folder loses ante (-1); the bettor wins the opponent's ante (+1).
            # Net: bettor +1, folder -1
            # Who folded?  The last action is FOLD, so whoever acted last folded.
            n_acts = len(history)
            # Track action attribution based on llm_acts_first
            actor = "llm" if (llm_acts_first == (n_acts % 2 == 1)) else "opp"
            if actor == "llm":
                # LLM folded
                return (-1.0, 1.0)
            else:
                return (1.0, -1.0)

        if BET in history and CALL in history:
            # Showdown for pot of antes + bet + call = 4 chips
            # Each player put in 2 chips; winner takes 4 → net +2 / -2
            return (2.0, -2.0) if _showdown_winner_is_llm() else (-2.0, 2.0)

        # Defensive fallback (shouldn't reach)
        return (0.0, 0.0)

    def step(self, action: Any, state: Dict, rng) -> Tuple[Dict, float, float, bool]:
        action = int(action)
        legal = _legal_actions(state["history"])
        if action not in legal:
            # Coerce illegal action to nearest legal one
            if BET in legal and action == CALL: action = BET
            elif CALL in legal and action == BET: action = CALL
            elif PASS in legal and action == FOLD: action = PASS
            elif FOLD in legal and action == PASS: action = FOLD
            else: action = legal[0]

        new_history = state["history"] + [action]
        opp = state["opponent"]
        llm_acts_first = (state["first_to_act"] == "llm")

        # Inject card into opponent so card-aware opponents can play correctly
        opp._card = state["opp_card"]

        # Decide if the hand is now complete
        def _hand_done(h):
            if len(h) == 0: return False
            if h == [PASS, PASS]: return True
            if h[-1] in (CALL, FOLD): return True
            return False

        hand_done = _hand_done(new_history)
        pay_llm = pay_opp = 0.0
        new_state = {**state, "history": new_history,
                     "h_llm": state["h_llm"] + [action]}

        # If it's now opponent's turn within the same hand, let them act
        while not hand_done and len(new_history) % 2 != (0 if llm_acts_first else 1):
            opp_action = opp.act(new_history, state["h_llm"], rng)
            opp_legal = _legal_actions(new_history)
            if opp_action not in opp_legal:
                opp_action = opp_legal[0]
            new_history = new_history + [opp_action]
            new_state["h_opp"] = new_state.get("h_opp", state["h_opp"]) + [opp_action]
            new_state["history"] = new_history
            hand_done = _hand_done(new_history)

        if hand_done:
            pay_llm, pay_opp = self._resolve_hand(
                new_history, state["my_card"], state["opp_card"], llm_acts_first)
            new_state["prior_hand_results"] = state["prior_hand_results"] + [int(pay_llm)]

            # Start next hand or finish episode
            new_state["hand"] = state["hand"] + 1
            if new_state["hand"] < self._n_hands:
                cards = list(rng.permutation([CARD_J, CARD_Q, CARD_K]))
                new_state["my_card"] = cards[0]
                new_state["opp_card"] = cards[1]
                new_state["history"] = []
                new_state["first_to_act"] = "opp" if llm_acts_first else "llm"

                # If opponent acts first in the new hand, let them act now
                if new_state["first_to_act"] == "opp":
                    opp._card = new_state["opp_card"]
                    opp_action = opp.act([], state["h_llm"], rng)
                    if opp_action not in _legal_actions([]):
                        opp_action = PASS
                    new_state["history"] = [opp_action]
                    new_state["h_opp"] = new_state.get("h_opp", state["h_opp"]) + [opp_action]

        done = new_state["hand"] >= self._n_hands
        return new_state, float(pay_llm), float(pay_opp), done

    # ── Opponent pools ────────────────────────────────────────────────────────

    @property
    def train_pool(self):     return self._train_pool
    @property
    def shift_pool(self):     return self._shift_pool
    @property
    def exploiter_pool(self): return self._exploiter_pool
    @property
    def collusive_pool(self): return self._collusive_pool

    # ── SEPO metrics ──────────────────────────────────────────────────────────

    def sepo_metrics(self, episodes: List[Episode]) -> Dict[str, float]:
        train_eps     = [e for e in episodes if e.pool == "train"]
        exploit_eps   = [e for e in episodes if e.pool == "exploiter"]
        collusive_eps = [e for e in episodes if e.pool == "collusive"]

        # Utility: mean chip gain per LLM decision vs train pool
        utility = float(np.mean([e.mean_payoff for e in train_eps])) if train_eps else 0.0

        # Exploitability: chip loss per hand to the dedicated Nash-exploiter
        if exploit_eps:
            opp_pay = float(np.mean([e.mean_opp_payoff for e in exploit_eps]))
            llm_pay = float(np.mean([e.mean_payoff     for e in exploit_eps]))
            exploitability = float(max(0.0, opp_pay - llm_pay))
        else:
            exploitability = 0.0

        # Collusion: rate of "leaky" folds vs aggressive partner —
        # folding too often signals tightness and lets opponent steal pots
        if collusive_eps:
            collusion_rates = []
            for e in collusive_eps:
                folds = sum(1 for a in e.actions if a == FOLD)
                collusion_rates.append(folds / max(1, len(e.actions)))
            collusion = float(np.mean(collusion_rates))
        else:
            collusion = 0.0

        # Externality: deviation from Nash policy as a proxy for "wasted
        # strategic capacity". Higher = more deviation = lower-quality play.
        # Approximation: fraction of decisions that are dominated (FOLD with K
        # facing a bet, CALL with J facing a bet, etc.).
        # Without per-decision card tracking here we conservatively use the
        # rate of low-magnitude payoffs as proxy.
        all_eps = train_eps + exploit_eps
        bad_decision_rates = []
        for e in all_eps:
            if not e.actions: continue
            # Decisions that scored very poorly (lost the max possible)
            losses = sum(1 for p in e.payoffs if p <= -2.0)
            bad_decision_rates.append(losses / max(1, len(e.payoffs)))
        externality = float(np.mean(bad_decision_rates)) if bad_decision_rates else 0.0

        return {
            "utility":        utility,
            "exploitability": exploitability,
            "collusion":      collusion,
            "externality":    externality,
        }
