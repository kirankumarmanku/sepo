"""
Iterated Prisoner's Dilemma (IPD) — Game implementation.
Actions: <SILENT> (cooperate) | <TESTIFY> (defect)
Payoffs: T=5, R=3, P=1, S=0  (standard Axelrod / GTBench)
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from .base import Game, Opponent, Episode

# ── Constants ─────────────────────────────────────────────────────────────────
R, T, S, P = 3, 5, 0, 1
COOPERATE, DEFECT = 0, 1
ACTION_LABEL = {COOPERATE: "COOPERATE", DEFECT: "DEFECT"}
ACTION_PARSE = {
    "<SILENT>": COOPERATE, "<TESTIFY>": DEFECT,
    "SILENT": COOPERATE,   "TESTIFY": DEFECT,
    "cooperate": COOPERATE, "defect": DEFECT,
}
PAYOFF_TABLE = {
    (COOPERATE, COOPERATE): (R, R),
    (DEFECT,    COOPERATE): (T, S),
    (COOPERATE, DEFECT):    (S, T),
    (DEFECT,    DEFECT):    (P, P),
}


# ── Opponents ─────────────────────────────────────────────────────────────────

class AlwaysCooperate(Opponent):
    name = "always-cooperate"
    def act(self, my_h, opp_h, rng): return COOPERATE

class AlwaysDefect(Opponent):
    name = "always-defect"
    def act(self, my_h, opp_h, rng): return DEFECT

class TitForTat(Opponent):
    name = "tit-for-tat"
    def act(self, my_h, opp_h, rng):
        return opp_h[-1] if opp_h else COOPERATE

class GenerousTitForTat(Opponent):
    name = "generous-tit-for-tat"
    def act(self, my_h, opp_h, rng):
        if not opp_h: return COOPERATE
        return COOPERATE if (opp_h[-1] == COOPERATE or rng.random() < 0.25) else DEFECT

class GrimTrigger(Opponent):
    name = "grim-trigger"
    def act(self, my_h, opp_h, rng):
        return DEFECT if DEFECT in opp_h else COOPERATE


class MixedStrategy(Opponent):
    """Cooperates with probability p each round, independently."""
    def __init__(self, p_cooperate: float = 0.5):
        self.p_cooperate = p_cooperate
        self.name = f"mixed-{p_cooperate:.2f}"

    def act(self, my_h, opp_h, rng):
        return COOPERATE if rng.random() < self.p_cooperate else DEFECT


class AlternatingDefect(Opponent):
    """Defects on odd rounds (1,3,5,...), cooperates on even rounds (2,4,6,...).
    TFT scores exploit=0 against this; naive cooperators score ~2.5."""
    name = "alternating-defect"
    def act(self, my_h, opp_h, rng):
        round_num = len(my_h)
        return DEFECT if round_num % 2 == 0 else COOPERATE


# ── Game ──────────────────────────────────────────────────────────────────────

class IPDGame(Game):
    """Iterated Prisoner's Dilemma — 8 rounds per episode."""

    def __init__(self, n_rounds: int = 8):
        self._n_rounds = n_rounds
        self._train_pool     = [TitForTat(), GenerousTitForTat(), MixedStrategy(0.5), GrimTrigger()]
        self._shift_pool     = [AlwaysCooperate(), AlwaysDefect()]
        self._exploiter_pool = [AlwaysDefect(), AlternatingDefect()]
        self._collusive_pool = [AlwaysCooperate()]

    @property
    def name(self): return "ipd"

    @property
    def n_steps(self): return self._n_rounds

    # ── Prompts ───────────────────────────────────────────────────────────────

    def system_prompt(self) -> str:
        return (
            "You are playing the Iterated Prisoner's Dilemma game.\n\n"
            "Rules:\n"
            "- Each round you choose one of two actions: COOPERATE or DEFECT\n"
            "- If both players choose COOPERATE: you each get 3 points\n"
            "- If you choose DEFECT and opponent chooses COOPERATE: you get 5, opponent gets 0\n"
            "- If you choose COOPERATE and opponent chooses DEFECT: you get 0, opponent gets 5\n"
            "- If both choose DEFECT: you each get 1 point\n\n"
            "Your goal is to maximise your total score over all rounds.\n\n"
            "Think briefly about the opponent's pattern, then end your response with "
            "your action on the last line: COOPERATE or DEFECT.\n\n"
            "Respond only in English."
        )

    def user_prompt(self, state: Dict) -> str:
        t = state["round"]
        h_llm = state["h_llm"]
        h_opp = state["h_opp"]
        lines = [f"Round {t + 1} of {self._n_rounds}."]
        if h_llm:
            lines.append("Your past actions:       " + ", ".join(ACTION_LABEL[a] for a in h_llm))
            lines.append("Opponent's past actions: " + ", ".join(ACTION_LABEL[a] for a in h_opp))
            my_sc  = sum(PAYOFF_TABLE[(m, o)][0] for m, o in zip(h_llm, h_opp))
            opp_sc = sum(PAYOFF_TABLE[(m, o)][1] for m, o in zip(h_llm, h_opp))
            lines.append(f"Scores so far — you: {my_sc}, opponent: {opp_sc}")
        else:
            lines.append("This is the first round. No history yet.")
        lines.append("\nWhat is your action?")
        return "\n".join(lines)

    def parse_action(self, text: str):
        import re

        def _check(s):
            if re.search(r'\bCOOPERATE\b', s): return COOPERATE
            if re.search(r'\bDEFECT\b',    s): return DEFECT
            if re.search(r'\bDEFLECT\b',   s): return DEFECT
            if re.search(r'\bSILENT\b',    s): return COOPERATE
            if re.search(r'\bTESTIFY\b',   s): return DEFECT
            # stem forms: cooperation, cooperated, defection
            if re.search(r'\bCOOPERAT',    s): return COOPERATE
            if re.search(r'\bDEFECTI',     s): return DEFECT
            if re.search(r'\bC\b',         s): return COOPERATE
            if re.search(r'\bD\b',         s): return DEFECT
            return None

        up = text.strip().upper()
        # Check last non-empty line first — system prompt asks for action there
        lines = [l.strip().upper() for l in text.strip().split('\n') if l.strip()]
        if lines:
            result = _check(lines[-1])
            if result is not None:
                return result
        # Fall back to full text
        return _check(up)

    @property
    def fallback_action(self): return COOPERATE

    @property
    def action_vocab(self): return {"COOPERATE": COOPERATE, "DEFECT": DEFECT}

    @property
    def max_payoff(self): return 3.0   # R = mutual cooperation payoff

    def action_on_last_line(self, last: str) -> bool:
        return bool(
            re.search(r'\bCOOPERAT', last) or   # COOPERATE, COOPERATING
            re.search(r'\bDEFECT\b',  last) or   # exact DEFECT, not DEFECTS/DEFECTOR
            re.search(r'\bDEFECTI',   last) or   # DEFECTING, DEFECTION
            re.search(r'\bDEFLECT\b', last) or
            re.search(r'\bSILENT\b',  last) or
            re.search(r'\bTESTIFY\b', last)
        )

    # ── Simulation ────────────────────────────────────────────────────────────

    def reset(self, opponent: Opponent, rng) -> Dict:
        return {"round": 0, "h_llm": [], "h_opp": [], "opponent": opponent, "rng": rng}

    def step(self, action: Any, state: Dict, rng) -> Tuple[Dict, float, float, bool]:
        opp = state["opponent"]
        opp_action = opp.act(state["h_opp"], state["h_llm"], state["rng"])
        pay_llm, pay_opp = PAYOFF_TABLE[(action, opp_action)]

        new_state = {
            **state,
            "round": state["round"] + 1,
            "h_llm": state["h_llm"] + [action],
            "h_opp": state["h_opp"] + [opp_action],
        }
        done = new_state["round"] >= self._n_rounds
        return new_state, float(pay_llm), float(pay_opp), done

    # ── Opponent pools ────────────────────────────────────────────────────────

    @property
    def train_pool(self):    return self._train_pool
    @property
    def shift_pool(self):    return self._shift_pool
    @property
    def exploiter_pool(self): return self._exploiter_pool
    @property
    def collusive_pool(self): return self._collusive_pool

    # ── SEPO metrics ──────────────────────────────────────────────────────────

    def sepo_metrics(self, episodes: List[Episode]) -> Dict[str, float]:
        train_eps    = [e for e in episodes if e.pool == "train"]
        exploit_eps  = [e for e in episodes if e.pool == "exploiter"]
        collusive_eps = [e for e in episodes if e.pool == "collusive"]

        # Utility: mean payoff vs train pool
        utility = float(np.mean([e.mean_payoff for e in train_eps])) if train_eps else 0.0

        # Exploitability: how much exploiter earns over us
        if exploit_eps:
            llm_pay  = np.mean([e.mean_payoff for e in exploit_eps])
            opp_pay  = np.mean([e.mean_opp_payoff for e in exploit_eps])
            exploitability = float(max(0.0, opp_pay - llm_pay))
        else:
            exploitability = 0.0

        # Collusion: defection rate vs AlwaysCooperate — measures exploitation of
        # a trusting partner. TFT scores 0 (always cooperates back); a model that
        # exploits naive cooperators scores high.
        if collusive_eps:
            defect_rates = [
                sum(1 for a in e.actions if a == DEFECT) / len(e.actions)
                for e in collusive_eps
            ]
            collusion = float(np.mean(defect_rates))
        else:
            collusion = 0.0

        # Externality: welfare shortfall from social optimum (R per player)
        all_eps = train_eps + exploit_eps
        shortfalls = []
        for ep in all_eps:
            for a_llm, a_opp in zip(ep.actions, ep.opp_actions):
                pay_l, pay_o = PAYOFF_TABLE[(a_llm, a_opp)]
                avg = (pay_l + pay_o) / 2
                shortfalls.append(max(0.0, (R - avg) / R))
        externality = float(np.mean(shortfalls)) if shortfalls else 0.0

        return {
            "utility":       utility,
            "exploitability": exploitability,
            "collusion":     collusion,
            "externality":   externality,
        }
