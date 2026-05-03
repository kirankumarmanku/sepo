"""
Abstract base classes for pluggable game environments.

To add a new game:
  1. Create games/<your_game>.py
  2. Subclass Game and implement all abstract methods
  3. Register in games/__init__.py

Planned games:
  - IPD (iterated prisoner's dilemma)         [DONE]
  - Resource (shared resource extraction)
  - Auction (repeated bidding)
  - Negotiation (split bargaining)
  - Pressure (coercion environment)
  - PublicGoods (contribution game)
  - TrustGame (sender/receiver)
  - BattleOfSexes (coordination)
  - StagHunt (cooperation)
  - SnowDrift (volunteer's dilemma)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class Episode:
    """Record of one completed episode."""
    opponent_name: str
    pool: str                      # "train" | "exploiter" | "collusive" | "shift"
    actions: List[Any]             # model's actions each step
    opp_actions: List[Any]         # opponent's actions each step
    payoffs: List[float]           # model's per-step payoffs
    opp_payoffs: List[float]       # opponent's per-step payoffs
    metadata: Dict = field(default_factory=dict)

    @property
    def total_payoff(self): return sum(self.payoffs)
    @property
    def mean_payoff(self): return sum(self.payoffs) / len(self.payoffs) if self.payoffs else 0.0
    @property
    def mean_opp_payoff(self): return sum(self.opp_payoffs) / len(self.opp_payoffs) if self.opp_payoffs else 0.0


class Opponent(ABC):
    name: str

    @abstractmethod
    def act(self, my_history: List, opp_history: List, rng) -> Any:
        """Return this opponent's action given histories."""


class Game(ABC):
    """
    Abstract interface every game must implement.

    Lifecycle per episode:
      state = game.reset(opponent, rng)
      while not done:
          user_msg = game.user_prompt(state)
          action   = model.generate(game.system_prompt(), user_msg)
          state, payoff, opp_payoff, done = game.step(action, state, rng)
      episode = game.make_episode(opponent, pool, ...)
      metrics = game.sepo_metrics(episodes)
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier e.g. 'ipd', 'resource'."""

    @property
    @abstractmethod
    def n_steps(self) -> int:
        """Number of steps per episode."""

    # ── Prompts ───────────────────────────────────────────────────────────────
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt sent to model at the start of every episode."""

    @abstractmethod
    def user_prompt(self, state: Dict) -> str:
        """Build the per-step user message from current game state."""

    @abstractmethod
    def parse_action(self, text: str) -> Any:
        """Parse model output text into a valid action. Return None on failure."""

    @property
    @abstractmethod
    def fallback_action(self) -> Any:
        """Action to use when parse_action returns None."""

    @property
    def action_vocab(self) -> Dict[str, Any]:
        """Map action strings to action values for constrained decoding. Override in each game."""
        return {}

    # ── Simulation ────────────────────────────────────────────────────────────
    @abstractmethod
    def reset(self, opponent: Opponent, rng) -> Dict:
        """Return initial state dict for a new episode."""

    @abstractmethod
    def step(self, action: Any, state: Dict, rng) -> Tuple[Dict, float, float, bool]:
        """
        Apply action, advance state.
        Returns: (new_state, model_payoff, opp_payoff, done)
        """

    # ── Opponent pools ────────────────────────────────────────────────────────
    @property
    @abstractmethod
    def train_pool(self) -> List[Opponent]:
        """Opponents used to estimate utility."""

    @property
    @abstractmethod
    def shift_pool(self) -> List[Opponent]:
        """Held-out opponents for robustness under distribution shift."""

    @property
    @abstractmethod
    def exploiter_pool(self) -> List[Opponent]:
        """Adversarial opponents for exploitability measurement."""

    @property
    @abstractmethod
    def collusive_pool(self) -> List[Opponent]:
        """Partners used to measure collusion risk."""

    # ── SEPO metrics ──────────────────────────────────────────────────────────
    @abstractmethod
    def sepo_metrics(self, episodes: List[Episode]) -> Dict[str, float]:
        """
        Compute all SEPO metrics from a list of completed episodes.

        Must return a dict with at least:
          utility       — mean payoff vs train pool
          exploitability — how much exploiter earns over us
          collusion     — collusion risk score
          externality   — externality cost
        """
