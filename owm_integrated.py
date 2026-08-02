"""
Opponent World Models -- integrated PoC.

Evaluates an opponent-modeling PLANNER against the ACTUAL trained SEPO policies
(reward-only / welfare-only / SEPO), inside the real SEPO simulator, on the same
opponent pools and the same exploitability metric. Two games (IPD + Negotiation)
show the effect is not IPD-specific.

Two planners are evaluated, differing ONLY in the objective they plan against:
  - "payoff"  -- maximize own payoff (the naive opponent-modeling agent)
  - "sepo"    -- maximize the SEPO objective
                 payoff - le*exploitability - lc*collusion - lx*externality
                 with the SAME lambdas the SEPO optimizer trains against.
Everything else (opponent model, belief update, planning horizon, opponent pools,
seeds) is identical, so the gap between the two points isolates the objective.

Run next to eval/run_sepo_experiments.py:  python owm_integrated.py
  -> owm_integrated.png + printed table
"""
from __future__ import annotations
import itertools, sys, os, zlib
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "eval"), "/mnt/project"):
    sys.path.insert(0, _p)
try:
    import run_sepo_experiments as sepo
except ModuleNotFoundError:
    import sepo_ref as sepo

import random

IPD_OPPS = ["always_cooperate", "always_defect", "tit_for_tat", "generous_tft", "grim"]
NEG_OPPS = ["fair_split", "balanced_split", "selfish_split", "hardball_split", "concede_split"]
PAY = {("C","C"):(3,3), ("D","C"):(5,0), ("C","D"):(0,5), ("D","D"):(1,1)}


# ============================ the planning objective ========================
# Recover SEPO's penalty weights from run_sepo_experiments.objective_value itself
# rather than restating them here, so the planner and the SEPO optimizer can
# never drift apart: probe the linear objective with a one-hot metrics dict.
def _sepo_lambdas():
    zero = {"payoff": 0.0, "welfare": 0.0, "exploitability": 0.0,
            "collusion_risk": 0.0, "externality_cost": 0.0}
    def weight(key):
        m = dict(zero); m[key] = 1.0
        return -sepo.objective_value("sepo", m)
    return weight("exploitability"), weight("collusion_risk"), weight("externality_cost")

LAM_E, LAM_C, LAM_X = _sepo_lambdas()   # (3.6, 3.2, 2.4) as of SEPO's current config


def plan_score(objective, own, opp, ext, coll, rounds):
    """Score a *whole episode* (realized so far + planned ahead).

    Exploitability is max(0, opp - own)/rounds -- an episode-level, non-additive
    quantity, so it cannot be decomposed into per-round rewards. The planner
    therefore always scores the full episode total, not just the continuation.

    Scale note: every term here is per-round, matching the axes of the frontier
    plot, so the SEPO objective is a straight line on that plot. SEPO's own
    objective_value() pairs *episode-total* payoff with per-round penalties;
    normalizing payoff too makes the penalties relatively stronger than in SEPO's
    training. That is the conservative direction for the claim below -- the
    safety-aware planner is not flattered on the exploitability axis.
    """
    if objective == "payoff":
        return own
    return (own / rounds
            - LAM_E * max(0.0, opp - own) / rounds
            - LAM_C * (coll / rounds)
            - LAM_X * (ext / rounds))


# ============================ IPD opponent-model planner ====================
# Belief over the 5 SEPO IPD strategy types from observed opponent moves; plan
# by exact expectimax (types are deterministic given our move history; the one
# stochastic type, generous_tft, is handled softly in the belief and via MAP in
# planning -> a realistic touch of model misspecification).
def opp_pred(htype, my_hist, opp_defected_by_me):
    """Predicted opp move for a hypothesis type given OUR move history."""
    if htype == "always_cooperate": return "C"
    if htype == "always_defect":    return "D"
    if htype == "tit_for_tat":      return "C" if not my_hist else my_hist[-1]
    if htype == "grim":             return "D" if opp_defected_by_me else "C"
    if htype == "generous_tft":     return "C" if (not my_hist or my_hist[-1]=="C") else "D"  # MAP
    return "C"

def belief_update(belief, my_hist, obs, rounds_seen):
    my_last = my_hist[-1] if my_hist else None
    lik = {}
    for h in belief:
        if h == "generous_tft" and my_last == "D":
            p_c = 0.25
            lik[h] = p_c if obs=="C" else (1-p_c)
        else:
            pred = opp_pred(h, my_hist, any(m=="D" for m in my_hist))
            lik[h] = 1.0 if obs==pred else 0.0
    new = {h: belief[h]*lik[h] for h in belief}
    s = sum(new.values())
    if s == 0:                       # observation inconsistent with all -> reset
        return {h: 1/len(belief) for h in belief}
    return {h: v/s for h, v in new.items()}

def ipd_rollout(htype, my_seq):
    """Own/opp payoff and externality of playing my_seq from scratch vs a type.

    Externality matches sepo.simulate_social_dilemma: max(0, 6 - joint)/6 per
    round. Collusion is identically 0 in that simulator, so it is not tracked.
    """
    own = opp = ext = 0.0
    mh = []
    for a in my_seq:
        b = opp_pred(htype, mh, any(m=="D" for m in mh))
        pa, pb = PAY[(a,b)]
        own += pa; opp += pb
        ext += max(0.0, 6 - (pa + pb)) / 6.0
        mh.append(a)
    return own, opp, ext

def ipd_planner_move(belief, my_hist, remaining, realized, objective, depth_cap=6):
    depth = min(remaining, depth_cap)
    rounds_scored = len(my_hist) + depth
    r_own, r_opp, r_ext = realized
    best_a, best_v = "C", -1e9
    for first in ("C","D"):
        v = 0.0
        for h, w in belief.items():
            if w == 0: continue
            b_own, b_opp, b_ext = ipd_rollout(h, my_hist)   # model's replay of the prefix
            best_cont = max(
                plan_score(objective,
                           r_own + (own - b_own),
                           r_opp + (opp - b_opp),
                           r_ext + (ext - b_ext),
                           0.0, rounds_scored)
                for own, opp, ext in (
                    ipd_rollout(h, my_hist + [first] + list(tail))
                    for tail in itertools.product("CD", repeat=depth-1)
                )
            )
            v += w * best_cont
        if v > best_v:
            best_a, best_v = first, v
    return best_a

def play_ipd_planner(opponent_name, rng, rounds=8, objective="payoff"):
    opp_strat = sepo.STRATEGIES["social_dilemma"][opponent_name]
    belief = {h: 1/len(IPD_OPPS) for h in IPD_OPPS}
    state_b = {"round":0, "opp_last":"C", "opp_defected_ever":False}
    my_hist = []
    own = opp = ext = 0.0
    for t in range(rounds):
        a = ipd_planner_move(belief, my_hist, rounds - t, (own, opp, ext), objective)
        b = opp_strat(state_b, rng)
        pa, pb = PAY[(a,b)]
        own += pa; opp += pb
        ext += max(0.0, 6 - (pa + pb)) / 6.0
        belief = belief_update(belief, my_hist, b, t)
        my_hist.append(a)
        state_b = {"round":t+1, "opp_last":a, "opp_defected_ever": state_b["opp_defected_ever"] or a=="D"}
    return own/rounds, max(0.0, opp-own)/rounds


# ============================ Negotiation planner ===========================
# The SEPO negotiation strategies are (near-)stateless in the round, so the
# opponent model is "next offer = last observed offer" (prior 5 before any
# observation) and planning is a 1-step argmax over the offer grid -- shared by
# both objectives. Under "payoff" the argmax is 10 - predicted_offer, i.e. claim
# the complement and close the deal; under "sepo" an offer that overshoots 10
# (no deal, both get 0) becomes the preferred move when the only closeable deal
# is one the opponent wins.
NEG_OFFERS = range(1, 10)

def play_neg_planner(opponent_name, rng, rounds=4, objective="payoff"):
    opp_strat = sepo.STRATEGIES["negotiation"][opponent_name]
    own = opp = ext = 0.0
    opp_last = None
    for t in range(rounds):
        pred = 5 if opp_last is None else opp_last
        best_o, best_v = 5, -1e9
        for o in NEG_OFFERS:
            pa, pb, w, inequ = sepo.bargaining_outcome(o, pred)
            step_ext = inequ + (1.0 if w == 0 else 0.0)
            v = plan_score(objective, own + pa, opp + pb, ext + step_ext, 0.0, t + 1)
            if v > best_v:
                best_o, best_v = o, v
        act_b = opp_strat({"round": t}, rng)
        ob = act_b["offer"]
        pa, pb, w, inequ = sepo.bargaining_outcome(best_o, ob)
        own += pa; opp += pb
        ext += inequ + (1.0 if w == 0 else 0.0)
        opp_last = ob
    return own/rounds, max(0.0, opp-own)/rounds


# ============================ evaluation ====================================
def opp_seed(seed, name):
    """Per-opponent seed. crc32, not hash(): str hashing is salted per process
    (PYTHONHASHSEED), which made the reported numbers move between runs."""
    return seed + zlib.crc32(name.encode()) % 1000

def eval_planner(game, opps, objective="payoff", seed=7, episodes=40):
    fn = play_ipd_planner if game=="ipd" else play_neg_planner
    ps, es = [], []
    for o in opps:
        rng = random.Random(opp_seed(seed, o))
        for _ in range(episodes):
            p, e = fn(o, rng, objective=objective)
            ps.append(p); es.append(e)
    return float(np.mean(ps)), float(np.mean(es))

def eval_sepo_mix(domain, opps, probs, seed=7, episodes=200):
    sim = sepo.SIMULATORS[domain]
    ps, es = [], []
    for o in opps:
        rng = random.Random(opp_seed(seed, o))
        for _ in range(episodes):
            ep = sim(probs, o, rng)
            rounds = 8 if domain=="social_dilemma" else 4
            ps.append(ep.own_payoff/rounds); es.append(ep.exploitability)
    return float(np.mean(ps)), float(np.mean(es))

def sepo_policies(domain):
    out = {}
    for obj in ["reward_only", "welfare_only", "sepo"]:
        probs, _ = sepo.optimize_policy(domain, obj, seed=13, iterations=10, samples=36, elites=8)
        out[obj] = probs
    return out


AGENTS = ["reward_only", "welfare_only", "sepo", "planner", "planner_safe"]

LABEL = {"reward_only": "SEPO reward-only", "welfare_only": "SEPO welfare-only",
         "sepo": "SEPO (full)", "planner": "OWM planner (payoff)",
         "planner_safe": "OWM planner (SEPO objective)"}
# Short forms for the in-plot direct labels.
TAG = {"reward_only": "reward-only", "welfare_only": "welfare-only",
       "sepo": "SEPO", "planner": "OWM (payoff)", "planner_safe": "OWM (SEPO obj.)"}
# Categorical hues, fixed order, validated all-pairs for scatter (OKLab CVD
# dE >= 6.4 protan/deutan, normal-vision floor 18.1, vs a white surface). Every
# point is direct-labeled, which is the secondary encoding the floor-band pair
# and the sub-3:1 orange both require.
COLOR = {"reward_only": "#c0392b", "welfare_only": "#e08a2b", "sepo": "#2b6cb0",
         "planner": "#2a8a5c", "planner_safe": "#b45ae8"}
MARKER = {"reward_only": "o", "welfare_only": "o", "sepo": "o",
          "planner": "*", "planner_safe": "P"}
# Direct-label placement: (dx, dy) in points, horizontal alignment, and whether
# to draw a leader line. Several points sit almost on top of each other (IPD
# reward-only/SEPO; the whole zero-exploitability cluster in negotiation), so
# labels are laddered off to one side and tied back with leaders.
OFFSET = {
    "ipd": {"reward_only":  (-14, -13, "right", True),
            "welfare_only": (10, -3, "left", False),
            "sepo":         (-14, 7, "right", True),
            "planner":      (-16, 4, "right", True),
            "planner_safe": (-16, 4, "right", True)},
    # The whole zero-exploitability cluster is labelled on the RIGHT, keeping the
    # left side clear as the approach lane for the objective-swap arrow.
    "neg": {"reward_only":  (14, -20, "left", True),
            "welfare_only": (14, -2, "left", True),
            "sepo":         (14, 16, "left", True),
            "planner":      (0, 15, "center", False),
            "planner_safe": (14, 34, "left", True)},
}
LEADER = dict(arrowstyle="-", color="#9a9a9a", lw=0.5, shrinkA=0, shrinkB=7)


def place_label(ax, text, xy, spec):
    dx, dy, ha, leader = spec
    ax.annotate(text, xy, textcoords="offset points", xytext=(dx, dy),
                fontsize=7, color="#333333", ha=ha, va="center", zorder=4,
                arrowprops=dict(LEADER) if leader else None)


def main():
    results = {}   # (game, agent) -> (payoff, exploit)

    # ---- IPD
    for obj, probs in sepo_policies("social_dilemma").items():
        results[("ipd", obj)] = eval_sepo_mix("social_dilemma", IPD_OPPS, probs)
    results[("ipd", "planner")] = eval_planner("ipd", IPD_OPPS, objective="payoff")
    results[("ipd", "planner_safe")] = eval_planner("ipd", IPD_OPPS, objective="sepo")

    # ---- Negotiation
    for obj, probs in sepo_policies("negotiation").items():
        results[("neg", obj)] = eval_sepo_mix("negotiation", NEG_OPPS, probs)
    results[("neg", "planner")] = eval_planner("neg", NEG_OPPS, objective="payoff")
    results[("neg", "planner_safe")] = eval_planner("neg", NEG_OPPS, objective="sepo")

    print(f"planning lambdas: exploit={LAM_E}, collusion={LAM_C}, externality={LAM_X}\n")
    print(f"{'game':<5}{'agent':<30}{'pay/rnd':>9}{'exploit':>10}{'SEPO obj':>10}")
    for g in ("ipd", "neg"):
        for a in AGENTS:
            p, e = results[(g, a)]
            print(f"{g:<5}{LABEL[a]:<30}{p:>9.2f}{e:>10.3f}{p - LAM_E*e:>10.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, game, title in [(axes[0],"ipd","A. IPD  (8-round)"),
                            (axes[1],"neg","B. Negotiation  (4-round)")]:
        # results holds (payoff, exploitability); the plot is x=exploitability,
        # y=payoff, so build the two planner points as explicit (x, y) pairs.
        pay_g, exp_g = results[(game, "planner")]
        pay_s, exp_s = results[(game, "planner_safe")]
        xy_greedy, xy_safe = (exp_g, pay_g), (exp_s, pay_s)
        # The two planners can land on the SAME point (IPD: the payoff-greedy
        # plan is already unexploitable, so the objective swap changes nothing).
        # Say that rather than nudging a marker off its true coordinates.
        coincide = abs(exp_g - exp_s) < 1e-6 and abs(pay_g - pay_s) < 1e-6

        for a in AGENTS:
            p, e = results[(game, a)]
            if coincide and a == "planner_safe":
                # Concentric ring on the shared point: both series read, neither moves.
                ax.scatter(e, p, s=620, marker="o", facecolors="none",
                           edgecolors=COLOR[a], linewidth=1.8, zorder=3)
                continue
            # White ring: the only separation between marks that overlap
            # (IPD reward-only sits almost exactly under SEPO).
            ax.scatter(e, p, s=230 if a.startswith("planner") else 130,
                       marker=MARKER[a], color=COLOR[a], edgecolor="#ffffff",
                       linewidth=1.5, zorder=3)
            if not (coincide and a == "planner"):
                place_label(ax, TAG[a], (e, p), OFFSET[game][a])
        if coincide:
            place_label(ax, "OWM: payoff & SEPO objective\nland on the same point",
                        xy_greedy, OFFSET[game]["planner"])
        else:
            # The objective swap, drawn as the move it makes on the frontier.
            # Bowed away from the direct-label ladder on the destination cluster.
            ax.annotate("", xy=xy_safe, xytext=xy_greedy, zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color="#777777",
                                        lw=1.3, linestyle=(0, (5, 3)),
                                        connectionstyle="arc3,rad=0.22",
                                        shrinkA=13, shrinkB=18))
            mid = (0.5*(exp_g + exp_s), 0.5*(pay_g + pay_s))
            ax.annotate("swap the planning objective", mid, textcoords="offset points",
                        xytext=(6, 26), fontsize=7, style="italic",
                        color="#777777", ha="center", zorder=5)
        ax.set_xlabel("exploitability (opp advantage / round)  <- better")
        ax.set_ylabel("payoff / round   better ->")
        ax.set_title(title)
        ax.invert_xaxis(); ax.grid(alpha=0.25)
        ax.margins(x=0.30, y=0.20)
        # The margin that gives the direct labels room can push the axis past
        # zero; exploitability is non-negative, so drop any negative tick.
        xl = ax.get_xlim()
        ax.set_xticks([t for t in ax.get_xticks() if t >= -1e-9])
        ax.set_xlim(xl)
    # Proxy handles so the legend shows each series' canonical marker even where
    # the panel drew it as a ring.
    handles = [plt.Line2D([], [], linestyle="none", marker=MARKER[a], color=COLOR[a],
                          markeredgecolor="#ffffff", markeredgewidth=1.2,
                          markersize=11 if a.startswith("planner") else 8, label=LABEL[a])
               for a in AGENTS]
    axes[0].legend(handles=handles, fontsize=7.5, loc="lower right")
    fig.suptitle("Opponent-model planner vs. trained SEPO policies (same simulator, same pools)",
                 fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig("owm_integrated.png", dpi=150)
    print("\nwrote owm_integrated.png")


if __name__ == "__main__":
    main()
