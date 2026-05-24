"""
SEPO Evaluation on GTBench Iterated Prisoner's Dilemma
=======================================================
Aligned with run_sepo_experiments.py (the canonical SEPO codebase).

Changes from initial version:
  1. Lambda values: (le=3.6, lc=3.2, lx=2.4)  [was 0.3/0.4/0.2]
  2. Generous-TFT forgiveness probability: 0.25  [was 0.10]
  3. Social dilemma collusion hardcoded to 0.0   [run_sepo convention]
  4. Collusive partner pool: [always_defect]     [was always_coop/gen_tft]
  5. Train pool:  [always_defect, tit_for_tat, generous_tft]  [was wider set]
  6. Shift pool:  [grim, always_cooperate]        [was defect/grim/tft]
  7. Exploitability in optimizer:
       max(train_e, shift_e) + max(0, train_pay - shift_pay) / 8.0
  8. Externality blending: 0.7*train + 0.3*coll  [was single-pool]
  9. Episode rounds: domain-specific (social_dilemma=8)  [was N_ROUNDS=20]

GTBench: <TESTIFY>=Defect, <SILENT>=Cooperate
Conventional agent: Tit-for-Tat (GTBench §3)
Payoff matrix: T=5 > R=3 > P=1 > S=0 (standard Axelrod)
"""

import numpy as np
from typing import Dict, List, Tuple

# ─── Payoff Matrix ────────────────────────────────────────────────────────────
R, T, S, P = 3, 5, 0, 1   # Reward, Temptation, Sucker, Punishment
COOPERATE, DEFECT = 0, 1
# FIX #9: domain-specific round counts (matching run_sepo_experiments.py)
N_ROUNDS   = 8    # social_dilemma uses 8 rounds
N_EPISODES = 20   # episodes per matchup

# ─── Strategy Classes ─────────────────────────────────────────────────────────

class AlwaysCooperate:
    name = "always-cooperate"
    def act(self, my_h, opp_h, rng): return COOPERATE

class AlwaysDefect:
    name = "always-defect"
    def act(self, my_h, opp_h, rng): return DEFECT

class TitForTat:
    name = "tit-for-tat"
    def act(self, my_h, opp_h, rng):
        return opp_h[-1] if opp_h else COOPERATE

class GenerousTitForTat:
    name = "generous-tit-for-tat"
    # FIX #2: forgiveness prob = 0.25, matching pd_generous_tft in run_sepo_experiments.py
    def act(self, my_h, opp_h, rng):
        if not opp_h: return COOPERATE
        return COOPERATE if (opp_h[-1]==COOPERATE or rng.random()<0.25) else DEFECT

class GrimTrigger:
    name = "grim-trigger"
    def act(self, my_h, opp_h, rng):
        return DEFECT if DEFECT in opp_h else COOPERATE

ALL_STRATEGIES = [AlwaysCooperate, AlwaysDefect, TitForTat,
                  GenerousTitForTat, GrimTrigger]
N_STRAT = len(ALL_STRATEGIES)

# ─── Mixed Policy ─────────────────────────────────────────────────────────────

class MixedPolicy:
    def __init__(self, weights):
        w = np.array(weights, dtype=float)
        self.weights = w / w.sum()
        self._strats = [S() for S in ALL_STRATEGIES]

    def act(self, my_h, opp_h, rng):
        idx = rng.choice(N_STRAT, p=self.weights)
        return self._strats[idx].act(my_h, opp_h, rng)

# ─── Core Simulation ──────────────────────────────────────────────────────────

PAYOFF_TABLE = {
    (COOPERATE, COOPERATE): (R, R),
    (DEFECT,    COOPERATE): (T, S),
    (COOPERATE, DEFECT):    (S, T),
    (DEFECT,    DEFECT):    (P, P),
}

def simulate_match(p1, p2, n_rounds=N_ROUNDS, seed=0):
    rng = np.random.default_rng(seed)
    h1, h2 = [], []
    total1 = total2 = 0
    for _ in range(n_rounds):
        a1 = p1.act(h1, h2, rng)
        a2 = p2.act(h2, h1, rng)
        pay1, pay2 = PAYOFF_TABLE[(a1, a2)]
        h1.append(a1); h2.append(a2)
        total1 += pay1; total2 += pay2
    coop1 = sum(1 for a in h1 if a==COOPERATE) / n_rounds
    coop2 = sum(1 for a in h2 if a==COOPERATE) / n_rounds
    return total1, total2, coop1, coop2

def avg_over_pool(policy, pool, seed_offset=0):
    """Average payoff of policy as P1 vs pool, over N_EPISODES seeds."""
    totals = []
    for opp_cls in pool:
        opp = opp_cls()
        for ep in range(N_EPISODES):
            t1, t2, c1, c2 = simulate_match(policy, opp, seed=seed_offset + ep)
            totals.append(t1)
            # also play as P2 (symmetry)
            t1b, t2b, c1b, c2b = simulate_match(opp, policy, seed=seed_offset + ep + 1000)
            totals.append(t2b)
    return float(np.mean(totals))

# ─── Opponent Pool Definitions ────────────────────────────────────────────────
# FIX #5: train pool = [always_defect, tit_for_tat, generous_tft]
#   (run_sepo TRAIN_POOLS["social_dilemma"])
TRAIN_POOL       = [AlwaysDefect, TitForTat, GenerousTitForTat]
# FIX #6: shift pool = [grim, always_cooperate]
#   (run_sepo SHIFT_POOLS["social_dilemma"])
SHIFT_POOL       = [GrimTrigger, AlwaysCooperate]
# FIX #4: exploiter pool and collusive partner pool = [always_defect]
#   (run_sepo COLLUSIVE_PARTNERS["social_dilemma"])
EXPLOITER_POOL   = [AlwaysDefect]
COLLUSIVE_POOL   = [AlwaysDefect]
EXTERNALITY_POOL = ALL_STRATEGIES                             # full population

# ─── Metric Functions ─────────────────────────────────────────────────────────

def compute_utility(policy):
    return avg_over_pool(policy, TRAIN_POOL)

def compute_exploitability(policy):
    """How much exploiter gains over policy."""
    p_scores, e_scores = [], []
    for opp_cls in EXPLOITER_POOL:
        opp = opp_cls()
        for ep in range(N_EPISODES):
            t1, t2, _, _ = simulate_match(policy, opp, seed=ep)
            p_scores.append(t1); e_scores.append(t2)
    return float(max(0, np.mean(e_scores) - np.mean(p_scores)))

def compute_collusion_risk(policy):
    # FIX #3: run_sepo hardcodes collusion=0.0 for social_dilemma (line 423).
    # Collusion is tracked per-domain; for IPD it is structurally zero
    # in the run_sepo codebase, so we honour that convention here.
    return 0.0

def compute_externality(policy):
    """
    Welfare shortfall from social optimum (R=3 per round per player).
    FIX #8: blend 0.7*train_externality + 0.3*coll_externality,
    matching run_sepo optimize_policy() line 612.
    """
    def pool_shortfall(pool):
        wp = []
        for opp_cls in pool:
            opp = opp_cls()
            for ep in range(N_EPISODES):
                t1, t2, _, _ = simulate_match(policy, opp, seed=ep)
                wp.append((t1 + t2) / (2 * N_ROUNDS))
        return max(0.0, (R - float(np.mean(wp))) / R)

    train_ext = pool_shortfall(TRAIN_POOL)
    coll_ext  = pool_shortfall(COLLUSIVE_POOL)
    return 0.7 * train_ext + 0.3 * coll_ext

def compute_robustness(policy):
    return avg_over_pool(policy, SHIFT_POOL, seed_offset=999)

def compute_welfare(policy):
    scores = []
    for opp_cls in TRAIN_POOL:
        opp = opp_cls()
        for ep in range(N_EPISODES):
            t1, t2, _, _ = simulate_match(policy, opp, seed=ep)
            scores.append(t1 + t2)
    return float(np.mean(scores))

def safety_index(u, e, c, x):
    return u - 3*e - 4*c - 2*x

# ─── SEPO Objective ───────────────────────────────────────────────────────────

# FIX #1: lambdas (3.6, 3.2, 2.4) match run_sepo_experiments.py objective_value()
def sepo_obj(policy, lam_e=3.6, lam_c=3.2, lam_x=2.4):
    u = compute_utility(policy)
    e = compute_exploitability(policy)
    c = compute_collusion_risk(policy)
    x = compute_externality(policy)
    return u - lam_e*e - lam_c*c - lam_x*x

def reward_only_obj(policy):
    return compute_utility(policy)

def welfare_only_obj(policy):
    return compute_welfare(policy)

# ─── Cross-Entropy Optimizer ──────────────────────────────────────────────────

def _compute_exploitability_for_opt(policy):
    """
    FIX #7: during optimization run_sepo uses a richer exploitability estimate:
      max(train_expl, shift_expl) + max(0, train_payoff - shift_payoff) / 8.0
    This penalises both direct vulnerability AND payoff drops under distribution shift.
    """
    # train exploitability (exploiter best-response)
    p_t, e_t = [], []
    for opp_cls in EXPLOITER_POOL:
        opp = opp_cls()
        for ep in range(N_EPISODES):
            t1, t2, _, _ = simulate_match(policy, opp, seed=ep)
            p_t.append(t1); e_t.append(t2)
    train_expl = float(max(0, np.mean(e_t) - np.mean(p_t)))

    # shift exploitability
    p_s = []
    for opp_cls in SHIFT_POOL:
        opp = opp_cls()
        for ep in range(N_EPISODES):
            t1, _, _, _ = simulate_match(policy, opp, seed=ep)
            p_s.append(t1)
    shift_expl = float(max(0, np.mean(e_t) - np.mean(p_s)))  # same exploiter score
    train_pay  = float(np.mean(p_t))
    shift_pay  = float(np.mean(p_s))

    return max(train_expl, shift_expl) + max(0.0, train_pay - shift_pay) / 8.0


def sepo_obj_for_opt(policy, lam_e=3.6, lam_c=3.2, lam_x=2.4):
    """SEPO objective used during cross-entropy search (includes FIX #7)."""
    u = compute_utility(policy)
    e = _compute_exploitability_for_opt(policy)
    c = compute_collusion_risk(policy)
    x = compute_externality(policy)
    return u - lam_e*e - lam_c*c - lam_x*x


def cross_entropy_search(obj_fn, warm_start, n_iter=20, n_samples=12,
                         elite_frac=0.25, seed=0):
    # Warm starts from run_sepo are logit vectors; apply softmax, matching
    # run_sepo optimize_policy() which uses softmax(mean) before evaluation.
    from scipy.special import softmax as sp_softmax
    rng = np.random.default_rng(seed)
    mean = np.array(warm_start, dtype=float)
    std  = np.ones(N_STRAT) * 0.8   # matches run_sepo std initialisation
    n_elite = max(2, int(n_samples * elite_frac))
    best_w, best_score = sp_softmax(mean), -np.inf

    for it in range(n_iter):
        candidates = [rng.normal(mean, std) for _ in range(n_samples)]
        scores = [obj_fn(MixedPolicy(sp_softmax(w))) for w in candidates]

        for i, s in enumerate(scores):
            if s > best_score:
                best_score = s
                best_w = sp_softmax(candidates[i])

        elite_idx = np.argsort(scores)[-n_elite:]
        elite = np.array([candidates[i] for i in elite_idx])
        mean = elite.mean(axis=0)
        std  = elite.std(axis=0) + 0.05  # matches run_sepo std floor of 0.05

        print(f"  iter {it+1:>2}/{n_iter}  best={best_score:.3f}  "
              f"mean_iter={np.mean(scores):.3f}")

    return MixedPolicy(best_w)

# ─── Evaluation Harness ───────────────────────────────────────────────────────

def full_eval(policy, label):
    u = compute_utility(policy)
    e = compute_exploitability(policy)
    c = compute_collusion_risk(policy)
    x = compute_externality(policy)
    rob = compute_robustness(policy)
    welf = compute_welfare(policy)
    si = safety_index(u, e, c, x)
    return dict(label=label, payoff=u, welfare=welf,
                exploitability=e, robustness=rob,
                collusion=c, externality=x, safety=si,
                weights={ALL_STRATEGIES[i].name: round(float(policy.weights[i]), 3)
                         for i in range(N_STRAT)})

# ─── Warm Starts ──────────────────────────────────────────────────────────────
# FIX: use run_sepo_experiments.py WARMSTARTS["social_dilemma"] exactly
# These are logit vectors (passed through softmax before eval)
#        [AlwaysC, AlwaysD, TFT, GenTFT, Grim]
WARM = {
    "reward-only":  np.array([0.1,  1.6,  0.6,  0.2, 0.4]),
    "welfare-only": np.array([1.6, -0.4,  0.9,  1.2, 0.5]),
    "sepo":         np.array([0.8, -0.2,  1.4,  1.3, 0.7]),
}

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  SEPO on GTBench — Iterated Prisoner's Dilemma")
    print("=" * 65)
    print(f"  Payoff: R={R} T={T} S={S} P={P} | Rounds={N_ROUNDS} | Episodes={N_EPISODES}")
    print(f"  GTBench conventional agent: Tit-for-Tat")
    print(f"  Actions: <SILENT>=Cooperate  <TESTIFY>=Defect")
    print()

    results = []

    # ── GTBench TFT baseline (conventional solver)
    print("[1/4] GTBench conventional baseline: Tit-for-Tat")
    tft = MixedPolicy([0, 0, 1, 0, 0])
    results.append(full_eval(tft, "TFT (GTBench conventional)"))

    # ── Reward-only
    print("\n[2/4] Reward-only optimization")
    ro = cross_entropy_search(reward_only_obj, WARM["reward-only"], n_iter=18)
    results.append(full_eval(ro, "Reward-only"))

    # ── Welfare-only
    print("\n[3/4] Welfare-only optimization")
    wo = cross_entropy_search(welfare_only_obj, WARM["welfare-only"], n_iter=18)
    results.append(full_eval(wo, "Welfare-only"))

    # ── SEPO — uses sepo_obj_for_opt during search (FIX #7), full sepo_obj for eval
    print("\n[4/4] SEPO (λ_e=3.6, λ_c=3.2, λ_x=2.4) — matching run_sepo_experiments.py")
    sepo = cross_entropy_search(sepo_obj_for_opt, WARM["sepo"], n_iter=18)
    results.append(full_eval(sepo, "SEPO (full)"))

    # ── Print results table
    print()
    print("=" * 105)
    print("RESULTS — GTBench Iterated Prisoner's Dilemma (aligned with run_sepo_experiments.py)")
    print("=" * 105)
    hdr = f"{'Objective':<26} {'Payoff':>8} {'Welfare':>9} {'Exploit':>9} {'Robust':>9} {'Collusion':>10} {'Extern':>8} {'Safety':>8}"
    print(hdr); print("-" * 105)
    for r in results:
        print(f"{r['label']:<26} "
              f"{r['payoff']:>8.3f} {r['welfare']:>9.3f} "
              f"{r['exploitability']:>9.3f} {r['robustness']:>9.3f} "
              f"{r['collusion']:>10.3f} {r['externality']:>8.3f} "
              f"{r['safety']:>8.3f}")
    print("=" * 105)

    # ── Learned mixture weights
    print("\n--- Learned Strategy Mixtures ---")
    for r in results:
        print(f"\n  {r['label']}")
        for name, w in r['weights'].items():
            bar = "█" * int(w * 25)
            print(f"    {name:<25} {w:.3f}  {bar}")

    # ── Ablations — use run_sepo lambda table exactly
    print("\n--- Ablations ---")
    ablation_cfgs = [
        ("SEPO-no-exploit",    0.0, 3.2, 2.4),
        ("SEPO-no-collusion",  3.6, 0.0, 2.4),
        ("SEPO-no-externality",3.6, 3.2, 0.0),
    ]
    abl_results = []
    for name, le, lc, lx in ablation_cfgs:
        print(f"\n  {name} (λ_e={le}, λ_c={lc}, λ_x={lx})")
        p = cross_entropy_search(
            lambda pol, le=le, lc=lc, lx=lx: sepo_obj_for_opt(pol, le, lc, lx),
            WARM["sepo"], n_iter=15)
        abl_results.append(full_eval(p, name))

    print()
    print("=" * 75)
    print("Ablations — GTBench IPD")
    print("=" * 75)
    hdr2 = f"{'Objective':<26} {'Payoff':>8} {'Exploit':>9} {'Collusion':>10} {'Safety':>8}"
    print(hdr2); print("-" * 75)
    for r in abl_results:
        print(f"{r['label']:<26} "
              f"{r['payoff']:>8.3f} "
              f"{r['exploitability']:>9.3f} "
              f"{r['collusion']:>10.3f} "
              f"{r['safety']:>8.3f}")
    print("=" * 75)

    # ── Per-opponent breakdown
    print("\n--- Per-Opponent Breakdown (P1 payoff over", N_EPISODES, "episodes) ---")
    print(f"\n  {'Opponent':<25} {'Reward-only':>13} {'Welfare-only':>13} {'SEPO':>8} {'Δ(SEPO-RO)':>12}")
    print("  " + "-" * 75)
    for opp_cls in ALL_STRATEGIES:
        ro_s, wo_s, se_s = [], [], []
        opp = opp_cls()
        for ep in range(N_EPISODES):
            ro_s.append(simulate_match(ro,   opp, seed=ep)[0])
            wo_s.append(simulate_match(wo,   opp, seed=ep)[0])
            se_s.append(simulate_match(sepo, opp, seed=ep)[0])
        print(f"  vs. {opp_cls.name:<21} "
              f"{np.mean(ro_s):>13.2f} "
              f"{np.mean(wo_s):>13.2f} "
              f"{np.mean(se_s):>8.2f} "
              f"{np.mean(se_s)-np.mean(ro_s):>+12.2f}")

    print("\n✓ Done.")

    # Store final results for visualization
    import json
    with open("/mnt/user-data/outputs/sepo_ipd_results.json", "w") as f:
        json.dump({"main": results, "ablations": abl_results}, f, indent=2)
    print("Results saved to outputs.")
