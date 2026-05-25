"""Generate SEPO result plots from eval_results/ metrics and training logs."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

EVAL_DIR = Path(__file__).parent.parent / "eval_results"
OUT_DIR = Path(__file__).parent
LOG_DIR = Path("/tmp")


def load_metrics(model, game, checkpoint):
    path = EVAL_DIR / model / game / f"{checkpoint}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    for key in data:
        if key != "setup":
            return data[key]
    return None


def load_log(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─── Plot 1: Safety across games (Gemma 3) ───────────────────────────────────

def plot_safety_across_games():
    games = ["ipd", "auction", "negotiation", "neg_gt"]
    labels = ["IPD", "Auction", "Negotiation v1", "Negotiation GT"]
    checkpoints = {
        "ipd": "grpo_step_0075",
        "auction": "grpo_step_0025",
        "negotiation": "grpo_step_0125",
        "neg_gt": "grpo_step_0025",
    }

    base_safety, sft_safety, sepo_safety = [], [], []
    for g in games:
        b = load_metrics("gemma3", g, "base")
        s = load_metrics("gemma3", g, "sft")
        grpo = load_metrics("gemma3", g, checkpoints[g])
        base_safety.append(b["safety"] if b else 0)
        sft_safety.append(s["safety"] if s else 0)
        sepo_safety.append(grpo["safety"] if grpo else 0)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, base_safety, width, label="Base", color="#4C72B0")
    ax.bar(x, sft_safety, width, label="SFT", color="#DD8452")
    ax.bar(x + width, sepo_safety, width, label="SEPO", color="#55A868")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Game")
    ax.set_ylabel("Safety Score")
    ax.set_title("Gemma 3 — Safety Score by Game (Base vs SFT vs SEPO)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gemma3_safety_by_game.png")
    plt.close()


# ─── Plot 2: Exploitability across games (Gemma 3) ───────────────────────────

def plot_exploitability_across_games():
    games = ["ipd", "auction", "negotiation", "neg_gt"]
    labels = ["IPD", "Auction", "Negotiation v1", "Negotiation GT"]
    checkpoints = {
        "ipd": "grpo_step_0075",
        "auction": "grpo_step_0025",
        "negotiation": "grpo_step_0125",
        "neg_gt": "grpo_step_0025",
    }

    base_exp, sft_exp, sepo_exp = [], [], []
    for g in games:
        b = load_metrics("gemma3", g, "base")
        s = load_metrics("gemma3", g, "sft")
        grpo = load_metrics("gemma3", g, checkpoints[g])
        base_exp.append(b["exploitability"] if b else 0)
        sft_exp.append(s["exploitability"] if s else 0)
        sepo_exp.append(grpo["exploitability"] if grpo else 0)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, base_exp, width, label="Base", color="#4C72B0")
    ax.bar(x, sft_exp, width, label="SFT", color="#DD8452")
    ax.bar(x + width, sepo_exp, width, label="SEPO", color="#55A868")
    ax.set_xlabel("Game")
    ax.set_ylabel("Exploitability (lower is better)")
    ax.set_title("Gemma 3 — Exploitability by Game (Base vs SFT vs SEPO)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gemma3_exploitability_by_game.png")
    plt.close()


# ─── Plot 3: Gemma 4 Kuhn progression ────────────────────────────────────────

def plot_gemma4_kuhn_progression():
    checkpoints = ["base", "sft", "grpo_step_0025", "grpo_step_0050", "grpo_step_0075", "grpo_final"]
    labels = ["Base", "SFT", "Step 25", "Step 50", "Step 75", "Final"]

    safety, exploit = [], []
    for ckpt in checkpoints:
        m = load_metrics("gemma4", "kuhn", ckpt)
        safety.append(m["safety"] if m else None)
        exploit.append(m["exploitability"] if m else None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(labels, safety, "o-", color="#55A868", linewidth=2, markersize=7)
    ax1.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Safety Score")
    ax1.set_title("Gemma 4 Kuhn — Safety Progression")
    ax1.tick_params(axis="x", rotation=30)

    ax2.plot(labels, exploit, "s-", color="#C44E52", linewidth=2, markersize=7)
    ax2.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Exploitability (lower is better)")
    ax2.set_title("Gemma 4 Kuhn — Exploitability Progression")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "gemma4_kuhn_progression.png")
    plt.close()


# ─── Plot 4: Cross-model safety comparison ───────────────────────────────────

def plot_cross_model_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    # Kuhn: Gemma 4 vs Qwen
    ax = axes[0]
    g4_base = load_metrics("gemma4", "kuhn", "base")
    g4_sepo = load_metrics("gemma4", "kuhn", "grpo_final")
    qw_base = load_metrics("qwen", "kuhn", "base")
    qw_sepo = load_metrics("qwen", "kuhn", "grpo_step_0075")
    models = ["Gemma 4\nBase", "Gemma 4\nSEPO", "Qwen\nBase", "Qwen\nSEPO"]
    values = [g4_base["safety"], g4_sepo["safety"], qw_base["safety"], qw_sepo["safety"]]
    colors = ["#4C72B0", "#55A868", "#4C72B0", "#55A868"]
    ax.bar(models, values, color=colors)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_title("Kuhn Poker")
    ax.set_ylabel("Safety Score")
    ax.tick_params(axis="x", rotation=30)

    # Neg GT: all three models
    ax = axes[1]
    g3_base = load_metrics("gemma3", "neg_gt", "base")
    g3_sepo = load_metrics("gemma3", "neg_gt", "grpo_step_0025")
    g4_base = load_metrics("gemma4", "neg_gt", "base")
    g4_sepo = load_metrics("gemma4", "neg_gt", "grpo_step_0075")
    qw_base = load_metrics("qwen", "neg_gt", "base")
    qw_sepo = load_metrics("qwen", "neg_gt", "grpo_step_0075")
    models = ["G3\nBase", "G3\nSEPO", "G4\nBase", "G4\nSEPO", "Qwen\nBase", "Qwen\nSEPO"]
    values = [g3_base["safety"], g3_sepo["safety"], g4_base["safety"], g4_sepo["safety"],
              qw_base["safety"], qw_sepo["safety"]]
    colors = ["#4C72B0", "#55A868"] * 3
    ax.bar(models, values, color=colors)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_title("Negotiation GT")
    ax.tick_params(axis="x", rotation=30)

    # IPD: Gemma 3
    ax = axes[2]
    g3_base = load_metrics("gemma3", "ipd", "base")
    g3_sft = load_metrics("gemma3", "ipd", "sft")
    g3_sepo = load_metrics("gemma3", "ipd", "grpo_step_0075")
    models = ["Base", "SFT", "SEPO"]
    values = [g3_base["safety"], g3_sft["safety"], g3_sepo["safety"]]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    ax.bar(models, values, color=colors)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_title("IPD (Gemma 3)")
    ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Safety Score — Cross-Model Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cross_model_safety.png", bbox_inches="tight")
    plt.close()


# ─── Plot 5: Gemma 4 exploitability all games ────────────────────────────────

def plot_gemma4_all_games():
    games = ["ipd", "auction", "kuhn", "neg_gt", "negotiation", "resource"]
    labels = ["IPD", "Auction", "Kuhn", "Neg GT", "Negotiation", "Resource"]
    best_ckpts = {
        "ipd": "grpo_step_0025", "auction": "grpo_step_0075",
        "kuhn": "grpo_final", "neg_gt": "grpo_step_0075",
        "negotiation": "grpo_step_0075", "resource": "grpo_step_0075",
    }

    base_exp, sepo_exp, valid_labels = [], [], []
    for g, lbl in zip(games, labels):
        b = load_metrics("gemma4", g, "base")
        s = load_metrics("gemma4", g, best_ckpts[g])
        if b and s:
            base_exp.append(b["exploitability"])
            sepo_exp.append(s["exploitability"])
            valid_labels.append(lbl)

    x = np.arange(len(valid_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, base_exp, width, label="Base", color="#4C72B0")
    ax.bar(x + width / 2, sepo_exp, width, label="SEPO (best ckpt)", color="#55A868")
    ax.set_xlabel("Game")
    ax.set_ylabel("Exploitability (lower is better)")
    ax.set_title("Gemma 4 — Exploitability: Base vs SEPO across All Games")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gemma4_exploitability_all_games.png")
    plt.close()


# ─── Plot 6: SFT Degradation → SEPO Correction (waterfall) ──────────────────

def plot_sft_degradation_waterfall():
    """Shows Base → SFT (exploit increases) → SEPO (exploit decreases)."""
    games = ["ipd", "auction", "neg_gt"]
    labels = ["IPD", "Auction", "Negotiation GT"]
    checkpoints = {"ipd": "grpo_step_0075", "auction": "grpo_step_0025", "neg_gt": "grpo_step_0025"}

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x_positions = np.arange(len(games)) * 4
    width = 0.8

    for i, (g, lbl) in enumerate(zip(games, labels)):
        b = load_metrics("gemma3", g, "base")
        s = load_metrics("gemma3", g, "sft")
        sepo = load_metrics("gemma3", g, checkpoints[g])
        if not (b and s and sepo):
            continue

        base_e = b["exploitability"]
        sft_e = s["exploitability"]
        sepo_e = sepo["exploitability"]

        xb = x_positions[i]
        xs = x_positions[i] + 1.1
        xg = x_positions[i] + 2.2

        ax.bar(xb, base_e, width, color="#4C72B0")
        ax.bar(xs, sft_e, width, color="#DD8452")
        ax.bar(xg, sepo_e, width, color="#55A868")

        # Arrow: SFT degradation
        ax.annotate("", xy=(xs, sft_e), xytext=(xb, base_e),
                    arrowprops=dict(arrowstyle="->", color="#C44E52", lw=2))
        # Arrow: SEPO correction
        ax.annotate("", xy=(xg, sepo_e), xytext=(xs, sft_e),
                    arrowprops=dict(arrowstyle="->", color="#55A868", lw=2))

        ax.text(xb + 1.1, -0.15, lbl, ha="center", fontsize=11, fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_ylabel("Exploitability")
    ax.set_title("SFT Degradation → SEPO Correction (Gemma 3)")
    ax.set_xticks(x_positions + 1.1)
    ax.set_xticklabels([""] * len(games))

    legend_elements = [
        mpatches.Patch(color="#4C72B0", label="Base"),
        mpatches.Patch(color="#DD8452", label="SFT (↑ degrades)"),
        mpatches.Patch(color="#55A868", label="SEPO (↓ corrects)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sft_degradation_waterfall.png")
    plt.close()


# ─── Plot 7: Radar chart — multi-metric profile ─────────────────────────────

def plot_radar_chart():
    """Spider chart showing Base vs SEPO across all SEPO dimensions."""
    categories = ["Utility", "1 - Exploit", "1 - Externality", "Safety (norm)"]

    def normalize_for_radar(metrics):
        u = metrics.get("payoff_per_round", metrics.get("payoff_mean", 0))
        e = metrics["exploitability"]
        x = metrics["externality"]
        s = metrics["safety"]
        return [
            min(u / 3.0, 1.0),
            max(1.0 - e, 0.0),
            max(1.0 - x, 0.0),
            max(min((s + 5) / 10.0, 1.0), 0.0),
        ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), subplot_kw=dict(polar=True))
    games_data = [
        ("IPD", "gemma3", "ipd", "grpo_step_0075"),
        ("Auction", "gemma3", "auction", "grpo_step_0025"),
        ("Neg GT", "gemma3", "neg_gt", "grpo_step_0025"),
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for ax, (title, model, game, ckpt) in zip(axes, games_data):
        base = load_metrics(model, game, "base")
        sepo = load_metrics(model, game, ckpt)
        if not (base and sepo):
            continue

        base_vals = normalize_for_radar(base) + [normalize_for_radar(base)[0]]
        sepo_vals = normalize_for_radar(sepo) + [normalize_for_radar(sepo)[0]]

        ax.plot(angles, base_vals, "o-", color="#4C72B0", linewidth=2, label="Base")
        ax.fill(angles, base_vals, alpha=0.15, color="#4C72B0")
        ax.plot(angles, sepo_vals, "s-", color="#55A868", linewidth=2, label="SEPO")
        ax.fill(angles, sepo_vals, alpha=0.15, color="#55A868")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 1)
        ax.set_title(title, pad=15, fontsize=12)
        ax.legend(loc="lower right", fontsize=8)

    plt.suptitle("Multi-Metric Radar — Base vs SEPO (Gemma 3)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "radar_base_vs_sepo.png", bbox_inches="tight")
    plt.close()


# ─── Plot 8: Safety improvement heatmap (models × games) ─────────────────────

def plot_safety_heatmap():
    """Heatmap of Δsafety(SEPO − Base) across all model/game combos."""
    models = ["gemma3", "gemma4", "qwen"]
    model_labels = ["Gemma 3", "Gemma 4", "Qwen"]
    games = ["ipd", "auction", "kuhn", "neg_gt", "negotiation"]
    game_labels = ["IPD", "Auction", "Kuhn", "Neg GT", "Negotiation"]

    best_ckpts = {
        ("gemma3", "ipd"): "grpo_step_0075",
        ("gemma3", "auction"): "grpo_step_0025",
        ("gemma3", "kuhn"): None,
        ("gemma3", "neg_gt"): "grpo_step_0025",
        ("gemma3", "negotiation"): "grpo_step_0125",
        ("gemma4", "ipd"): "grpo_step_0025",
        ("gemma4", "auction"): "grpo_step_0075",
        ("gemma4", "kuhn"): "grpo_final",
        ("gemma4", "neg_gt"): "grpo_step_0075",
        ("gemma4", "negotiation"): "grpo_step_0075",
        ("qwen", "ipd"): "grpo_step_0075",
        ("qwen", "auction"): "grpo_step_0075",
        ("qwen", "kuhn"): "grpo_step_0075",
        ("qwen", "neg_gt"): "grpo_step_0075",
        ("qwen", "negotiation"): "grpo_step_0075",
    }

    data = np.full((len(models), len(games)), np.nan)
    for i, m in enumerate(models):
        for j, g in enumerate(games):
            ckpt = best_ckpts.get((m, g))
            if ckpt is None:
                continue
            base = load_metrics(m, g, "base")
            sepo = load_metrics(m, g, ckpt)
            if base and sepo:
                data[i, j] = sepo["safety"] - base["safety"]

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-3, vmax=3)

    ax.set_xticks(range(len(game_labels)))
    ax.set_xticklabels(game_labels)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels)

    for i in range(len(models)):
        for j in range(len(games)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 2 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=color, fontsize=10)

    ax.set_title("Δ Safety (SEPO − Base) by Model × Game")
    plt.colorbar(im, ax=ax, label="Safety improvement", shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "safety_improvement_heatmap.png")
    plt.close()


# ─── Plot 9: Exploit convergence over training steps ─────────────────────────

def plot_exploit_convergence():
    """Exploit over training steps for multiple runs overlaid (smoothed)."""
    logs = {
        "Gemma4 Kuhn": LOG_DIR / "grpo_gemma4_kuhn_log.json",
        "Gemma4 Multi (IPD+Auc+Neg)": LOG_DIR / "grpo_gemma4_multi_log.json",
        "Gemma4 Neg GT": LOG_DIR / "grpo_gemma4_neg_gtbench_log.json",
        "Qwen Kuhn": LOG_DIR / "grpo_qwen_kuhn_log.json",
        "Qwen Multi": LOG_DIR / "grpo_qwen_v2_final_log.json",
        "Qwen Neg GT": LOG_DIR / "grpo_qwen_neg_gt_log.json",
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#55A868", "#4C72B0", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

    window = 10

    for (name, path), color in zip(logs.items(), colors):
        log = load_log(path)
        if not log:
            continue
        steps = np.array([e["step"] for e in log])
        exploit = np.array([e["exploitability"] for e in log])

        # Raw data as faint line
        ax.plot(steps, exploit, "-", color=color, linewidth=0.5, alpha=0.25)

        # Smoothed (rolling mean)
        if len(exploit) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(exploit, kernel, mode="valid")
            ax.plot(steps[window - 1:], smoothed, "-", color=color, linewidth=2.2, label=name)
        else:
            ax.plot(steps, exploit, "-", color=color, linewidth=2, label=name)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Exploitability (10-step rolling avg)")
    ax.set_title("Exploit Convergence During SEPO Training")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(bottom=-0.05)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exploit_convergence.png")
    plt.close()


# ─── Plot 10: Per-model exploit reduction (slope chart) ──────────────────────

def plot_exploit_slope():
    """Paired dots (Base → SEPO) connected by lines for each game, per model."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    model_configs = [
        ("Gemma 3", "gemma3", {
            "IPD": "grpo_step_0075", "Auction": "grpo_step_0025",
            "Neg GT": "grpo_step_0025", "Negotiation": "grpo_step_0125",
        }),
        ("Gemma 4", "gemma4", {
            "IPD": "grpo_step_0025", "Auction": "grpo_step_0075",
            "Kuhn": "grpo_final", "Neg GT": "grpo_step_0075",
        }),
        ("Qwen", "qwen", {
            "Kuhn": "grpo_step_0075", "Neg GT": "grpo_step_0075",
        }),
    ]

    game_map = {"IPD": "ipd", "Auction": "auction", "Kuhn": "kuhn",
                "Neg GT": "neg_gt", "Negotiation": "negotiation"}

    for ax, (model_name, model_key, ckpts) in zip(axes, model_configs):
        y_pos = 0
        for game_label, ckpt in ckpts.items():
            g = game_map[game_label]
            base = load_metrics(model_key, g, "base")
            sepo = load_metrics(model_key, g, ckpt)
            if not (base and sepo):
                continue

            base_e = base["exploitability"]
            sepo_e = sepo["exploitability"]

            color = "#55A868" if sepo_e < base_e else "#C44E52"
            ax.plot([base_e, sepo_e], [y_pos, y_pos], "-", color=color, linewidth=2.5, alpha=0.7)
            ax.plot(base_e, y_pos, "o", color="#4C72B0", markersize=9, zorder=5)
            ax.plot(sepo_e, y_pos, "s", color="#55A868", markersize=9, zorder=5)
            ax.text(-0.05, y_pos, game_label, ha="right", va="center", fontsize=10,
                    transform=ax.get_yaxis_transform())
            y_pos += 1

        ax.set_yticks([])
        ax.set_xlabel("Exploitability")
        ax.set_title(model_name)
        ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=10, label="Base"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#55A868", markersize=10, label="SEPO"),
    ]
    axes[2].legend(handles=legend_elements, loc="lower right")

    plt.suptitle("Exploitability Reduction: Base → SEPO (per model)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exploit_slope_chart.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_safety_across_games()
    print("1/10 gemma3_safety_by_game.png")

    plot_exploitability_across_games()
    print("2/10 gemma3_exploitability_by_game.png")

    plot_gemma4_kuhn_progression()
    print("3/10 gemma4_kuhn_progression.png")

    plot_cross_model_comparison()
    print("4/10 cross_model_safety.png")

    plot_gemma4_all_games()
    print("5/10 gemma4_exploitability_all_games.png")

    plot_sft_degradation_waterfall()
    print("6/10 sft_degradation_waterfall.png")

    plot_radar_chart()
    print("7/10 radar_base_vs_sepo.png")

    plot_safety_heatmap()
    print("8/10 safety_improvement_heatmap.png")

    plot_exploit_convergence()
    print("9/10 exploit_convergence.png")

    plot_exploit_slope()
    print("10/10 exploit_slope_chart.png")

    print(f"\nAll plots saved to {OUT_DIR}/")
