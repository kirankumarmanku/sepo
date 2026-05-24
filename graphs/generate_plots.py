"""Generate SEPO result plots from eval_results/ metrics."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

EVAL_DIR = Path(__file__).parent.parent / "eval_results"
OUT_DIR = Path(__file__).parent


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


def plot_safety_across_games():
    """Bar chart: Safety score across games for Base vs SFT vs SEPO (Gemma 3)."""
    games = ["ipd", "auction", "negotiation", "neg_gt"]
    labels = ["IPD", "Auction", "Negotiation v1", "Negotiation GT"]
    checkpoints = {
        "ipd": "grpo_step_0075",
        "auction": "grpo_step_0025",
        "negotiation": "grpo_step_0125",
        "neg_gt": "grpo_step_0025",
    }

    base_safety = []
    sft_safety = []
    sepo_safety = []

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

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Game")
    ax.set_ylabel("Safety Score")
    ax.set_title("Gemma 3 — Safety Score by Game (Base vs SFT vs SEPO)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gemma3_safety_by_game.png")
    plt.close()


def plot_exploitability_across_games():
    """Bar chart: Exploitability across games for Base vs SFT vs SEPO (Gemma 3)."""
    games = ["ipd", "auction", "negotiation", "neg_gt"]
    labels = ["IPD", "Auction", "Negotiation v1", "Negotiation GT"]
    checkpoints = {
        "ipd": "grpo_step_0075",
        "auction": "grpo_step_0025",
        "negotiation": "grpo_step_0125",
        "neg_gt": "grpo_step_0025",
    }

    base_exp = []
    sft_exp = []
    sepo_exp = []

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


def plot_gemma4_kuhn_progression():
    """Line chart: Gemma 4 Kuhn safety/exploit across checkpoints."""
    checkpoints = ["base", "sft", "grpo_step_0025", "grpo_step_0050", "grpo_step_0075", "grpo_final"]
    labels = ["Base", "SFT", "Step 25", "Step 50", "Step 75", "Final"]

    safety = []
    exploit = []
    for ckpt in checkpoints:
        m = load_metrics("gemma4", "kuhn", ckpt)
        if m:
            safety.append(m["safety"])
            exploit.append(m["exploitability"])
        else:
            safety.append(None)
            exploit.append(None)

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


def plot_cross_model_comparison():
    """Grouped bar: Safety across models (Gemma 3, Gemma 4, Qwen) for key games."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    # IPD — Gemma 3 only (others don't have comparable IPD results)
    # Kuhn — Gemma 4 and Qwen
    # Neg GT — all three

    # Kuhn: Gemma 4 vs Qwen
    ax = axes[0]
    g4_base = load_metrics("gemma4", "kuhn", "base")
    g4_sepo = load_metrics("gemma4", "kuhn", "grpo_final")
    qw_base = load_metrics("qwen", "kuhn", "base")
    qw_sepo = load_metrics("qwen", "kuhn", "grpo_step_0075")

    models = ["Gemma 4\nBase", "Gemma 4\nSEPO", "Qwen\nBase", "Qwen\nSEPO"]
    values = [
        g4_base["safety"] if g4_base else 0,
        g4_sepo["safety"] if g4_sepo else 0,
        qw_base["safety"] if qw_base else 0,
        qw_sepo["safety"] if qw_sepo else 0,
    ]
    colors = ["#4C72B0", "#55A868", "#4C72B0", "#55A868"]
    ax.bar(models, values, color=colors)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_title("Kuhn Poker")
    ax.set_ylabel("Safety Score")
    ax.tick_params(axis="x", rotation=30)

    # Neg GT: Gemma 3, Gemma 4, Qwen
    ax = axes[1]
    g3_base = load_metrics("gemma3", "neg_gt", "base")
    g3_sepo = load_metrics("gemma3", "neg_gt", "grpo_step_0025")
    g4_base = load_metrics("gemma4", "neg_gt", "base")
    g4_sepo = load_metrics("gemma4", "neg_gt", "grpo_step_0075")
    qw_base = load_metrics("qwen", "neg_gt", "base")
    qw_sepo = load_metrics("qwen", "neg_gt", "grpo_step_0075")

    models = ["G3\nBase", "G3\nSEPO", "G4\nBase", "G4\nSEPO", "Qwen\nBase", "Qwen\nSEPO"]
    values = [
        g3_base["safety"] if g3_base else 0,
        g3_sepo["safety"] if g3_sepo else 0,
        g4_base["safety"] if g4_base else 0,
        g4_sepo["safety"] if g4_sepo else 0,
        qw_base["safety"] if qw_base else 0,
        qw_sepo["safety"] if qw_sepo else 0,
    ]
    colors = ["#4C72B0", "#55A868"] * 3
    ax.bar(models, values, color=colors)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_title("Negotiation GT")
    ax.tick_params(axis="x", rotation=30)

    # IPD: Gemma 3 only
    ax = axes[2]
    g3_base = load_metrics("gemma3", "ipd", "base")
    g3_sft = load_metrics("gemma3", "ipd", "sft")
    g3_sepo = load_metrics("gemma3", "ipd", "grpo_step_0075")

    models = ["Base", "SFT", "SEPO"]
    values = [
        g3_base["safety"] if g3_base else 0,
        g3_sft["safety"] if g3_sft else 0,
        g3_sepo["safety"] if g3_sepo else 0,
    ]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    ax.bar(models, values, color=colors)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_title("IPD (Gemma 3)")
    ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Safety Score — Cross-Model Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cross_model_safety.png", bbox_inches="tight")
    plt.close()


def plot_gemma4_all_games():
    """Gemma 4: Exploitability reduction across all games."""
    games = ["ipd", "auction", "kuhn", "neg_gt", "negotiation", "resource"]
    labels = ["IPD", "Auction", "Kuhn", "Neg GT", "Negotiation", "Resource"]
    best_ckpts = {
        "ipd": "grpo_step_0025",
        "auction": "grpo_step_0075",
        "kuhn": "grpo_final",
        "neg_gt": "grpo_step_0075",
        "negotiation": "grpo_step_0075",
        "resource": "grpo_step_0075",
    }

    base_exp = []
    sepo_exp = []
    valid_labels = []

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


if __name__ == "__main__":
    plot_safety_across_games()
    print("Generated: gemma3_safety_by_game.png")

    plot_exploitability_across_games()
    print("Generated: gemma3_exploitability_by_game.png")

    plot_gemma4_kuhn_progression()
    print("Generated: gemma4_kuhn_progression.png")

    plot_cross_model_comparison()
    print("Generated: cross_model_safety.png")

    plot_gemma4_all_games()
    print("Generated: gemma4_exploitability_all_games.png")

    print(f"\nAll plots saved to {OUT_DIR}/")
