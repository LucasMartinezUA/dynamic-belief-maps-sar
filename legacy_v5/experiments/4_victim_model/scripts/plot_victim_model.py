#!/usr/bin/env python3
"""Experiment 4: Plot victim mobility model robustness (single columnwidth, IEEE RAL).

Panel (a): Likelihood invariance — strip overlaid distributions across three
    mobility models, confirming that L (Eq. 8) is independent of the victim's
    realized position by construction.
Panel (b): victims_found — pooled over all five planning modes (n=500 per model).
    Lost Person (Koester) is significantly the hardest to find (p<1e-5 vs both
    Random Walk and Route Following); Random Walk vs Route Following: n.s.
    Planners do not differ significantly within any model (n=100).

Input:   ../study_victim_model_20260531_095441.csv
Output:  results/victim_model_study.pdf
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR = SCRIPT_DIR.parent
CSV_PATH = STUDY_DIR / "study_victim_model_20260531_095441.csv"
OUTDIR = STUDY_DIR / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTDIR / "victim_model_study.pdf"

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=0.85)
plt.rcParams.update(
    {
        "font.family": "serif",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

# ── Shared constants ─────────────────────────────────────────────────────────
VICTIM_MODELS = ["random_walk", "route_following", "lost_person"]
MODEL_LABELS = ["Random\nWalk", "Route\nFollowing", "Lost Person\n(Koester)"]
MODEL_COLORS = ["#3498db", "#2ecc71", "#e74c3c"]
MODEL_COLOR_MAP = dict(zip(VICTIM_MODELS, MODEL_COLORS))

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["model_label"] = df["victim_model"].apply(
    lambda m: dict(zip(VICTIM_MODELS, MODEL_LABELS))[m]
)
df["model_label_flat"] = df["victim_model"].apply(
    lambda m: {"random_walk": "Random Walk", "route_following": "Route Following", "lost_person": "Lost Person\n(Koester)"}[m]
)

# ── Pre-compute pairwise stat annotations for panel (b) ─────────────────────
# Pooled across all 5 planners for each victim model (n=500 each)
pooled = {}
for m in VICTIM_MODELS:
    pooled[m] = sorted(
        df[df["victim_model"] == m]["victims_found"].values
    )

pairwise_results = {}
pairs_to_annotate = [
    ("random_walk", "lost_person"),
    ("route_following", "lost_person"),
    ("random_walk", "route_following"),
]
# Map pair to x-indices
idx_map = {m: i for i, m in enumerate(VICTIM_MODELS)}

for ma, mb in pairs_to_annotate:
    n_a, n_b = len(pooled[ma]), len(pooled[mb])
    u, p = stats.mannwhitneyu(pooled[ma], pooled[mb], alternative="two-sided")
    # Cliff's delta from Mann-Whitney U
    u1 = stats.mannwhitneyu(pooled[ma], pooled[mb], alternative="greater").statistic
    delta = 2 * u1 / (n_a * n_b) - 1  # positive → ma > mb
    sig = p < 0.001
    label = f"δ={delta:+.3f}, p={'<10⁻⁴' if p < 1e-4 else f'{p:.0e}'}"
    pairwise_results[(ma, mb)] = {
        "delta": delta,
        "p": p,
        "sig": sig,
        "label": label,
        "stars": "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")),
    }

# ── Build figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2, figsize=(7.16, 3.0), gridspec_kw={"width_ratios": [1, 1.0]}
)

# ═══════════════════════════════════════════════════════════════════════════
# PANEL (a): Likelihood invariance
# ═══════════════════════════════════════════════════════════════════════════
ax_a = axes[0]
rng = np.random.default_rng(42)
for i, model in enumerate(VICTIM_MODELS):
    subset = df[df["victim_model"] == model]
    x = np.full(len(subset), i) + rng.uniform(-0.18, 0.18, len(subset))
    ax_a.scatter(
        x, subset["likelihood"],
        s=1.2, alpha=0.25, color="#555555", edgecolors="none", rasterized=True,
    )

ax_a.set_xticks(range(len(VICTIM_MODELS)))
ax_a.set_xticklabels(MODEL_LABELS, fontsize=7)
ax_a.set_ylabel("Likelihood  $\\mathcal{L}$", fontsize=8)
ax_a.set_ylim(0.24, 0.46)
ax_a.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax_a.set_title("(a)  $\\mathcal{L}$ invariance", fontsize=9, pad=6)
ax_a.text(
    0.5, 0.02,
    "$\\mathcal{L}$ identical across models\n(coverage-only metric, Eq.~8)",
    transform=ax_a.transAxes, ha="center", va="bottom",
    fontsize=6.5, style="italic", color="#666666",
)

# ═══════════════════════════════════════════════════════════════════════════
# PANEL (b): victims_found — POOLED over planners, grouped by victim model
# ═══════════════════════════════════════════════════════════════════════════
ax_b = axes[1]

# Violin: x = victim model (pooled, n=500 each), no hue
# (hue=self + legend=False suppresses the future deprecation warning)
violin_parts = sns.violinplot(
    data=df,
    x="model_label",
    y="victims_found",
    hue="model_label",
    order=MODEL_LABELS,
    hue_order=MODEL_LABELS,
    palette=MODEL_COLORS,
    cut=0,
    inner="quartile",
    linewidth=0.5,
    saturation=0.85,
    legend=False,
    ax=ax_b,
)

# Mean markers
for xi, model_label in enumerate(MODEL_LABELS):
    model = VICTIM_MODELS[xi]
    vals = df[df["victim_model"] == model]["victims_found"]
    mean_v = vals.mean()
    ax_b.plot(
        xi, mean_v, "D",
        color="black", markersize=4.5,
        markeredgewidth=0.5, markeredgecolor="white", zorder=10,
    )
    # Mean label
    ax_b.annotate(
        f"{mean_v:.2f}",
        xy=(xi, mean_v),
        xytext=(xi + 0.25, mean_v + 0.22),
        fontsize=6, fontweight="bold", color="black",
    )

# Significance brackets between models
y_max = df["victims_found"].max() + 0.3
y_bracket_base = y_max + 0.15
bracket_height = 0.08

# random_walk vs lost_person (x=0, x=2)
sig_info = pairwise_results[("random_walk", "lost_person")]
y0 = y_bracket_base
ax_b.plot([0, 0, 2, 2], [y0, y0 + bracket_height, y0 + bracket_height, y0], "k-", lw=0.6)
ax_b.text(1, y0 + bracket_height + 0.04, sig_info["stars"], ha="center", fontsize=6.5, fontweight="bold", color="#e74c3c")
# route_following vs lost_person (x=1, x=2)
sig_info2 = pairwise_results[("route_following", "lost_person")]
y1 = y_bracket_base + 0.22
ax_b.plot([1, 1, 2, 2], [y1, y1 + bracket_height, y1 + bracket_height, y1], "k-", lw=0.6)
ax_b.text(1.5, y1 + bracket_height + 0.04, sig_info2["stars"], ha="center", fontsize=6.5, fontweight="bold", color="#e74c3c")
# random_walk vs route_following (x=0, x=1)
sig_info3 = pairwise_results[("random_walk", "route_following")]
y2 = y_bracket_base + 0.44
ax_b.plot([0, 0, 1, 1], [y2, y2 + bracket_height, y2 + bracket_height, y2], "k-", lw=0.6)
ax_b.text(0.5, y2 + bracket_height + 0.04, sig_info3["stars"], ha="center", fontsize=6.5, fontweight="bold", color="#95a5a6")

ax_b.set_ylabel("Victims found  (pooled, $n{=}500$)", fontsize=8)
ax_b.set_xlabel("")
ax_b.set_title("(b)  Findability by victim model", fontsize=9, pad=6)
ax_b.set_ylim(-0.2, y2 + 0.6)

# Annotation at bottom: within-model note
ax_b.text(
    0.5, 0.015,
    "Planners n.s. within each model ($n{=}100$)",
    transform=ax_b.transAxes, ha="center", va="bottom",
    fontsize=6, style="italic", color="#666666",
)
# Annotation for lost_person significance
ax_b.annotate(
    "Lost Person (Koester)\nsignificantly hardest\n($p{<}10^{-5}$ vs RW, $p{<}10^{-3}$ vs RF)",
    xy=(2, 1.55), xytext=(2.8, 2.3),
    fontsize=5.5, color="#e74c3c", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.6),
)

# ── Save ─────────────────────────────────────────────────────────────────────
fig.tight_layout(w_pad=0.8)
fig.savefig(OUTPUT)
print(f"Saved → {OUTPUT}")
