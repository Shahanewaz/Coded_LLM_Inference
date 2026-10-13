"""Fig. 3: privacy (neural-attack token recovery, layer 0, worst colluding pair) vs quality (WikiText-2 PPL)
for Qwen2.5-1.5B. All numbers are measured (results2_privacy.json, results_gpu.json); none is modeled."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pstyle import PAL, style

# (token recovery %, PPL on the same 20-window subset)
UNCODED = (99.9, 9.49)
BF16 = [(75.7, 9.51, 1), (27.0, 10.57, 10), (1.6, 48079, 100)]
FP16 = (1.6, 11.77, 100)
RNS = (1.7, 9.74)
PRIOR = 4.8

fig, ax = plt.subplots(figsize=(3.05, 2.5))
ax.axvspan(0.9, PRIOR, color="#eeeeee", zorder=0, lw=0)
ax.axvline(PRIOR, color="#888888", lw=0.7, ls=(0, (1, 1.5)), zorder=1)
ax.text(PRIOR * 0.93, 3.2e5, "below the\nmost-frequent-\ntoken prior", fontsize=6.6, color="#666666",
        ha="right", va="center", linespacing=0.95)
ax.plot([p[0] for p in BF16], [p[1] for p in BF16], color=PAL["bf16"], lw=1.0, ls=(0, (4, 2)), zorder=2)
ax.scatter([p[0] for p in BF16], [p[1] for p in BF16], s=26, color=PAL["bf16"], marker="o", zorder=3,
           edgecolors="white", linewidths=0.5)
for x, y, b in BF16:
    off = {1: (5, 7), 10: (6, 4), 100: (6, -1)}[b]
    ax.annotate(rf"$\beta{{=}}{b}$", (x, y), textcoords="offset points", xytext=off, fontsize=7.2, color=PAL["bf16"])
ax.scatter([FP16[0]], [FP16[1]], s=24, color=PAL["fp16"], marker="^", zorder=5, edgecolors="black", linewidths=0.4)
ax.annotate(r"FP16, $\beta{=}100$ (PPL 11.77)", (FP16[0], FP16[1]), xytext=(1.12, 55), fontsize=7.0, color=PAL["fp16"], va="center", bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9),
            arrowprops=dict(arrowstyle="-", lw=0.5, color=PAL["fp16"], shrinkA=1, shrinkB=2.5))
ax.annotate("RNS-Coded (PPL 9.74)", (RNS[0], RNS[1]), xytext=(2.3, 6.1), fontsize=7.0, color=PAL["rns"], va="center", bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9),
            arrowprops=dict(arrowstyle="-", lw=0.5, color=PAL["rns"], shrinkA=1, shrinkB=3.5))
ax.scatter([UNCODED[0]], [UNCODED[1]], s=28, color=PAL["ink"], marker="s", zorder=3)
ax.scatter([RNS[0]], [RNS[1]], s=85, color=PAL["rns"], marker="*", zorder=4, edgecolors="white", linewidths=0.4)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.9, 200); ax.set_ylim(5, 1.5e6)
ax.set_xlabel("Layer-0 tokens recovered by the attacker (%)")
ax.set_ylabel("WikiText-2 perplexity")
style(ax)
h = [Line2D([], [], color=PAL["ink"], marker="s", ls="none", ms=4.5),
     Line2D([], [], color=PAL["bf16"], marker="o", ls=(0, (4, 2)), lw=1.0, ms=4.5),
     Line2D([], [], color=PAL["fp16"], marker="^", ls="none", ms=5),
     Line2D([], [], color=PAL["rns"], marker="*", ls="none", ms=8)]
lab = ["Uncoded worker (BF16)", "Analog, BF16 shares", "Analog, FP16 shares", "RNS-Coded (W8A8)"]
ax.legend(h, lab, loc="upper right", fontsize=6.9, handlelength=1.9, borderpad=0.45, labelspacing=0.35)
fig.tight_layout(pad=0.25)
for e in ["pdf", "png"]:
    fig.savefig(f"/home/claude/paper/fig_frontier.{e}", dpi=300)
