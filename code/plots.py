"""Figures for the paper (matplotlib, PDF, single-column width)."""
import json, os, glob, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/claude/code/results"
OUT = "/home/claude/paper"
from pstyle import PAL, style
COL = {"bf16": PAL["bf16"], "fp16": PAL["fp16"], "fp32": PAL["fp32"]}
MK = {"bf16": "o", "fp16": "s", "fp32": "^"}
NAME = {"bf16": "BF16", "fp16": "FP16", "fp32": "FP32"}
INK, MUTED = PAL["ink"], PAL["muted"]


def fig_barrier():
    from matplotlib.lines import Line2D
    d = json.load(open(os.path.join(RES, "barrier.json")))
    tg = json.load(open(os.path.join(RES, "tightness_gaussian.json")))
    rows = d["rows"]
    beta = np.array([r["beta"] for r in rows])
    fig, ax = plt.subplots(figsize=(3.05, 2.3))
    for f in ["bf16", "fp16", "fp32"]:
        y = np.array([r[f] for r in rows]); ok = np.isfinite(y)
        ax.plot(beta[ok], y[ok], color=COL[f], lw=1.2, alpha=0.9)
        t = [r for r in tg if r["fmt"] == f]
        tb = np.array([r["beta"] for r in t]); tr = np.array([r["rel_mse"] for r in t]); bd = np.array([r["bound"] for r in t])
        ax.plot(tb, bd, color=COL[f], ls=(0, (3, 2)), lw=1.0)
        ax.scatter(tb, tr, s=16, facecolors="none", edgecolors=COL[f], marker=MK[f], linewidths=1.0, zorder=3)
    ax.axhline(d["w8a8_rel_mse"], color=PAL["rns"], lw=1.1, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Mask-to-signal ratio $\beta$")
    ax.set_ylabel("Relative MSE of decoded product")
    ax.set_ylim(1e-17, 1e12)
    ax.set_yticks([1e-15, 1e-10, 1e-5, 1e0, 1e5])
    style(ax)
    h = [Line2D([], [], color=COL[f], lw=2) for f in COL] + [
        Line2D([], [], color=MUTED, ls=(0, (3, 2)), lw=1.0),
        Line2D([], [], color=MUTED, marker="o", ls="none", mfc="none", ms=4),
        Line2D([], [], color=MUTED, lw=1.2),
        Line2D([], [], color=PAL["rns"], ls=":", lw=1.1)]
    lab = [NAME[f] for f in COL] + ["Bound, Thm. 1(a)", "Tight code, Thm. 1(b)", "Optimized code, GPT-2", "RNS-Coded (0 nats)"]
    ax.legend(h, lab, loc="upper left", fontsize=6.6, ncol=2, handlelength=1.6, columnspacing=0.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_barrier.pdf")); fig.savefig(os.path.join(OUT, "fig_barrier.png"), dpi=200)


def load_ppl(pattern):
    out = {}
    for f in glob.glob(os.path.join(RES, pattern)):
        d = json.load(open(f)); out[d["cfg"]] = d
    return out


def fig_ppl():
    from coded import gaussian_leakage_per_entry
    from matplotlib.lines import Line2D
    E = np.load("/home/claude/code/realcode_8_4_2.npy")
    n_worker = 512 * (3 * 768 + 3072) * 12 // 4        # share entries per worker per 512-token request
    runs = load_ppl("ppl_*_w16_l0.json")
    fig, ax = plt.subplots(figsize=(3.05, 2.3))
    TOP = 1e6
    for f in ["bf16", "fp16", "fp32"]:
        pts = []
        for cfg, d in runs.items():
            if cfg.startswith(f"analog:{f},"):
                b = float(cfg.split(",")[1])
                leak = gaussian_leakage_per_entry(E, 4, 2, b) * n_worker
                pts.append((leak, d["ppl"] if d["nonfinite_windows"] == 0 else np.nan, b))
        pts.sort()
        L, P, B = map(np.array, zip(*pts))
        ok = np.isfinite(P)
        ax.plot(L[ok], np.minimum(P[ok], TOP), color=COL[f], marker=MK[f], ms=3.2, label=NAME[f] + " analog")
        if f == "bf16":
            for l, p_, b in zip(L[ok], P[ok], B[ok]):
                ax.annotate(f"$\\beta$={b:g}", (l, min(p_, TOP)), textcoords="offset points", xytext=(4, 2), fontsize=5.6, color=MUTED)
        if (~ok).any():
            ax.scatter(L[~ok], np.full((~ok).sum(), 1.3e3), marker="x", color=COL[f], s=16, lw=1.0)
            ax.annotate("FP16 overflow", (L[~ok].min(), 1.3e3), textcoords="offset points", xytext=(-4, -10), fontsize=5.6, color=COL["fp16"])
    ref = runs["w8a8"]
    ax.axhline(ref["ppl"], color=PAL["rns"], lw=1.1, ls=":")
    H = 512 * np.log(runs["fp32"]["ppl"])
    ax.axvline(H, color=MUTED, lw=0.6, ls="--")
    ax.text(H * 1.25, 4e4, "prompt\nentropy", fontsize=5.8, color=MUTED, va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(18, TOP * 1.5); ax.set_xlim(1, 1e8)
    ax.set_xlabel("Certified leakage to worst pair (nats per request)")
    ax.set_ylabel("WikiText-2 perplexity")
    style(ax)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=PAL["rns"], ls=":", lw=1.1)); l.append(f"RNS-Coded (0 nats)")
    ax.legend(h, l, loc="upper right", fontsize=6.6, handlelength=1.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_ppl.pdf")); fig.savefig(os.path.join(OUT, "fig_ppl.png"), dpi=200)


if __name__ == "__main__":
    which = sys.argv[1:] or ["barrier", "ppl"]
    if "barrier" in which: fig_barrier()
    if "ppl" in which: fig_ppl()
