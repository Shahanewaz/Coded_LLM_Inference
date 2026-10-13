"""Shared figure style: Times-metric text (Liberation Serif, TrueType) and STIX math, matching the ACL template."""
import matplotlib
from matplotlib import font_manager
import glob
for f in glob.glob("/usr/share/fonts/truetype/liberation/LiberationSerif-*.ttf"):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
PAL = {"bf16": "#1f5fa8", "fp16": "#d9661f", "fp32": "#2e8b57", "rns": "#b2182b", "ink": "#111111", "muted": "#555555"}
matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.7, "axes.edgecolor": "#222222",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "#bbbbbb", "legend.fancybox": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def style(ax):
    ax.grid(True, which="major", color="#d9d9d9", lw=0.45, ls=(0, (4, 3)))
    ax.set_axisbelow(True)
