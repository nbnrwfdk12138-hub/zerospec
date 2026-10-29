"""Figure style for the paper.

Widths of 89 mm (one column) and 183 mm (two columns), 7-pt base font, open
axes with outward ticks, the Okabe-Ito color palette, bold lowercase panel
letters, and vector PDF plus 600-dpi PNG output.
"""
from __future__ import annotations
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
W1 = 89 * MM            # single column
W2 = 183 * MM           # double column

# Okabe-Ito (colorblind safe)
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermilion="#D55E00", purple="#CC79A7",
          grey="#7F7F7F")
CYCLE = [OI["blue"], OI["vermilion"], OI["green"], OI["orange"], OI["purple"],
         OI["sky"], OI["black"]]


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 600,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "pdf.fonttype": 42,            # editable text in PDF
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True,
        "figure.constrained_layout.use": True,
    })


def panel_label(ax, letter, dx=-0.02, dy=1.02, fs=8.5):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fs,
            fontweight="bold", va="bottom", ha="right", family="Arial")


def savefig(fig, name, figdir):
    os.makedirs(figdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"), bbox_inches=None)
    print(f"[fig] saved {name}.pdf/.png -> {figdir}")
