"""Figs. 6-9 of the supplementary material.

The figures are drawn at their printed size in the compact 6-pt style of
make_panels_r1.py.

  figS6_c   : detector validation (make_fig6.panel_a), 50 x 28 mm, single column
  figS7_row : ablations (make_fig6 panels b, c, d), 183 x 32 mm, three panels
  figS8_row : bearing (make_fig7 panels a, b, c), 183 x 34 mm, three panels
  figS9_c   : real white noise near DC (make_figS.panel_realnoise), 60 x 26 mm

Outputs: figures/rows3_r1/<name>.{pdf,png}
"""
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from zerospec.figstyle import MM, savefig
import make_panels_r1 as mp          # sets the 6-pt compact rcParams
import make_fig6 as f6
import make_fig7 as f7
import make_figS as fS
import matplotlib as mpl

# the figure modules call figstyle.set_style() on import; restore the compact style
mpl.rcParams.update({
    "font.size": 6.0, "axes.labelsize": 6.0, "axes.titlesize": 6.2,
    "xtick.labelsize": 5.4, "ytick.labelsize": 5.4, "legend.fontsize": 4.9,
    "lines.linewidth": 0.85, "lines.markersize": 2.4,
    "legend.handlelength": 1.2, "legend.labelspacing": 0.25,
    "legend.borderaxespad": 0.2, "legend.columnspacing": 0.8,
    "legend.handletextpad": 0.5,
    "axes.labelpad": 1.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
})

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "figures", "rows3_r1")

ROWS = {
    "figS7_row": ([f6.panel_b, f6.panel_c, f6.panel_d], 183 * MM, 32 * MM, [1, 1, 1]),
    "figS8_row": ([f7.panel_a, f7.panel_b, f7.panel_c], 183 * MM, 34 * MM, [1.25, 1, 1]),
}
SINGLES = {
    "figS6_c": (f6.panel_a, 52 * MM, 28 * MM),
    "figS9_c": (fS.panel_realnoise, 52 * MM, 22.5 * MM),
}


def label(ax, letter):
    ax.text(-0.22, 1.0, "(%s)" % letter, transform=ax.transAxes, fontsize=7.5,
            fontweight="bold", va="bottom", ha="left")


def build_row(name):
    fns, w, h, ratios = ROWS[name]
    fig = plt.figure(figsize=(w, h))
    gs = GridSpec(1, len(fns), figure=fig, width_ratios=ratios, wspace=0.03)
    for k, fn in enumerate(fns):
        ax = fig.add_subplot(gs[0, k])
        fn(fig, ax)
        label(ax, "abc"[k])
    savefig(fig, name, OUT)
    plt.close(fig)


def detector_two_axes(fig, w):
    """SM Fig. 6: white-noise density against the grid step (left) and against
    the window truncation (right), on separate x-axes with a shared y-axis."""
    import numpy as np
    from zerospec.figstyle import OI
    ab = f6.ab
    gs = GridSpec(1, 2, figure=fig, wspace=0.08)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1], sharey=axL)
    steps = np.array(ab["a1_steps"], float)
    axL.errorbar(np.arange(len(steps)), ab["a1_vals"], yerr=2 * np.array(ab["a1_se"]),
                 fmt="o-", ms=2.6, lw=0.9, color=OI["blue"], capsize=1.2)
    axL.set_xticks(np.arange(len(steps)))
    axL.set_xticklabels(["1/%d" % round(1 / s) for s in steps])
    axL.set_xlim(-0.4, len(steps) - 0.6)
    axL.set_xlabel(r"grid step $\delta$")
    axL.set_ylabel("white-noise density")
    half = np.array(ab["a4_half"], float)
    axR.errorbar(np.arange(len(half)), ab["a4_vals"], yerr=2 * np.array(ab["a4_se"]),
                 fmt="s-", ms=2.6, lw=0.9, color=OI["green"], capsize=1.2)
    axR.set_xticks(np.arange(len(half)))
    axR.set_xticklabels([r"$\pm%d$" % h for h in half])
    axR.set_xlim(-0.4, len(half) - 0.6)
    axR.set_xlabel("window truncation")
    plt.setp(axR.get_yticklabels(), visible=False)
    for ax in (axL, axR):
        ax.axhline(1.0, color=OI["grey"], lw=0.6, ls=":")
    axL.set_ylim(0.99, 1.01)


def build_single(name):
    fn, w, h = SINGLES[name]
    fig = plt.figure(figsize=(w, h))
    if name == "figS6_c":
        detector_two_axes(fig, w)
    else:
        ax = fig.add_subplot(111)
        fn(fig, ax)
    savefig(fig, name, OUT)
    plt.close(fig)


if __name__ == "__main__":
    names = sys.argv[1:] or list(ROWS) + list(SINGLES)
    for n in names:
        (build_row if n in ROWS else build_single)(n)
    print("sm panels done")
