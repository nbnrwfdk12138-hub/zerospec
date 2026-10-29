"""Figs. 1-5 of the paper as rows of three panels.

Each figure is one PDF at its printed size (183 mm wide, 56 mm tall), so that
LaTeX rescales nothing.  The panel functions are the compact variants of
make_panels_r1.py (6-pt baseline); every text size is multiplied by a common
factor K at creation time (rcParams for the defaults, a patch of the explicit
``fontsize`` / ``labelsize`` keyword arguments for the rest), so the relative
sizes tuned for the compact panels are preserved while axis labels become
7.8 pt and tick labels 7 pt.

Composition (panel letters are drawn in the figure):
  fig1_row3 : (a) bimodal density      (b) modulated density   (c) tone density
  fig2_row3 : (a) spectrogram + zeros  (b) one record          (c) sup-error vs T
  fig3_row3 : (a) chirp+tone, mask     (b) full-band bias      (c) bias at tone rows
  fig4_row3 : (a) GW150914 spectrogram (b) no-injection ASD    (c) swept-band bias
  fig5_row3 : (a) two estimands        (b) DKITCHEN            (c) PCAFETER

Single panels at the same scale are also written to figures/panels3_r1/.

Outputs: figures/rows3_r1/<name>.{pdf,png}, figures/panels3_r1/<panel>.{pdf,png}
"""
import os
import sys
import matplotlib as mpl
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from zerospec.figstyle import MM, savefig
import make_panels_r1 as mp

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "figures", "rows3_r1")
OUT1 = os.path.join(HERE, "..", "figures", "panels3_r1")
FN = {name: fn for name, fn, _ in mp.PANELS}

K = 1.3            # text scale relative to the 6-pt compact baseline
LW = 1.1           # line-width and marker scale
W2 = 183 * MM
H = 54 * MM
COLORBAR = {"fig1_a", "fig3_a", "fig4_a", "fig5_a"}

mpl.rcParams.update({
    "font.size": 6.0 * K, "axes.labelsize": 6.0 * K, "axes.titlesize": 6.2 * K,
    "xtick.labelsize": 5.4 * K, "ytick.labelsize": 5.4 * K,
    "legend.fontsize": 4.9 * K,
    "lines.linewidth": 0.85 * LW, "lines.markersize": 2.4 * LW,
})


def _patch(cls, name, key):
    orig = getattr(cls, name)

    def wrapper(self, *args, **kw):
        v = kw.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            kw[key] = v * K
        return orig(self, *args, **kw)

    wrapper.__name__ = name
    setattr(cls, name, wrapper)


for _name, _key in (("legend", "fontsize"), ("text", "fontsize"),
                    ("annotate", "fontsize"), ("set_title", "fontsize"),
                    ("tick_params", "labelsize")):
    _patch(maxes.Axes, _name, _key)

ROWS = {
    "fig1_row3": (["fig1_b", "fig1_c", "fig1_d"], [1, 1, 1]),
    "fig2_row3": (["fig1_a", "fig2_a", "fig2_b"], [1.18, 1, 1]),
    "fig3_row3": (["fig3_a", "fig3_c", "fig3_d"], [1.18, 1, 1]),
    "fig4_row3": (["fig4_a", "fig4_b", "fig4_d"], [1.18, 1, 1]),
    "fig5_row3": (["fig5_b", "fig5_c", "fig5_d"], [1, 1, 1]),
}


# panel variants re-tuned for the larger panels
def fig1_b(fig, ax):
    mp.f1.panel_b(fig, ax)
    for t in ax.texts:                      # rms label: lower right, clear of the points
        if t.get_text().startswith("rms"):
            t.set_transform(ax.transAxes)
            t.set_position((0.97, 0.04))
            t.set_ha("right")


def fig1_d(fig, ax):
    mp.fig1_panel_d(fig, ax)
    ax.set_ylim(1.2e-3, 30)                 # head-room for the legend above the ring
    ax.legend(loc="upper left", fontsize=4.6, handlelength=1.0)
    for t in ax.texts:                      # e^{-gamma_max}: label from the lower left
        if "gamma" in t.get_text():
            t.xy = (-0.05, 4.2e-3)
            t.set_position((-1.7, 2.2e-3))
            t.set_ha("left")
        if t.get_text() == "ring":          # point at the theoretical ring peak
            t.xy = (0.644, 2.09)
            t.set_position((1.0, 6.0))


def fig2_a(fig, ax):
    mp.fig2_panel_a(fig, ax)
    ax.set_ylim(-13.5, 3.5)                 # room for the two-column legend


def _env(fig, ax, env, gap, legend):
    mp.f5._env_panel(ax, env)
    ax.set_title("")
    ax.text(0.5, 0.98, "%s, Jensen gap %s dB" % (env, gap), transform=ax.transAxes,
            ha="center", va="top", fontsize=5.6)
    ax.set_ylim(-14, 23)
    if legend:
        ax.legend(loc="upper left", fontsize=4.6, handlelength=1.1,
                  bbox_to_anchor=(0.0, 0.92))


def fig5_c(fig, ax):
    _env(fig, ax, "DKITCHEN", "4.8", False)


def fig5_d(fig, ax):
    _env(fig, ax, "PCAFETER", "3.3", True)


FN.update({"fig1_b": fig1_b, "fig1_d": fig1_d, "fig2_a": fig2_a,
           "fig5_c": fig5_c, "fig5_d": fig5_d})


def label(ax, letter):
    # fontsize is divided by K because Axes.text is patched to multiply by K
    ax.text(-0.24, 1.0, "(%s)" % letter, transform=ax.transAxes,
            fontsize=8.5 / K, fontweight="bold", va="bottom", ha="left")


def build_row(name):
    panels, ratios = ROWS[name]
    fig = plt.figure(figsize=(W2, H))
    gs = GridSpec(1, len(panels), figure=fig, width_ratios=ratios, wspace=0.03)
    for k, pname in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        FN[pname](fig, ax)
        label(ax, "abc"[k])
    savefig(fig, name, OUT)
    plt.close(fig)


def build_single(pname):
    w = (72 if pname in COLORBAR else 61) * MM
    fig = plt.figure(figsize=(w, H))
    ax = fig.add_subplot(111)
    FN[pname](fig, ax)
    savefig(fig, pname, OUT1)
    plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    rows = [a for a in args if a in ROWS]
    singles = [a for a in args if a in FN]
    if not args:
        rows, singles = list(ROWS), list(FN)
    elif "singles" in args:
        singles = list(FN)
    for n in rows:
        build_row(n)
    for p in singles:
        build_single(p)
    print("rows3_r1 done")
