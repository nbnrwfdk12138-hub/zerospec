"""Panels of Fig. 3: strong in-band signals over synthetic colored noise."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from zerospec import figstyle
from zerospec.figstyle import OI, W2, MM, panel_label, savefig

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
PAN = os.path.join(FIG, "panels")
figstyle.set_style()
DB = 10 / np.log(10)

ex = np.load(os.path.join(RES, "e3_example.npz"))
cv = np.load(os.path.join(RES, "e3_curves.npz"))

STY = dict(zeros=(OI["vermilion"], "o", "zeros (masked)"),
           zeros_nomask=(OI["sky"], "v", "zeros (no mask)"),
           welch=(OI["orange"], "s", "Welch (mean)"),
           welch_med=(OI["yellow"], "P", "Welch (median)"),
           minstat=(OI["green"], "D", "min-stat (oracle cal.)"),
           mmse=(OI["purple"], "^", "MMSE-SPP"))


def panel_a(fig, ax):
    img = ex["logP2"]
    tg, fg = ex["tg"], ex["fg"]
    vmax = np.quantile(img, 0.997)
    im = ax.imshow(img, origin="lower", aspect="auto",
                   extent=[tg[0], tg[-1], fg[0], fg[-1]],
                   cmap="magma", vmin=vmax - 3.4, vmax=vmax,
                   interpolation="nearest", rasterized=True)
    ax.plot(ex["zt"], ex["zf"], "o", ms=0.8, mfc="white", mec="none", alpha=0.75)
    M = ex["masked"].astype(float)
    ax.contour(ex["mtc"], ex["mfc"], M, levels=[0.5], colors=[OI["sky"]],
               linewidths=0.7)
    ax.set_xlim(10, 500)
    ax.set_ylim(-5.5, 5.5)
    ax.set_xlabel("time $t$ (normalized)")
    ax.set_ylabel("frequency $f$")
    ax.annotate("sweeping chirp", xy=(260, 0.3), xytext=(120, 3.6),
                color="white", fontsize=6,
                arrowprops=dict(arrowstyle="-", color="white", lw=0.7))
    ax.annotate("persistent tone", xy=(300, -4.5), xytext=(330, -2.6),
                color="white", fontsize=6,
                arrowprops=dict(arrowstyle="-", color="white", lw=0.7))
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.045)
    cb.set_label(r"$\log_{10}|V|^2$", labelpad=1)
    cb.outline.set_linewidth(0.5)


def panel_b(fig, ax):
    fr = ex["fref"]
    fb = ex["fexp"][ex["bridged"]]
    if fb.size:
        ax.axvspan(fb.min() - 0.06, fb.max() + 0.06, color=OI["grey"],
                   alpha=0.18, lw=0)
    ax.plot(fr, ex["truth"] * DB, color="black", lw=1.3, label="truth")
    ax.plot(fr, ex["zeros"] * DB, color=OI["vermilion"], lw=1.0,
            label="zeros (masked)")
    ax.plot(fr, ex["welch"] * DB, color=OI["orange"], lw=0.8, ls="--",
            label="Welch (mean)")
    ax.plot(fr, ex["welch_med"] * DB, color=OI["yellow"], lw=0.8, ls="--",
            label="Welch (median)")
    ax.plot(fr, ex["minstat"] * DB, color=OI["green"], lw=0.8, ls="-.",
            label="min-stat")
    ax.plot(fr, ex["mmse"] * DB, color=OI["purple"], lw=0.7, ls=":",
            label="MMSE-SPP")
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel("smoothed PSD (dB)")
    ax.set_xlim(fr[0], fr[-1])
    tmin, tmax = (ex["truth"] * DB).min(), (ex["truth"] * DB).max()
    ax.set_ylim(tmin - 5, tmax + 11)
    ax.annotate(r"tone peaks $\uparrow$", xy=(-4.5, tmax + 10.4), fontsize=5.4,
                ha="center", color=OI["grey"])
    ax.legend(loc="lower right", fontsize=5.0, ncols=2)
    ax.text(0.02, 0.04, "shaded: bridged under tone", transform=ax.transAxes,
            fontsize=5.6, color=OI["grey"])


def _bias_panel(ax, key):
    g = np.array(cv["gammas"], float)
    for m, (c, mk, lab) in STY.items():
        med = cv[f"{key}_{m}"][:, 0] * DB
        lo = cv[f"{key}_{m}"][:, 1] * DB
        hi = cv[f"{key}_{m}"][:, 2] * DB
        ax.errorbar(g, med, yerr=[med - lo, hi - med], marker=mk, ms=3,
                    lw=1.0, capsize=1.5, color=c, label=lab)
    ax.axhline(0, color=OI["grey"], lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"per-cell signal SNR $\gamma_0$")


def panel_c(fig, ax):
    _bias_panel(ax, "g_full")
    ax.set_ylabel("full-band bias (dB)")
    ax.legend(loc="upper left", fontsize=5.0)


def panel_d(fig, ax):
    _bias_panel(ax, "g_tone")
    ax.set_ylabel("bias at persistent-tone rows (dB)")


PANELS = [("a", panel_a, (89 * MM, 55 * MM)),
          ("b", panel_b, (89 * MM, 62 * MM)),
          ("c", panel_c, (89 * MM, 62 * MM)),
          ("d", panel_d, (89 * MM, 62 * MM))]


def compose():
    fig = plt.figure(figsize=(W2, 0.62 * W2))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.15, 1])
    for (letter, fn, _), pos in zip(PANELS,
                                    [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]):
        ax = fig.add_subplot(pos)
        fn(fig, ax)
        panel_label(ax, letter)
    savefig(fig, "fig3_robustness", FIG)
    plt.close(fig)


def export_panels():
    for letter, fn, size in PANELS:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        fn(fig, ax)
        savefig(fig, f"fig3_{letter}", PAN)
        plt.close(fig)


if __name__ == "__main__":
    compose()
    export_panels()
    print("fig3 done")
