"""Panels of Fig. 1 (intensity laws against Monte Carlo) and of Fig. 2(a).

Writes a composite figure (figures/fig1_intensity_theory.*) and single panels
(figures/panels/fig1_{a,b,c,d}.*); make_rows3_r1.py assembles the final figures.
"""
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

snap = np.load(os.path.join(RES, "e1_snapshot.npz"))
col = np.load(os.path.join(RES, "e1_colored.npz"))
mod = np.load(os.path.join(RES, "e1_modulated.npz"))
non = np.load(os.path.join(RES, "e1_noncentral.npz"))


def panel_a(fig, ax):
    img = snap["logP2"]
    tg, fg = snap["tgrid"], snap["fgrid"]
    vmax = np.quantile(img, 0.995)
    im = ax.imshow(img, origin="lower", aspect="auto",
                   extent=[tg[0], tg[-1], fg[0], fg[-1]],
                   cmap="magma", vmin=vmax - 3.2, vmax=vmax,
                   interpolation="nearest", rasterized=True)
    ax.plot(snap["zt"], snap["zf"], "o", ms=1.7, mfc="white", mec="none",
            alpha=0.95)
    ax.set_xlim(4, 44)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("time $t$ (normalized)")
    ax.set_ylabel("frequency $f$ (normalized)")
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.045)
    cb.set_label(r"$\log_{10}|V|^2$", labelpad=1)
    cb.outline.set_linewidth(0.5)


def panel_b(fig, ax):
    keep = (col["fc"] > -2.2) & (col["fc"] < 2.2)
    ax.axhline(1.0, color=OI["grey"], lw=0.6, ls=":", zorder=1)
    ax.fill_between(col["fine"], col["th_fine"], 1.0, color=OI["blue"],
                    alpha=0.12, lw=0)
    ax.plot(col["fine"], col["th_fine"], color=OI["blue"], lw=1.1,
            label=r"theory $1+\frac{1}{4\pi}(\log P)''$")
    ax.errorbar(col["fc"][keep], col["emp"][keep], yerr=2 * col["se"][keep],
                fmt="o", ms=2.4, lw=0.7, color=OI["vermilion"], mfc="white",
                mew=0.7, capsize=0, label="Monte Carlo (±2 s.e.)")
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel(r"zero density $\rho$")
    ax.set_xlim(-2.3, 2.3)
    ax.legend(loc="upper right", borderaxespad=0.2)
    z = (col["emp"][keep] - col["th_bins"][keep]) / col["se"][keep]
    ax.text(0.02, 0.04, rf"rms $z$ = {np.sqrt(np.mean(z**2)):.2f}",
            transform=ax.transAxes, fontsize=6)


def panel_c(fig, ax):
    t0 = mod["tloc"][0]
    ax.axhline(1.0, color=OI["grey"], lw=0.6, ls=":", zorder=1)
    ax.plot(mod["fine"] - t0, mod["th_fine"], color=OI["blue"], lw=1.1,
            label=r"theory $1+\frac{1}{4\pi}\partial_t^2\log(\sigma^2\!*\!|g|^2)$")
    ax.errorbar(mod["tloc"] - t0, mod["emp"], yerr=2 * mod["se"], fmt="o",
                ms=2.4, lw=0.7, color=OI["vermilion"], mfc="white", mew=0.7,
                label="Monte Carlo (±2 s.e.)")
    sig2 = 1 + 0.8 * np.sin(2 * np.pi * (mod["fine"]) / mod["T0"])
    ax2 = ax.twinx()
    ax2.plot(mod["fine"] - t0, sig2, color=OI["green"], lw=0.8, ls="--",
             alpha=0.8)
    ax2.set_ylabel(r"$\sigma^2(t)$", color=OI["green"])
    ax2.tick_params(axis="y", colors=OI["green"], labelsize=6)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(OI["green"])
    ax.set_xlabel(r"time within modulation period")
    ax.set_ylabel(r"zero density $\rho$")
    ax.set_xlim(0, mod["T0"])
    ax.legend(loc="upper left", borderaxespad=0.2)


def panel_d(fig, ax):
    keep = np.abs(non["fc"]) < 1.8
    ax.axhline(1.0, color=OI["grey"], lw=0.6, ls=":", zorder=1)
    ax.plot(non["fine"], non["th_fine"], color=OI["blue"], lw=1.1,
            label=r"$e^{-\gamma}\,(1+4\pi(f\!-\!f_0)^2\gamma)$")
    ax.plot(non["fine"], non["supp_fine"], color=OI["grey"], lw=0.9, ls="--",
            label=r"$e^{-\gamma}$ alone")
    lo = np.maximum(non["emp"][keep] - 2 * non["se"][keep], 1.2e-3)
    ax.errorbar(non["fc"][keep], np.maximum(non["emp"][keep], 1.2e-3),
                yerr=[np.maximum(non["emp"][keep], 1.2e-3) - lo,
                      2 * non["se"][keep]],
                fmt="o", ms=2.4, lw=0.7, color=OI["vermilion"], mfc="white",
                mew=0.7, label="Monte Carlo (±2 s.e.)")
    ax.set_yscale("log")
    ax.set_ylim(1.2e-3, 5)
    ax.set_xlim(-1.8, 1.8)
    ax.set_xlabel(r"$f-f_0$")
    ax.set_ylabel(r"zero density $\rho$")
    ax.annotate("ring", xy=(0.82, 2.6), xytext=(1.25, 1.1),
                arrowprops=dict(arrowstyle="-", lw=0.6), fontsize=6)
    ax.annotate("suppression\n$e^{-\\gamma_0}$", xy=(0.05, 4.5e-3),
                xytext=(0.45, 1.5e-2),
                arrowprops=dict(arrowstyle="-", lw=0.6), fontsize=6)
    ax.legend(loc="lower right", borderaxespad=0.1, fontsize=5.6)


PANELS = [("a", panel_a, (89 * MM, 55 * MM)),
          ("b", panel_b, (89 * MM, 62 * MM)),
          ("c", panel_c, (89 * MM, 62 * MM)),
          ("d", panel_d, (89 * MM, 62 * MM))]


def compose():
    fig = plt.figure(figsize=(W2, 0.62 * W2))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.15, 1])
    for (letter, fn, _), pos in zip(PANELS, [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]):
        ax = fig.add_subplot(pos)
        fn(fig, ax)
        panel_label(ax, letter)
    savefig(fig, "fig1_intensity_theory", FIG)
    plt.close(fig)


def export_panels():
    for letter, fn, size in PANELS:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        fn(fig, ax)
        savefig(fig, f"fig1_{letter}", PAN)
        plt.close(fig)


if __name__ == "__main__":
    compose()
    export_panels()
    print("fig1 done")
