"""Panels of SM Fig. 8: variable-speed bearing vibration (Ottawa data set)."""
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

exH = np.load(os.path.join(RES, "e3d_example_H-A-1.npz"))
bH = np.load(os.path.join(RES, "e3d_bearing_H-A-1.npz"))
bI = np.load(os.path.join(RES, "e3d_bearing_I-A-1.npz"))


def panel_a(fig, ax):
    img = exH["logP2"]
    t_s, f_hz = exH["t_s"], exH["f_hz"]
    vmax = np.quantile(img, 0.998)
    im = ax.imshow(img, origin="lower", aspect="auto",
                   extent=[t_s[0], t_s[-1], f_hz[0] / 1000, f_hz[-1] / 1000],
                   cmap="magma", vmin=vmax - 3.5, vmax=vmax,
                   interpolation="nearest", rasterized=True)
    ax.plot(exH["zt"], exH["zf"] / 1000, "o", ms=0.6, mfc="white", mec="none",
            alpha=0.55)
    tt = np.linspace(t_s[0], t_s[-1], 200)
    fsh = np.interp(tt, exH["fsh_t"], exH["fsh_f"])
    for k in (25, 50, 75, 100):
        ax.plot(tt, k * fsh / 1000, color=OI["sky"], lw=0.5, ls="--", alpha=0.8)
    ax.set_ylim(0.5, 2.5)
    ax.set_xlim(t_s[0], t_s[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (kHz)")
    f_lo, f_hi = np.interp([t_s[0], t_s[-1]], exH["fsh_t"], exH["fsh_f"])
    ax.text(0.02, 0.97, "healthy, shaft %.0f$\\to$%.0f Hz;\ndashed: orders 25, 50, 75, 100"
            % (f_lo, f_hi), transform=ax.transAxes, fontsize=5.4, color="white",
            va="top")
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.05)
    cb.set_label(r"$\log_{10}|V|^2$", labelpad=1)
    cb.outline.set_linewidth(0.5)


def panel_b(fig, ax):
    f = bH["fb"] / 1000
    ax.plot(f, bH["lref_a"] * DB, color="black", lw=1.1,
            label="tacho-notched ref.")
    ax.plot(f, bH["zeros"] * DB, color=OI["vermilion"], lw=0.9, label="zeros")
    ax.plot(f, bH["welch"] * DB, color=OI["orange"], lw=0.7, ls="--",
            label="Welch")
    ax.plot(f, bH["ms_white"] * DB, color=OI["green"], lw=0.7, ls="-.",
            label="min-stat (generic cal.)")
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("floor (dB)")
    ax.set_title("healthy (H-A-1)", fontsize=6.5)
    ax.legend(fontsize=5.0, loc="upper right")


def panel_c(fig, ax):
    f = bI["fb"] / 1000
    ax.plot(f, bI["lref_a"] * DB, color="black", lw=1.1,
            label="trimmed mean (burst-incl.)")
    ax.plot(f, bI["lref_m"] * DB, color=OI["grey"], lw=0.9, ls="--",
            label="median floor")
    ax.plot(f, bI["zeros"] * DB, color=OI["vermilion"], lw=0.9, label="zeros")
    ax.plot(f, bI["welch"] * DB, color=OI["orange"], lw=0.7, ls=":",
            label="Welch")
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("floor (dB)")
    ax.set_title("inner-race fault (I-A-1)", fontsize=6.5)
    ax.legend(fontsize=5.0, loc="upper right")


PANELS = [("a", panel_a, (74 * MM, 55 * MM)),
          ("b", panel_b, (58 * MM, 52 * MM)),
          ("c", panel_c, (58 * MM, 52 * MM))]


def compose():
    fig = plt.figure(figsize=(W2, 0.34 * W2))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 1, 1])
    for (letter, fn, _), pos in zip(PANELS, [gs[0, 0], gs[0, 1], gs[0, 2]]):
        ax = fig.add_subplot(pos)
        fn(fig, ax)
        panel_label(ax, letter)
    savefig(fig, "fig7_bearing", FIG)
    plt.close(fig)


def export_panels():
    for letter, fn, size in PANELS:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        fn(fig, ax)
        savefig(fig, f"fig7_{letter}", PAN)
        plt.close(fig)


if __name__ == "__main__":
    compose()
    export_panels()
    print("fig7 done")
