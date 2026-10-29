"""Panels of Fig. 5: noise PSD estimation during continuous speech."""
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

ex = np.load(os.path.join(RES, "e3c_example.npz"))
sw = np.load(os.path.join(RES, "e3c_speech.npz"))

STY = dict(zeros=(OI["vermilion"], "o", "zeros vs geometric"),
           zeros_arith=(OI["sky"], "v", "zeros+corr. vs arith."),
           welch_on=(OI["orange"], "s", "Welch vs arith."),
           minstat=(OI["green"], "D", "min-stat vs arith."),
           mmse=(OI["purple"], "^", "MMSE-SPP vs arith."))


def panel_a(fig, ax):
    img = ex["logP2"]
    vmax = np.quantile(img, 0.998)
    im = ax.imshow(img, origin="lower", aspect="auto",
                   extent=[ex["t_s"][0], ex["t_s"][-1],
                           ex["f_hz"][0] / 1000, ex["f_hz"][-1] / 1000],
                   cmap="magma", vmin=vmax - 3.6, vmax=vmax,
                   interpolation="nearest", rasterized=True)
    ax.plot(ex["zt_s"], ex["zf_hz"] / 1000, "o", ms=0.7, mfc="white",
            mec="none", alpha=0.6)
    ax.set_xlim(0.3, 7.7)
    ax.set_ylim(0.125, 7.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (kHz)")
    ax.text(0.02, 0.93, "LibriSpeech + PCAFETER babble, SNR +5 dB",
            transform=ax.transAxes, fontsize=5.8, color="white")
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.045)
    cb.set_label(r"$\log_{10}|V|^2$", labelpad=1)
    cb.outline.set_linewidth(0.5)


def panel_b(fig, ax):
    f = ex["fline"] / 1000
    ax.plot(f, ex["ltruth_arith"] * DB, color="black", lw=1.2,
            label=r"truth $\log\langle P\rangle_t$ (arith.)")
    ax.plot(f, ex["ltruth_geo"] * DB, color=OI["grey"], lw=1.0, ls="--",
            label=r"truth $\langle\log P\rangle_t$ (geom.)")
    ax.plot(f, ex["zeros"] * DB, color=OI["vermilion"], lw=1.0, label="zeros")
    ax.plot(f, ex["minstat"] * DB, color=OI["green"], lw=0.8, ls="-.",
            label="min-stat (offline cal.)")
    ax.plot(f, ex["welch_on"] * DB, color=OI["orange"], lw=0.8, ls=":",
            label="Welch (on mixture)")
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("noise PSD (dB)")
    ax.set_xlim(f[0], f[-1])
    ax.legend(loc="upper right", fontsize=5.2)
    ax.annotate("Jensen gap", xy=(2.2, ex["ltruth_geo"][515] * DB + 1),
                xytext=(3.4, ex["ltruth_arith"][515] * DB + 6), fontsize=5.6,
                arrowprops=dict(arrowstyle="-", lw=0.6))


def _env_panel(ax, env):
    snrs = np.array(sw["snrs"], float)
    for m, (c, mk, lab) in STY.items():
        med = [np.median(sw[f"bias_{env}_{int(s)}_{m}"]) * DB for s in snrs]
        ax.plot(snrs, med, marker=mk, ms=3, lw=1.0, color=c, label=lab)
    ax.axhline(0, color=OI["grey"], lw=0.6, ls=":")
    gap = float(sw[f"gap_{env}"]) * DB
    ax.set_title(f"{env}  (Jensen gap {gap:.1f} dB)", fontsize=6.5)
    ax.set_xlabel("mixture SNR (dB)")
    ax.set_ylabel("bias vs stated estimand (dB)")
    ax.set_ylim(-14, 18)


def panel_c(fig, ax):
    _env_panel(ax, "DKITCHEN")
    ax.legend(loc="upper left", fontsize=5.2)


def panel_d(fig, ax):
    _env_panel(ax, "PCAFETER")


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
    savefig(fig, "fig5_speech", FIG)
    plt.close(fig)


def export_panels():
    for letter, fn, size in PANELS:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        fn(fig, ax)
        savefig(fig, f"fig5_{letter}", PAN)
        plt.close(fig)


if __name__ == "__main__":
    compose()
    export_panels()
    print("fig5 done")
