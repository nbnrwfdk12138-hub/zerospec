"""Panels of Fig. 4: on-source noise estimation on LIGO strain (GW150914, H1)."""
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

lg = np.load(os.path.join(RES, "e3b_ligo.npz"))
snap = np.load(os.path.join(RES, "e3b_event_snapshot.npz"))
DB = 10 / np.log(10)

ASD = lambda logP: np.sqrt(2.0 * np.exp(logP))
ASD_P = lambda P: np.sqrt(2.0 * P)


def panel_a(fig, ax):
    img = snap["logP2"]
    img = img - np.median(img, axis=1, keepdims=True)
    t_s, f_hz = snap["t_s"], snap["f_hz"]
    im = ax.imshow(img, origin="lower", aspect="auto",
                   extent=[t_s[0] - snap["t_evt"], t_s[-1] - snap["t_evt"],
                           f_hz[0], f_hz[-1]],
                   cmap="magma", vmin=-0.9, vmax=1.4,
                   interpolation="nearest", rasterized=True)
    ax.plot(snap["zt_s"] - snap["t_evt"], snap["zf_hz"], "o", ms=1.6,
            mfc="white", mec="none", alpha=0.95)
    ax.set_xlim(-1.6, 0.55)
    ax.set_ylim(64, 360)
    ax.set_xlabel("time from GW150914 (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.annotate("chirp\n(zeros expelled)", xy=(-0.07, 170), xytext=(-0.95, 270),
                color="white", fontsize=6.0,
                arrowprops=dict(arrowstyle="-", color="white", lw=0.7))
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.045)
    cb.set_label(r"row-whitened $\log_{10}|V|^2$", labelpad=1)
    cb.outline.set_linewidth(0.5)


def panel_b(fig, ax):
    f = lg["fline"]
    for f0 in lg["diag_fest"][lg["diag_bridged"]]:
        ax.axvspan(f0 - 0.25, f0 + 0.25, color=OI["grey"], alpha=0.18, lw=0)
    ax.plot(f, ASD_P(lg["Pref"]), color="black", lw=1.2,
            label="off-source Welch (ref.)")
    ax.plot(f, ASD(lg["curve_g0_minstat"]), color=OI["green"], lw=0.8,
            alpha=0.9, label="min-stat (on-source)")
    ax.plot(f, ASD(lg["curve_g0_zeros"]), color=OI["vermilion"], lw=1.0,
            label="zeros (on-source)")
    ax.set_yscale("log")
    ax.set_xlim(64, 512)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"ASD (strain$/\sqrt{\mathrm{Hz}}$)")
    ax.legend(loc="upper right", fontsize=5.8)
    ax.text(0.02, 0.93, "shaded: lines, declared bridged",
            transform=ax.transAxes, fontsize=5.6, color=OI["grey"])


def panel_c(fig, ax):
    f = lg["fline"]
    sw = lg["swept"]
    ax.axvspan(sw[0], sw[1], color=OI["sky"], alpha=0.12, lw=0)
    ax.plot(f, ASD_P(lg["Pref"]), color="black", lw=1.2, label="reference")
    ax.plot(f, ASD(lg["curve_g300_welch_on"]), color=OI["orange"], lw=0.9,
            ls="--", label="Welch on-source (mean)")
    ax.plot(f, ASD(lg["curve_g300_zeros"]), color=OI["vermilion"], lw=1.0,
            label="zeros (on-source)")
    ax.set_yscale("log")
    ax.set_xlim(64, 512)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"ASD (strain$/\sqrt{\mathrm{Hz}}$)")
    ax.legend(loc="upper right", fontsize=5.8)
    ax.text(0.03, 0.05, r"injected chirp, per-cell SNR $\gamma=300$"
            "\n(swept band shaded)", transform=ax.transAxes, fontsize=5.6)


def panel_d(fig, ax):
    g = np.array(lg["gammas"], float)
    gplot = np.where(g == 0, 3.0, g)
    sty = dict(zeros=(OI["vermilion"], "o", "zeros"),
               welch_on=(OI["orange"], "s", "Welch on-source (mean)"),
               welch_med=(OI["yellow"], "P", "Welch on-source (median)"),
               minstat=(OI["green"], "D", "min-stat (off-source cal.)"),
               mmse=(OI["purple"], "^", "MMSE-SPP"))
    for m, (c, mk, lab) in sty.items():
        ax.plot(gplot, lg[f"bias_{m}"] * DB, marker=mk, ms=3, lw=1.0, color=c,
                label=lab)
    ax.axhline(0, color=OI["grey"], lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([3, 10, 30, 100, 300])
    ax.set_xticklabels(["0", "10", "30", "100", "300"])
    ax.set_xlabel(r"injected per-cell SNR $\gamma$")
    ax.set_ylabel("swept-band bias (dB)")
    ax.legend(loc="upper left", fontsize=5.4)


PANELS = [("a", panel_a, (89 * MM, 58 * MM)),
          ("b", panel_b, (89 * MM, 62 * MM)),
          ("c", panel_c, (89 * MM, 62 * MM)),
          ("d", panel_d, (89 * MM, 62 * MM))]


def compose():
    fig = plt.figure(figsize=(W2, 0.66 * W2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    for (letter, fn, _), pos in zip(PANELS,
                                    [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]):
        ax = fig.add_subplot(pos)
        fn(fig, ax)
        panel_label(ax, letter)
    savefig(fig, "fig4_ligo_onsource", FIG)
    plt.close(fig)


def export_panels():
    for letter, fn, size in PANELS:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        fn(fig, ax)
        savefig(fig, f"fig4_{letter}", PAN)
        plt.close(fig)


if __name__ == "__main__":
    compose()
    export_panels()
    print("fig4 done")
