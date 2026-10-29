"""Compact single panels of Figs. 1-5 (6-pt base font, about 44 mm wide).

Panels whose legends or annotations do not fit at this size have compact
variants here.  In the modulated-noise panel the time axis is shown modulo
the modulation period, with sigma^2(t) = 1 + 0.8 sin(2 pi t / 4) in absolute
time.  make_rows3_r1.py assembles the final figures from these panels.

Outputs: figures/panels_r1/figN_x.{pdf,png}
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from zerospec import figstyle
from zerospec.figstyle import OI, MM, savefig
from zerospec.theory import rho_modulated

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "figures")
OUT = os.path.join(FIG, "panels_r1")
os.makedirs(OUT, exist_ok=True)

figstyle.set_style()
mpl.rcParams.update({
    "font.size": 6.0, "axes.labelsize": 6.0, "axes.titlesize": 6.2,
    "xtick.labelsize": 5.4, "ytick.labelsize": 5.4, "legend.fontsize": 4.9,
    "lines.linewidth": 0.85, "lines.markersize": 2.4,
    "legend.handlelength": 1.2, "legend.labelspacing": 0.25,
    "legend.borderaxespad": 0.2, "legend.columnspacing": 0.8,
    "legend.handletextpad": 0.5,
    "axes.labelpad": 1.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
})

import make_fig1 as f1
import make_fig2 as f2
import make_fig3 as f3
import make_fig4 as f4
import make_fig5 as f5

DB = 10 / np.log(10)


# Fig. 1
def fig1_panel_c(fig, ax):
    """Fig. 1(c): time modulo the period; theory on a fresh grid; no legend
    (symbols as in panel b)."""
    mod = f1.mod
    T0 = float(mod["T0"])
    sig2 = lambda t: 1.0 + 0.8 * np.sin(2 * np.pi * t / T0)
    tl = np.mod(mod["tloc"], T0)
    o = np.argsort(tl)
    tt = np.linspace(-0.5, T0 + 0.5, 500)
    th = rho_modulated(sig2, tt)
    keep = (tt >= 0) & (tt <= T0)
    ax.axhline(1.0, color=OI["grey"], lw=0.6, ls=":", zorder=1)
    ax.plot(tt[keep], th[keep], color=OI["blue"], lw=1.0, zorder=3)
    ax.errorbar(tl[o], mod["emp"][o], yerr=2 * mod["se"][o], fmt="o",
                ms=2.2, lw=0.7, color=OI["vermilion"], mfc="white", mew=0.7,
                zorder=4)
    ax2 = ax.twinx()
    ax2.plot(tt[keep], sig2(tt[keep]), color=OI["green"], lw=0.8, ls="--",
             alpha=0.8)
    ax2.set_ylabel(r"$\sigma^2(t)$", color=OI["green"])
    ax2.tick_params(axis="y", colors=OI["green"], labelsize=5.4)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(OI["green"])
    ax.set_xlabel(r"time $t$ mod 4")
    ax.set_ylabel(r"zero density $\rho$")
    ax.set_xlim(0, T0)


def fig1_panel_d(fig, ax):
    non = f1.non
    keep = np.abs(non["fc"]) < 1.8
    ax.axhline(1.0, color=OI["grey"], lw=0.6, ls=":", zorder=1)
    ax.plot(non["fine"], non["th_fine"], color=OI["blue"], lw=1.0,
            label=r"$e^{-\gamma}(1+4\pi(f\!-\!f_0)^2\gamma)$")
    ax.plot(non["fine"], non["supp_fine"], color=OI["grey"], lw=0.9, ls="--",
            label=r"$e^{-\gamma}$ alone")
    emp = np.maximum(non["emp"][keep], 1.2e-3)
    lo = np.maximum(non["emp"][keep] - 2 * non["se"][keep], 1.2e-3)
    ax.errorbar(non["fc"][keep], emp, yerr=[emp - lo, 2 * non["se"][keep]],
                fmt="o", ms=2.2, lw=0.7, color=OI["vermilion"], mfc="white",
                mew=0.7, label="Monte Carlo")
    ax.set_yscale("log")
    ax.set_ylim(1.2e-3, 5)
    ax.set_xlim(-1.8, 1.8)
    ax.set_xlabel(r"$f-f_0$")
    ax.set_ylabel(r"zero density $\rho$")
    ax.annotate("ring", xy=(0.82, 2.6), xytext=(1.2, 1.15),
                arrowprops=dict(arrowstyle="-", lw=0.6), fontsize=5.2)
    ax.annotate("$e^{-\\gamma_{\\max}}$", xy=(0.05, 4.5e-3),
                xytext=(0.42, 1.8e-2),
                arrowprops=dict(arrowstyle="-", lw=0.6), fontsize=5.2)
    ax.legend(loc="lower left", fontsize=4.6, handlelength=1.0)


# Fig. 2
def fig2_panel_a(fig, ax):
    ex = f2.ex
    f = ex["f"]
    ax.fill_between(f, ex["zero_lo"] * DB, ex["zero_hi"] * DB,
                    color=OI["vermilion"], alpha=0.18, lw=0,
                    label="zeros, 10–90%")
    ax.plot(f, ex["truth"] * DB, color="black", lw=1.2, label="truth")
    ax.plot(f, ex["zero"] * DB, color=OI["vermilion"], lw=0.9,
            label="zeros (1 record)")
    ax.plot(f, ex["welch"] * DB, color=OI["orange"], lw=0.8, ls="--",
            label="Welch")
    ax.plot(f, ex["minstat"] * DB, color=OI["green"], lw=0.8, ls="-.",
            label="min-stat (oracle)")
    ax.plot(f, ex["mmse"] * DB, color=OI["purple"], lw=0.7, ls=":",
            label="MMSE-SPP")
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel("smoothed PSD (dB)")
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(-10.5, 3.5)
    ax.legend(loc="lower center", ncols=2, fontsize=4.4, handlelength=1.0,
              columnspacing=0.6, borderaxespad=0.1)
    lpp = np.gradient(np.gradient(ex["truth"], f), f)
    ipk = np.argmax(ex["truth"])
    b = 0.7 * 500.0 ** (-1 / 5)
    bias_pk = 0.5 * b ** 2 * lpp[ipk] * DB
    ax.annotate(rf"kernel bias ${bias_pk:+.1f}$ dB",
                xy=(f[ipk], ex["zero_hi"][ipk] * DB), xytext=(0.55, 2.6),
                fontsize=5.0, arrowprops=dict(arrowstyle="-", lw=0.6))


def fig2_panel_b(fig, ax):
    rt = f2.rt
    T, med, lo, hi = rt["T"], rt["med"], rt["lo"], rt["hi"]
    ax.fill_between(T, np.array(lo) * DB, np.array(hi) * DB, color=OI["blue"],
                    alpha=0.18, lw=0)
    ax.plot(T, np.array(med) * DB, "o-", color=OI["blue"], ms=2.6,
            label="sup error (median, IQR)")
    guide = med[2] * (T / T[2]) ** (-2 / 5)
    ax.plot(T, guide * DB, ls="--", color=OI["grey"], lw=0.9,
            label=r"$T^{-2/5}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("record length $T$")
    ax.set_ylabel(r"$\sup_f|\hat\Lambda-\Lambda|$ (dB)")
    ax.legend(loc="upper right", fontsize=4.6, handlelength=1.1)
    ax.text(0.04, 0.06, f"slope {float(rt['slope']):.2f}",
            transform=ax.transAxes, fontsize=5.4)


# Fig. 3
def fig3_panel_a(fig, ax):
    ex = f3.ex
    img = ex["logP2"]
    tg, fg = ex["tg"], ex["fg"]
    vmax = np.quantile(img, 0.997)
    im = ax.imshow(img, origin="lower", aspect="auto",
                   extent=[tg[0], tg[-1], fg[0], fg[-1]],
                   cmap="magma", vmin=vmax - 3.4, vmax=vmax,
                   interpolation="nearest", rasterized=True)
    ax.plot(ex["zt"], ex["zf"], "o", ms=0.6, mfc="white", mec="none", alpha=0.7)
    M = ex["masked"].astype(float)
    ax.contour(ex["mtc"], ex["mfc"], M, levels=[0.5], colors=[OI["sky"]],
               linewidths=0.6)
    ax.set_xlim(10, 500)
    ax.set_ylim(-5.5, 5.5)
    ax.set_xlabel("time $t$ (normalized)")
    ax.set_ylabel("frequency $f$")
    ax.annotate("chirp", xy=(260, 0.3), xytext=(110, 3.4),
                color="white", fontsize=5.4,
                arrowprops=dict(arrowstyle="-", color="white", lw=0.6))
    ax.annotate("tone", xy=(300, -4.5), xytext=(330, -2.4),
                color="white", fontsize=5.4,
                arrowprops=dict(arrowstyle="-", color="white", lw=0.6))
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.045)
    cb.set_label(r"$\log_{10}|V|^2$", labelpad=1)
    cb.outline.set_linewidth(0.5)


def fig3_panel_b(fig, ax):
    ex = f3.ex
    fr = ex["fref"]
    fb = ex["fexp"][ex["bridged"]]
    if fb.size:
        ax.axvspan(fb.min() - 0.06, fb.max() + 0.06, color=OI["grey"],
                   alpha=0.18, lw=0)
    ax.plot(fr, ex["truth"] * DB, color="black", lw=1.2, label="truth")
    ax.plot(fr, ex["zeros"] * DB, color=OI["vermilion"], lw=0.9,
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
    ax.set_ylim(tmin - 7, tmax + 11)
    ax.annotate(r"tone peaks $\uparrow$", xy=(-4.5, tmax + 10.2), fontsize=5.0,
                ha="center", color=OI["grey"])
    ax.legend(loc="lower right", fontsize=4.4, ncols=2, handlelength=1.0,
              columnspacing=0.6, borderaxespad=0.1)


def fig3_panel_d(fig, ax):
    f3._bias_panel(ax, "g_tone")
    ax.set_ylabel("bias at tone rows (dB)")


# Fig. 4
def fig4_panel_b(fig, ax):
    lg = f4.lg
    f = lg["fline"]
    # shading: rows bridged by the mask in this (no-injection) run, widened by
    # +-2.5 Hz; exactly the rows excluded from the metrics (line_excl)
    ex = lg["line_excl"].astype(bool)
    edges = np.flatnonzero(np.diff(np.r_[0, ex.astype(int), 0]))
    for i0, i1 in zip(edges[::2], edges[1::2]):
        ax.axvspan(f[i0] - 0.5, f[i1 - 1] + 0.5, color=OI["grey"], alpha=0.22,
                   lw=0)
    ax.plot(f, f4.ASD_P(lg["Pref"]), color="black", lw=1.1,
            label="off-source Welch (ref.)")
    ax.plot(f, f4.ASD(lg["curve_g0_minstat"]), color=OI["green"], lw=0.75,
            alpha=0.9, label="min-stat (on-source)")
    ax.plot(f, f4.ASD(lg["curve_g0_zeros"]), color=OI["vermilion"], lw=0.95,
            label="zeros (on-source)")
    ax.set_yscale("log")
    ax.set_xlim(64, 512)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"ASD (strain$/\sqrt{\mathrm{Hz}}$)")
    ax.legend(loc="upper left", fontsize=4.6, handlelength=1.1)


def fig4_panel_c(fig, ax):
    lg = f4.lg
    f = lg["fline"]
    sw = lg["swept"]
    ax.axvspan(sw[0], sw[1], color=OI["sky"], alpha=0.12, lw=0)
    ax.plot(f, f4.ASD_P(lg["Pref"]), color="black", lw=1.1, label="reference")
    ax.plot(f, f4.ASD(lg["curve_g300_welch_on"]), color=OI["orange"], lw=0.85,
            ls="--", label="Welch (mean)")
    ax.plot(f, f4.ASD(lg["curve_g300_zeros"]), color=OI["vermilion"], lw=0.95,
            label="zeros")
    ax.set_yscale("log")
    ax.set_xlim(64, 512)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"ASD (strain$/\sqrt{\mathrm{Hz}}$)")
    ax.legend(loc="upper left", fontsize=4.6, handlelength=1.1,
              bbox_to_anchor=(0.0, 0.88))
    ax.text(0.98, 0.95, r"injected chirp, $\gamma=300$ (shaded)",
            transform=ax.transAxes, fontsize=5.0, ha="right", va="top")


# Fig. 5
def fig5_panel_a(fig, ax):
    ex = f5.ex
    img = ex["logP2"]
    vmax = np.quantile(img, 0.998)
    im = ax.imshow(img, origin="lower", aspect="auto",
                   extent=[ex["t_s"][0], ex["t_s"][-1],
                           ex["f_hz"][0] / 1000, ex["f_hz"][-1] / 1000],
                   cmap="magma", vmin=vmax - 3.6, vmax=vmax,
                   interpolation="nearest", rasterized=True)
    ax.plot(ex["zt_s"], ex["zf_hz"] / 1000, "o", ms=0.5, mfc="white",
            mec="none", alpha=0.6)
    ax.set_xlim(0.3, 7.7)
    ax.set_ylim(0.125, 7.0)
    ax.set_xticks([2, 4, 6])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (kHz)")
    ax.text(0.02, 0.94, "speech + babble, SNR +5 dB",
            transform=ax.transAxes, fontsize=5.2, color="white")
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.045)
    cb.set_label(r"$\log_{10}|V|^2$", labelpad=1)
    cb.outline.set_linewidth(0.5)


def fig5_panel_c(fig, ax):
    f5._env_panel(ax, "DKITCHEN")
    ax.set_title("")
    ax.text(0.5, 0.98, "DKITCHEN, Jensen gap 4.8 dB", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.6)
    ax.legend(loc="upper left", fontsize=4.6, handlelength=1.1,
              bbox_to_anchor=(0.0, 0.92))


def fig5_panel_d(fig, ax):
    f5._env_panel(ax, "PCAFETER")
    ax.set_title("")
    ax.text(0.5, 0.98, "PCAFETER, Jensen gap 3.3 dB", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.6)


# panel registry: (output name, function, size in mm)
W = 44 * MM
H = 40 * MM
WS = 52 * MM          # spectrogram panels carry a colorbar
PANELS = [
    ("fig1_a", f1.panel_a, (WS, H)),
    ("fig1_b", f1.panel_b, (W, H)),
    ("fig1_c", fig1_panel_c, (W, H)),
    ("fig1_d", fig1_panel_d, (W, H)),
    ("fig2_a", fig2_panel_a, (W, H)),
    ("fig2_b", fig2_panel_b, (W, H)),
    ("fig3_a", fig3_panel_a, (WS, H)),
    ("fig3_b", fig3_panel_b, (W, H)),
    ("fig3_c", f3.panel_c, (W, H)),
    ("fig3_d", fig3_panel_d, (W, H)),
    ("fig4_a", f4.panel_a, (WS, H)),
    ("fig4_b", fig4_panel_b, (W, H)),
    ("fig4_c", fig4_panel_c, (W, H)),
    ("fig4_d", f4.panel_d, (W, H)),
    ("fig5_a", fig5_panel_a, (WS, H)),
    ("fig5_b", f5.panel_b, (W, H)),
    ("fig5_c", fig5_panel_c, (W, H)),
    ("fig5_d", fig5_panel_d, (W, H)),
]


def render(only=None):
    for name, fn, size in PANELS:
        if only and name not in only:
            continue
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        fn(fig, ax)
        savefig(fig, name, OUT)
        plt.close(fig)


if __name__ == "__main__":
    import sys
    render(set(sys.argv[1:]) or None)
    print("panels_r1 done")
