"""Panels of Fig. 2: estimator accuracy on clean colored noise and convergence rate."""
import os
import ast
import numpy as np
import matplotlib.pyplot as plt
from zerospec import figstyle
from zerospec.figstyle import OI, W2, MM, panel_label, savefig

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
PAN = os.path.join(FIG, "panels")
figstyle.set_style()
DB = 10 / np.log(10)

ex = np.load(os.path.join(RES, "e2_example.npz"))
rt = np.load(os.path.join(RES, "e2_rate.npz"))


def panel_a(fig, ax):
    f = ex["f"]
    ax.fill_between(f, ex["zero_lo"] * DB, ex["zero_hi"] * DB,
                    color=OI["vermilion"], alpha=0.18, lw=0,
                    label="zeros, 10–90% of runs")
    ax.plot(f, ex["truth"] * DB, color="black", lw=1.3, label="truth $\\log P$")
    ax.plot(f, ex["zero"] * DB, color=OI["vermilion"], lw=1.0,
            label="zeros (one record)")
    ax.plot(f, ex["welch"] * DB, color=OI["orange"], lw=0.8, ls="--",
            label="Welch")
    ax.plot(f, ex["minstat"] * DB, color=OI["green"], lw=0.8, ls="-.",
            label="min-stat (oracle cal.)")
    ax.plot(f, ex["mmse"] * DB, color=OI["purple"], lw=0.7, ls=":",
            label="MMSE-SPP")
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel("smoothed PSD (dB)")
    ax.set_xlim(f[0], f[-1])
    ax.legend(loc="lower center", ncols=2, fontsize=5.6)
    lpp = np.gradient(np.gradient(ex["truth"], f), f)
    ipk = np.argmax(ex["truth"])
    b = 0.7 * 500.0 ** (-1 / 5)
    bias_pk = 0.5 * b ** 2 * lpp[ipk] * DB
    ax.annotate(rf"kernel bias $\frac{{b^2}}{{2}}(\log P)'' \approx {bias_pk:+.1f}$ dB",
                xy=(f[ipk], ex["zero_hi"][ipk] * DB), xytext=(0.62, -3.4),
                fontsize=5.4, arrowprops=dict(arrowstyle="-", lw=0.6))


def panel_b(fig, ax):
    T, med, lo, hi = rt["T"], rt["med"], rt["lo"], rt["hi"]
    ax.fill_between(T, np.array(lo) * DB, np.array(hi) * DB, color=OI["blue"],
                    alpha=0.18, lw=0)
    ax.plot(T, np.array(med) * DB, "o-", color=OI["blue"], ms=3,
            label="sup-error (median, IQR)")
    guide = med[2] * (T / T[2]) ** (-2 / 5)
    ax.plot(T, guide * DB, ls="--", color=OI["grey"], lw=0.9,
            label=r"$T^{-2/5}$ (theory)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("record length $T$ (normalized)")
    ax.set_ylabel(r"$\sup_f|\hat\Lambda-\Lambda|$ (dB)")
    ax.legend(loc="upper right")
    ax.text(0.04, 0.06, f"empirical slope = {float(rt['slope']):.2f}",
            transform=ax.transAxes, fontsize=6)


PANELS = [("a", panel_a, (89 * MM, 62 * MM)),
          ("b", panel_b, (89 * MM, 62 * MM))]


def compose():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 0.34 * W2))
    for (letter, fn, _), ax in zip(PANELS, axes):
        fn(fig, ax)
        panel_label(ax, letter)
    savefig(fig, "fig2_estimator_rate", FIG)
    plt.close(fig)


def export_panels():
    for letter, fn, size in PANELS:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        fn(fig, ax)
        savefig(fig, f"fig2_{letter}", PAN)
        plt.close(fig)


if __name__ == "__main__":
    compose()
    export_panels()
    print("fig2 done")
