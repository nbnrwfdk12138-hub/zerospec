"""Panels of SM Figs. 6 and 7: detector validation and ablations."""
import os
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

ab = np.load(os.path.join(RES, "e4_ablations.npz"), allow_pickle=True)


def panel_a(fig, ax):
    steps = np.array(ab["a1_steps"], float)
    ax.errorbar(steps, ab["a1_vals"], yerr=2 * np.array(ab["a1_se"]), fmt="o-",
                ms=3, lw=1.0, color=OI["blue"], label="grid step $\\delta$")
    half = np.array(ab["a4_half"], float)
    ax.errorbar(half / 16, ab["a4_vals"], yerr=2 * np.array(ab["a4_se"]),
                fmt="s--", ms=3, lw=1.0, color=OI["green"],
                label="truncation $\\pm\\{3,4,5\\}$ (at $\\delta{=}1/8$)")
    ax.axhline(1.0, color=OI["grey"], lw=0.6, ls=":")
    ax.set_xlabel(r"grid step $\delta$ (units)")
    ax.set_ylabel("white-noise density")
    ax.set_ylim(0.99, 1.01)
    ax.legend(fontsize=5.0, loc="lower right")


def panel_b(fig, ax):
    cs = np.array(ab["a2_cs"], float)
    med = np.array(ab["a2_med"]) * DB
    lo = np.array(ab["a2_lo"]) * DB
    hi = np.array(ab["a2_hi"]) * DB
    ax.fill_between(cs, lo, hi, color=OI["blue"], alpha=0.18, lw=0)
    ax.plot(cs, med, "o-", ms=3, lw=1.0, color=OI["blue"])
    ax.axvline(0.7, color=OI["grey"], lw=0.6, ls=":")
    ax.text(0.72, med.max() * 0.95, "default", fontsize=5.4, color=OI["grey"])
    ax.set_xlabel(r"bandwidth constant $c$  ($b=cT^{-1/5}$)")
    ax.set_ylabel("rms error (dB)")


def panel_c(fig, ax):
    alphas = np.array(ab["a3_alphas"], float)
    dil = np.array(ab["a3_dilates"], float)
    contam = np.array(ab["a3_contam"]) * DB
    clean = np.array(ab["a3_clean"]) * DB
    for ia, al in enumerate(alphas):
        ax.plot(dil, contam[ia], "o-", ms=3, lw=1.0,
                color=[OI["sky"], OI["blue"], OI["purple"]][ia],
                label=rf"$\alpha={al}$, contaminated")
    ax.plot(dil, clean[1], "s--", ms=3, lw=0.9, color=OI["green"],
            label="clean rms, $\\alpha\\leq0.008$")
    ax.plot(dil, clean[2], "s:", ms=2.5, lw=0.8, color=OI["green"], alpha=0.6,
            label="clean rms, $\\alpha=0.025$")
    ax.axhline(0, color=OI["grey"], lw=0.6, ls=":")
    ax.set_xlabel("mask dilation (cells)")
    ax.set_ylabel("bias / rms (dB)")
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(-19, 4)
    ax.legend(fontsize=4.8, loc="lower right")


def panel_d(fig, ax):
    al5 = np.array(ab["a5_alphas"], float)
    ax2 = ax.twinx()
    l1, = ax.plot(al5, ab["a5_rmsz"], "o-", ms=3, lw=1.0, color=OI["blue"],
                  label="intensity rms($z$)")
    l2, = ax2.plot(al5, np.array(ab["a5_estrms"]) * DB, "s--", ms=3, lw=1.0,
                   color=OI["vermilion"], label="estimator rms (dB)")
    ax.axhline(1.0, color=OI["grey"], lw=0.6, ls=":")
    ax.set_xlabel(r"power-law exponent $a$  ($S\propto(0.05+|\nu|)^{-a}$)")
    ax.set_ylabel("intensity rms($z$)", color=OI["blue"])
    ax2.set_ylabel("estimator rms (dB)", color=OI["vermilion"])
    ax2.spines["right"].set_visible(True)
    ax.legend(handles=[l1, l2], fontsize=5.0, loc="upper left")


PANELS = [("a", panel_a, (60 * MM, 50 * MM)),
          ("b", panel_b, (60 * MM, 50 * MM)),
          ("c", panel_c, (60 * MM, 50 * MM)),
          ("d", panel_d, (60 * MM, 50 * MM))]


def compose():
    fig, axes = plt.subplots(1, 4, figsize=(W2, 0.27 * W2))
    for (letter, fn, _), ax in zip(PANELS, axes):
        fn(fig, ax)
        panel_label(ax, letter)
    savefig(fig, "fig6_ablations", FIG)
    plt.close(fig)


def export_panels():
    for letter, fn, size in PANELS:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        fn(fig, ax)
        savefig(fig, f"fig6_{letter}", PAN)
        plt.close(fig)


if __name__ == "__main__":
    compose()
    export_panels()
    print("fig6 done")
