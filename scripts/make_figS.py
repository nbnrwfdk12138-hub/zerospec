"""Panel of SM Fig. 9: zero density of real white noise.

The density from results/s5_realnoise.npz is plotted against frequency with the
circular prediction rho = 1 and the exact law of SM Proposition S1, which has
no fitted parameter.

Output: figures/panels_r1/figS_realnoise.{pdf,png}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from zerospec import figstyle
from zerospec.figstyle import OI, MM, savefig

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "figures", "panels_r1")
figstyle.set_style()


def panel_realnoise(fig, ax):
    d = np.load(os.path.join(RES, "s5_realnoise.npz"))
    fc, emp, se = d["fc"], d["emp"], d["se"]
    keep = (fc >= 0.25) & (fc <= 2.6)
    ax.axhline(1.0, color=OI["blue"], lw=1.0, label=r"circular model, $\rho\equiv1$")
    ff = np.linspace(0.25, 2.6, 400)
    x = 4 * np.pi * ff ** 2
    exact = (1 - (1 + x) * np.exp(-x)) / (1 - np.exp(-x)) ** 1.5
    ax.plot(ff, exact, color=OI["grey"], lw=0.9, ls="--",
            label="exact law for real noise, no fitted parameter")
    ax.errorbar(fc[keep], emp[keep], yerr=2 * se[keep], fmt="o", ms=2.4,
                lw=0.7, color=OI["vermilion"], mfc="white", mew=0.7,
                label=f"real white noise, Monte Carlo ({int(d['n_mc'])} records)")
    ax.set_xlabel("frequency $f$ (normalized)")
    ax.set_ylabel(r"zero density $\rho$")
    ax.set_xlim(0.2, 2.65)
    ax.set_ylim(0.5, 1.1)
    ax.legend(loc="lower right")


if __name__ == "__main__":
    fig = plt.figure(figsize=(89 * MM, 58 * MM))
    ax = fig.add_subplot(111)
    panel_realnoise(fig, ax)
    savefig(fig, "figS_realnoise", OUT)
    plt.close(fig)
    print("figS done")
