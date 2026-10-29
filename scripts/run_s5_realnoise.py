"""S-V: zero density of the STFT of real white noise against frequency (SM Fig. 9).

For real white noise the circular prediction is rho = 1 at every f, so the
measured deviation rho(f) - 1 isolates the non-circularity.  The density is
tabulated on frequency bins of width 1/8 and compared with the exact law of
SM Proposition S1, averaged over each bin; the script prints the standardized
deviations for f >= 0.25 and the predicted deficit at f = 1.

Output: results/s5_realnoise.npz
"""
import os
import numpy as np
from zerospec import GaussSTFT, detect_zeros

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)

T, DT = 192.0, 1 / 16
N_MC = 300
FGRID = np.arange(-0.75, 3.0 + 1e-9, 0.125)   # detector margin below 0
T_MARGIN = 6.0


def rho_real_white(f):
    """Zero intensity of the STFT of real white noise (SM Proposition S1)."""
    x = 4 * np.pi * np.asarray(f, float) ** 2
    return (1 - (1 + x) * np.exp(-x)) / (1 - np.exp(-x)) ** 1.5


def rho_real_white_bin(c, h):
    """rho_real_white averaged over the frequency bin [c - h, c + h]."""
    u = np.linspace(c - h, c + h, 401)
    return np.trapezoid(rho_real_white(u), u) / (2 * h)


def main():
    rng = np.random.default_rng(55)
    st = GaussSTFT(T, DT)
    N = st.N
    per_run = []
    fc = tc = None
    for i in range(N_MC):
        x = rng.standard_normal(N) / np.sqrt(DT)        # real white noise, PSD 1
        V, tg = st.stft(x.astype(complex), FGRID)
        zs, fc, tc = detect_zeros(V, FGRID, tg)
        sel = (tc >= T_MARGIN) & (tc <= T - T_MARGIN)
        zs, tcs = zs[:, sel], tc[sel]
        dtp = tcs[1] - tcs[0]
        dfp = fc[1] - fc[0]
        per_run.append(zs.sum(axis=1) / ((tcs[-1] - tcs[0] + dtp) * dfp))
    arr = np.array(per_run)
    emp, se = arr.mean(0), arr.std(0, ddof=1) / np.sqrt(N_MC)
    sel = fc >= 0.25
    h = (fc[1] - fc[0]) / 2
    exact = np.array([rho_real_white_bin(c, h) for c in fc[sel]])
    z = (emp[sel] - exact) / se[sel]
    for ff, e, s, r in zip(fc[sel], emp[sel], se[sel], exact):
        print(f"f={ff:5.3f}  rho={e:.4f} +- {s:.4f}   exact law {r:.4f}")
    print(f"f >= 0.25: standardized deviations from the exact law, rms {np.sqrt(np.mean(z ** 2)):.2f}, "
          f"max {np.max(np.abs(z)):.2f}; predicted deficit at f = 1: {1 - rho_real_white(1.0):.1e}")
    np.savez(os.path.join(RES, "s5_realnoise.npz"), fc=fc, emp=emp, se=se, n_mc=N_MC, T=T)
    print("s5 done")


if __name__ == "__main__":
    main()
