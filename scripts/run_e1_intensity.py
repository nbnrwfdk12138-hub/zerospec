"""E1: Monte Carlo validation of the intensity theory (data generation).

Outputs results/e1_*.npz consumed by make_fig1.py:
  (a) one spectrogram realization + its zeros          -> e1_snapshot.npz
  (b) colored stationary density vs theory             -> e1_colored.npz
  (c) nonstationary (modulated) density vs theory      -> e1_modulated.npz
  (d) noncentral white+tone, log-depth suppression     -> e1_noncentral.npz
"""
import os
import numpy as np
from zerospec import GaussSTFT, detect_zeros, synth_stationary, synth_modulated
from zerospec.theory import smoothed_psd, rho_stationary, rho_modulated, \
    rho_noncentral_tone, bin_average

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(2026)

S_bimodal = lambda nu: 0.12 + 1.4 * np.exp(-(nu - 0.9) ** 2 / (2 * 0.55 ** 2)) \
                            + 0.9 * np.exp(-(nu + 1.1) ** 2 / (2 * 0.40 ** 2))


def mc_density_f(make_x, T, fgrid, n_mc, t_margin=6.0):
    st = GaussSTFT(T)
    per = []
    for _ in range(n_mc):
        V, tg = st.stft(make_x(), fgrid)
        zs, fc, tc = detect_zeros(V, fgrid, tg)
        sel = (tc >= t_margin) & (tc <= T - t_margin)
        area = (fgrid[1] - fgrid[0]) * (tc[sel][-1] - tc[sel][0])
        per.append(zs[:, sel].sum(axis=1) / area)
    arr = np.array(per)
    return fc, arr.mean(0), arr.std(0, ddof=1) / np.sqrt(n_mc)


# (a) snapshot
def snapshot():
    T = 48.0
    st = GaussSTFT(T)
    fgrid = np.arange(-2.6, 2.6 + 1e-9, 1 / 16)        # finer grid for the image
    x = synth_stationary(S_bimodal, T, rng)
    V, tg = st.stft(x, fgrid, tstep=1)
    zs, fc, tc = detect_zeros(V, fgrid, tg)
    iz, jz = np.nonzero(zs)
    np.savez(os.path.join(RES, "e1_snapshot.npz"),
             logP2=np.log10(np.abs(V) ** 2 + 1e-12), fgrid=fgrid, tgrid=tg,
             zf=fc[iz], zt=tc[jz],
             fline=np.linspace(-2.6, 2.6, 400),
             Ptrue=smoothed_psd(S_bimodal, np.linspace(-2.6, 2.6, 400)))
    print("snapshot done")


# (b) colored stationary
def colored(n_mc=250, T=192.0):
    fgrid = np.arange(-2.6, 2.6 + 1e-9, 0.125)
    fc, emp, se = mc_density_f(lambda: synth_stationary(S_bimodal, T, rng), T, fgrid, n_mc)
    fine = np.linspace(-2.3, 2.3, 500)
    np.savez(os.path.join(RES, "e1_colored.npz"), fc=fc, emp=emp, se=se,
             fine=fine, th_fine=rho_stationary(S_bimodal, fine),
             th_bins=bin_average(lambda F: np.interp(F, fine, rho_stationary(S_bimodal, fine)),
                                 fc, 0.125),
             n_mc=n_mc, T=T)
    print("colored done")


# (c) nonstationary
def modulated(n_mc=220, T=192.0, T0=4.0):
    sig2 = lambda t: 1.0 + 0.8 * np.sin(2 * np.pi * t / T0)
    fgrid = np.arange(-2.0, 2.0 + 1e-9, 0.125)
    st = GaussSTFT(T)
    per_run, cells, tcs0, dtp, nb = [], None, None, None, None
    for _ in range(n_mc):
        V, tg = st.stft(synth_modulated(sig2, T, rng), fgrid)
        zs, fc, tc = detect_zeros(V, fgrid, tg)
        sel = (tc >= 6) & (tc <= T - 6)
        zs, tcs = zs[:, sel], tc[sel]
        if nb is None:
            dtp = tcs[1] - tcs[0]
            nb = int(round(T0 / dtp))
            tcs0 = tcs[0]
            df_tot = (fc[-1] - fc[0]) + (fc[1] - fc[0])
            idx = np.round((tcs - tcs0) / dtp).astype(int) % nb
            cells = np.bincount(idx, minlength=nb) * dtp * df_tot
        acc = np.bincount(idx, weights=zs.sum(axis=0), minlength=nb)
        per_run.append(acc / cells)
    arr = np.array(per_run)
    tloc = tcs0 + dtp * np.arange(nb)
    fine = np.linspace(tloc[0], tloc[0] + T0, 300)
    np.savez(os.path.join(RES, "e1_modulated.npz"),
             tloc=tloc, emp=arr.mean(0), se=arr.std(0, ddof=1) / np.sqrt(n_mc),
             fine=fine, th_fine=rho_modulated(sig2, fine), T0=T0, n_mc=n_mc)
    print("modulated done")


# (d) noncentral
def noncentral(n_mc=400, T=192.0, A=2.0, f0=0.0):
    fgrid = np.arange(-2.0, 2.0 + 1e-9, 0.125)
    tone = lambda: A * np.exp(2j * np.pi * f0 * np.arange(int(T * 16)) / 16)
    white = lambda nu: np.ones_like(np.atleast_1d(nu), float)
    fc, emp, se = mc_density_f(lambda: synth_stationary(white, T, rng) + tone(),
                               T, fgrid, n_mc)
    fine = np.linspace(-2.0, 2.0, 600)
    np.savez(os.path.join(RES, "e1_noncentral.npz"), fc=fc, emp=emp, se=se, A=A,
             fine=fine, th_fine=rho_noncentral_tone(fine, A, f0),
             th_bins=bin_average(lambda F: rho_noncentral_tone(F, A, f0), fc, 0.125),
             supp_fine=np.exp(-A ** 2 * np.sqrt(2.0) * np.exp(-2 * np.pi * (fine - f0) ** 2)),
             n_mc=n_mc)
    print("noncentral done")


if __name__ == "__main__":
    snapshot()
    colored()
    modulated()
    noncentral()
    print("E1 data generation complete.")
