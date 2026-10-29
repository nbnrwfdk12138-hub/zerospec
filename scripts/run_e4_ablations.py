"""E4: ablations (SM Figs. 6 and 7).

A1  detector grid step delta in {1/4, 1/8, 1/16}: white-noise density
A2  bandwidth constant c in {0.35, ..., 1.4}: estimator rms error at T=512
A3  masking (alpha, dilation): bias under a chirp and cost on clean noise
A4  window truncation +-{3, 4, 5}: white-noise density
A5  power-law spectra (0.05 + |nu|)^-a, a in {0.5, 1.0, 1.5}: intensity
    formula and estimator

Outputs: results/e4_ablations.npz
"""
import os
import numpy as np
from zerospec.core import GaussSTFT, detect_zeros, synth_stationary
from zerospec.estimator import ZeroSpecEstimator
from zerospec.theory import smoothed_psd, rho_stationary

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
DB = 10 / np.log(10)
WHITE = lambda nu: np.ones_like(np.atleast_1d(nu), float)

S_test = lambda nu: np.exp(0.30 * nu) * (
    0.15 + 1.8 / (1 + ((nu - 0.4) / 0.35) ** 2) ** 2
    + 0.5 * np.exp(-(nu + 1.3) ** 2 / 0.18))


# A1: grid step
def a1_grid(n_mc=40, T=192.0):
    out = {}
    for step, tstep in [(0.25, 4), (0.125, 2), (0.0625, 1)]:
        rng = np.random.default_rng(31)
        st = GaussSTFT(T)
        fg = np.arange(-2.5, 2.5 + 1e-9, step)
        dens = []
        for _ in range(n_mc):
            V, tg = st.stft(synth_stationary(WHITE, T, rng), fg, tstep=tstep)
            zs, fc, tc = detect_zeros(V, fg, tg)
            sel = (tc >= 6) & (tc <= T - 6)
            area = step * (tc[sel][-1] - tc[sel][0]) * (len(fc))
            dens.append(zs[:, sel].sum() / area)
        out[step] = (float(np.mean(dens)), float(np.std(dens) / np.sqrt(n_mc)))
        print(f"A1 delta={step}: density {out[step][0]:.4f} +- {out[step][1]:.4f}")
    return out


# A2: bandwidth constant
def a2_bandwidth(n_mc=30, T=512.0):
    cs = [0.35, 0.5, 0.7, 1.0, 1.4]
    out = {}
    for c in cs:
        errs = []
        for i in range(n_mc):
            rng = np.random.default_rng(5000 + i)
            x = synth_stationary(S_test, T, rng)
            bw = c * (T - 12) ** (-1 / 5)
            est = ZeroSpecEstimator(band=(-1.8, 1.8), mask=False, bandwidth=bw)
            r = est.fit(x, T)
            truth = np.log(smoothed_psd(S_test, r.f))
            errs.append(np.sqrt(np.mean((r.Lambda - truth) ** 2)))
        out[c] = (float(np.median(errs)), float(np.quantile(errs, .25)),
                  float(np.quantile(errs, .75)))
        print(f"A2 c={c}: rms {out[c][0]*DB:.2f} dB")
    return out


# A3: masking parameters
def a3_masking(n_mc=12, T=512.0):
    """Clean-noise cost and chirp-contamination bias on the (alpha,dilate)
    grid, on a smooth bimodal spectrum (sharp quasi-line resonances are
    declared-bridged by design and excluded from this ablation)."""
    from run_e4_helpers import chirp_mix, S_smooth
    alphas = [0.002, 0.008, 0.025]
    dilates = [1, 2, 3]
    clean = np.zeros((len(alphas), len(dilates)))
    contam = np.zeros((len(alphas), len(dilates)))
    for ia, al in enumerate(alphas):
        for idl, dl in enumerate(dilates):
            ce, be = [], []
            for i in range(n_mc):
                rng = np.random.default_rng(9000 + i)
                x = synth_stationary(S_smooth, T, rng)
                est = ZeroSpecEstimator(band=(-1.8, 1.8), mask=True,
                                        alpha=al, dilate=dl)
                r = est.fit(x, T)
                truth = np.log(smoothed_psd(S_smooth, r.f))
                ce.append(np.sqrt(np.mean((r.Lambda - truth) ** 2)))
                y, _ = chirp_mix(x, T, rng, gamma0=30.0)
                r2 = est.fit(y, T)
                be.append(np.mean(r2.Lambda - truth))
            clean[ia, idl] = np.median(ce)
            contam[ia, idl] = np.median(be)
            print(f"A3 alpha={al} dilate={dl}: clean rms {clean[ia,idl]*DB:.2f} dB, "
                  f"contaminated bias {contam[ia,idl]*DB:+.2f} dB")
    return alphas, dilates, clean, contam


# A4: window truncation
def a4_truncation(n_mc=40, T=192.0):
    out = {}
    for halfw in (3.0, 4.0, 5.0):
        rng = np.random.default_rng(77)
        st = GaussSTFT(T)
        half = int(round(halfw / st.dt))
        tgw = np.arange(-half, half + 1) * st.dt
        st.window = 2.0 ** 0.25 * np.exp(-np.pi * tgw ** 2)
        fg = np.arange(-2.5, 2.5 + 1e-9, 0.125)
        dens = []
        for _ in range(n_mc):
            V, tg = st.stft(synth_stationary(WHITE, T, rng), fg)
            zs, fc, tc = detect_zeros(V, fg, tg)
            sel = (tc >= 6) & (tc <= T - 6)
            area = 0.125 * (tc[sel][-1] - tc[sel][0]) * len(fc)
            dens.append(zs[:, sel].sum() / area)
        out[halfw] = (float(np.mean(dens)), float(np.std(dens) / np.sqrt(n_mc)))
        print(f"A4 +-{halfw}: density {out[halfw][0]:.4f} +- {out[halfw][1]:.4f}")
    return out


# A5: power-law spectra
def a5_powerlaw(n_mc=120, T=192.0):
    """1/f^a spectra: intensity formula check + estimator rms (T=512)."""
    res = {}
    for a in (0.5, 1.0, 1.5):
        S = lambda nu, a=a: (0.05 + np.abs(nu)) ** (-a)
        rng = np.random.default_rng(123)
        st = GaussSTFT(T)
        fg = np.arange(-2.5, 2.5 + 1e-9, 0.125)
        per = []
        for _ in range(n_mc):
            V, tg = st.stft(synth_stationary(S, T, rng), fg)
            zs, fc, tc = detect_zeros(V, fg, tg)
            sel = (tc >= 6) & (tc <= T - 6)
            area = 0.125 * (tc[sel][-1] - tc[sel][0])
            per.append(zs[:, sel].sum(axis=1) / area)
        arr = np.array(per)
        emp, se = arr.mean(0), arr.std(0, ddof=1) / np.sqrt(n_mc)
        th = rho_stationary(S, fc)
        keep = (fc > -2.2) & (fc < 2.2)
        z = (emp - th)[keep] / np.maximum(se[keep], 5e-3)
        # estimator rms at T=512
        errs = []
        for i in range(24):
            rng2 = np.random.default_rng(4000 + i)
            x = synth_stationary(S, 512.0, rng2)
            est = ZeroSpecEstimator(band=(-1.8, 1.8), mask=False)
            r = est.fit(x, 512.0)
            truth = np.log(smoothed_psd(S, r.f))
            errs.append(np.sqrt(np.mean((r.Lambda - truth) ** 2)))
        res[a] = dict(fc=fc[keep], emp=emp[keep], se=se[keep], th=th[keep],
                      rmsz=float(np.sqrt(np.mean(z ** 2))),
                      est_rms=float(np.median(errs)))
        print(f"A5 a={a}: intensity rms(z)={res[a]['rmsz']:.2f}, "
              f"estimator rms {res[a]['est_rms']*DB:.2f} dB")
    return res


if __name__ == "__main__":
    a1 = a1_grid()
    a2 = a2_bandwidth()
    al, dl, clean, contam = a3_masking()
    a4 = a4_truncation()
    a5 = a5_powerlaw()
    np.savez(os.path.join(RES, "e4_ablations.npz"),
             a1_steps=list(a1.keys()), a1_vals=[v[0] for v in a1.values()],
             a1_se=[v[1] for v in a1.values()],
             a2_cs=list(a2.keys()), a2_med=[v[0] for v in a2.values()],
             a2_lo=[v[1] for v in a2.values()], a2_hi=[v[2] for v in a2.values()],
             a3_alphas=al, a3_dilates=dl, a3_clean=clean, a3_contam=contam,
             a4_half=list(a4.keys()), a4_vals=[v[0] for v in a4.values()],
             a4_se=[v[1] for v in a4.values()],
             **{f"a5_{k}_{a}": v[k] for a, v in a5.items()
                for k in ("fc", "emp", "se", "th")},
             a5_alphas=list(a5.keys()),
             a5_rmsz=[v["rmsz"] for v in a5.values()],
             a5_estrms=[v["est_rms"] for v in a5.values()])
    print("E4 complete.")
