"""E3: strong in-band signals over synthetic colored noise (Fig. 3).

Band of 11 normalized units with full-band sweeping chirps, so that each
frequency row is occupied only part of the time, plus one persistent tone,
which is masked and bridged.  The local SNR gamma0 is constant along each
track.

Methods: zeros (masked and unmasked), Welch (mean and median), oracle-
calibrated minimum statistics, MMSE-SPP.  Metrics: full-band mean
log-spectral bias, and the bias at the tone rows (the bridged region).

Outputs: results/e3_example.npz, results/e3_curves.npz
"""
import os
import numpy as np
from zerospec import GaussSTFT, detect_zeros, synth_stationary, ZeroSpecEstimator
from zerospec.theory import smoothed_psd
from zerospec import baselines

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)

BAND = (-5.5, 5.5)
T = 512.0
DT = 1 / 16
NU_MAX = 7.5
DB = 10 / np.log(10)
F_TONE = -4.5
MARGIN = 1.5

S_noise = lambda nu: np.exp(0.08 * nu) * (
    0.18 + 0.9 * np.exp(-(nu + 3.5) ** 2 / (2 * 1.2 ** 2))
    + 1.1 * np.exp(-(nu - 1.5) ** 2 / (2 * 1.8 ** 2)))


def d_gamma1(g0):
    """footprint half-width where local SNR drops to 1."""
    return np.sqrt(max(np.log(g0), 1e-6) / (2 * np.pi))


def make_signal(gamma0, n_sweep, with_tone, T, rng):
    tau = np.arange(int(T / DT)) * DT
    x = np.zeros_like(tau, dtype=complex)
    tracks = []
    spans = [(-5.2, 5.2), (5.0, -5.0), (-4.8, 5.0)]
    for k in range(n_sweep):
        f0, f1 = spans[k % 3]
        finst = f0 + (f1 - f0) * tau / T
        phase = 2 * np.pi * np.cumsum(finst) * DT
        A = np.sqrt(gamma0 * smoothed_psd(S_noise, finst) / np.sqrt(2.0))
        x += A * np.exp(1j * (phase + 2 * np.pi * rng.uniform()))
        tracks.append(("sweep", f0, f1))
    if with_tone:
        A = np.sqrt(gamma0 * smoothed_psd(S_noise, [F_TONE])[0] / np.sqrt(2.0))
        x += A * np.exp(1j * (2 * np.pi * F_TONE * tau + 2 * np.pi * rng.uniform()))
        tracks.append(("tone", F_TONE, F_TONE))
    return x, tracks


def area_occupancy(gamma0, n_sweep, with_tone):
    d1 = d_gamma1(gamma0)
    bw = BAND[1] - BAND[0]
    occ = n_sweep * 2 * d1 / bw + (2 * d1 / bw if with_tone else 0.0)
    return min(occ, 1.0)


_MS_BIAS = {}


def _spectrogram(x, fg, dec=8):
    st = GaussSTFT(T)
    V, tg = st.stft(x, fg)
    sel = (tg >= 6) & (tg <= T - 6)
    return np.abs(V[:, sel][:, ::dec]) ** 2, tg[sel][::dec]


def ms_bias(fg, n_cal=12):
    key = len(fg)
    if key not in _MS_BIAS:
        rngc = np.random.default_rng(777)
        noise_P2s, tcd = [], None
        for _ in range(n_cal):
            p2, tcd = _spectrogram(
                synth_stationary(S_noise, T, rngc, nu_max=NU_MAX), fg)
            noise_P2s.append(p2)
        Ptrue = smoothed_psd(S_noise, fg)
        _MS_BIAS[key] = (baselines.min_stat_oracle_bias(noise_P2s, tcd, Ptrue), tcd)
    return _MS_BIAS[key]


def run_methods(y, fref):
    out = {}
    est = ZeroSpecEstimator(band=BAND, margin=MARGIN, mask=True)
    r = est.fit(y, T)
    out["zeros"] = np.interp(fref, r.f, r.Lambda)
    r0 = ZeroSpecEstimator(band=BAND, margin=MARGIN, mask=False).fit(y, T)
    out["zeros_nomask"] = np.interp(fref, r0.f, r0.Lambda)
    fg = np.arange(BAND[0] - 0.5, BAND[1] + 0.5 + 1e-9, 0.125)
    P2, tcd = _spectrogram(y, fg)
    bias, _ = ms_bias(fg)
    out["welch"] = np.log(baselines.welch_h(y, DT, fref, seg_units=8.0))
    out["welch_med"] = np.log(baselines.welch_h(y, DT, fref, seg_units=8.0,
                                              average="median"))
    out["minstat"] = np.interp(fref, fg, np.log(baselines.min_stat(P2, tcd, bias=bias)))
    out["mmse"] = np.interp(fref, fg, np.log(baselines.mmse_spp(P2, tcd)))
    return out, r


METHODS = ["zeros", "zeros_nomask", "welch", "welch_med", "minstat", "mmse"]


def condition(gamma0, n_sweep, with_tone, n_mc, seed0):
    fref = np.arange(BAND[0] + 0.3, BAND[1] - 0.3 + 1e-9, 0.05)
    truth = np.log(smoothed_psd(S_noise, fref))
    tone_rows = np.abs(fref - F_TONE) <= d_gamma1(max(gamma0, 3.0))
    acc = {m: dict(full=[], tone=[]) for m in METHODS}
    for i in range(n_mc):
        rng = np.random.default_rng(seed0 + i)
        x = synth_stationary(S_noise, T, rng, nu_max=NU_MAX)
        s, _ = make_signal(gamma0, n_sweep, with_tone, T, rng)
        res, _ = run_methods(x + s, fref)
        for m in METHODS:
            d = res[m] - truth
            acc[m]["full"].append(np.mean(d))
            if with_tone:
                acc[m]["tone"].append(np.mean(d[tone_rows]))
    out = {}
    for m in METHODS:
        q = lambda v: (np.median(v), np.quantile(v, .25), np.quantile(v, .75)) \
            if len(v) else (np.nan,) * 3
        out[m] = dict(full=q(acc[m]["full"]), tone=q(acc[m]["tone"]))
    return out


def sweep(n_mc=16):
    gammas = [3.0, 10.0, 30.0, 100.0]
    res_g = {m: dict(full=[], tone=[]) for m in METHODS}
    for g in gammas:
        c = condition(g, 1, True, n_mc, seed0=int(g * 1000))
        for m in METHODS:
            res_g[m]["full"].append(c[m]["full"])
            res_g[m]["tone"].append(c[m]["tone"])
        print(f"gamma={g}: " + "  ".join(
            f"{m}={c[m]['full'][0]*DB:+.2f}dB" for m in METHODS))
    configs = [(1, False), (1, True), (2, True), (3, True)]
    occs = [area_occupancy(30.0, k, t) for k, t in configs]
    res_o = {m: dict(full=[], tone=[]) for m in METHODS}
    for (k, tn) in configs:
        c = condition(30.0, k, tn, n_mc, seed0=70_000 + 13 * k + (1 if tn else 0))
        for m in METHODS:
            res_o[m]["full"].append(c[m]["full"])
            res_o[m]["tone"].append(c[m]["tone"])
        print(f"config k={k} tone={tn}: " + "  ".join(
            f"{m}={c[m]['full'][0]*DB:+.2f}dB" for m in METHODS))
    np.savez(os.path.join(RES, "e3_curves.npz"),
             gammas=gammas, occs=occs,
             **{f"g_full_{m}": np.array(res_g[m]["full"]) for m in METHODS},
             **{f"g_tone_{m}": np.array(res_g[m]["tone"]) for m in METHODS},
             **{f"o_full_{m}": np.array(res_o[m]["full"]) for m in METHODS},
             **{f"o_tone_{m}": np.array(res_o[m]["tone"]) for m in METHODS})
    print("curves done")


def example(gamma0=30.0):
    rng = np.random.default_rng(424242)
    x = synth_stationary(S_noise, T, rng, nu_max=NU_MAX)
    s, tracks = make_signal(gamma0, 1, True, T, rng)
    y = x + s
    fref = np.arange(BAND[0] + 0.3, BAND[1] - 0.3 + 1e-9, 0.05)
    truth = np.log(smoothed_psd(S_noise, fref))
    res, r = run_methods(y, fref)
    st = GaussSTFT(T)
    fg = np.arange(BAND[0], BAND[1] + 1e-9, 1 / 16)
    V, tg = st.stft(y, fg, tstep=4)
    zs, fc, tc = detect_zeros(V, fg, tg)
    np.savez(os.path.join(RES, "e3_example.npz"),
             logP2=np.log10(np.abs(V[:, ::2]) ** 2 + 1e-12), fg=fg, tg=tg[::2],
             zf=fc[np.nonzero(zs)[0]], zt=tc[np.nonzero(zs)[1]],
             masked=r.diagnostics["masked"], mfc=r.diagnostics["fc"],
             mtc=r.diagnostics["tc"], exposure=r.exposure, fexp=r.f,
             bridged=r.bridged, fref=fref, truth=truth, gamma0=gamma0,
             **res)
    print("example done")


if __name__ == "__main__":
    example()
    sweep()
    print("E3 complete.")
