"""E2: estimator accuracy on clean colored noise and convergence rate (Fig. 2).

Outputs:
  results/e2_example.npz   one-record comparison at T=512 (all methods)
  results/e2_rate.npz      sup-error vs T for the zero estimator (MC quantiles)
"""
import os
import numpy as np
from zerospec import GaussSTFT, synth_stationary, ZeroSpecEstimator
from zerospec.theory import smoothed_psd
from zerospec import baselines

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(7)

# resonance + spectral tilt (exercises the a1 gauge) + side bump
S_test = lambda nu: np.exp(0.30 * nu) * (
    0.15 + 1.8 / (1 + ((nu - 0.4) / 0.35) ** 2) ** 2
    + 0.5 * np.exp(-(nu + 1.3) ** 2 / 0.18))

BAND = (-1.8, 1.8)
DB = 10 / np.log(10)          # nats -> dB


_MS_BIAS = {}


def _spectrogram(x, T, fg, dec=8):
    st = GaussSTFT(T)
    V, tg = st.stft(x, fg)
    sel = (tg >= 6) & (tg <= T - 6)
    P2 = np.abs(V[:, sel][:, ::dec]) ** 2          # ~1-unit frame spacing
    return P2, tg[sel][::dec]


def ms_bias(T, fg, n_cal=16):
    """Oracle per-row min-stat calibration from noise-only simulations."""
    key = (T, len(fg))
    if key not in _MS_BIAS:
        rngc = np.random.default_rng(999)
        noise_P2s, tcd = [], None
        for _ in range(n_cal):
            p2, tcd = _spectrogram(synth_stationary(S_test, T, rngc), T, fg)
            noise_P2s.append(p2)
        Ptrue = smoothed_psd(S_test, fg)
        _MS_BIAS[key] = (baselines.min_stat_oracle_bias(noise_P2s, tcd, Ptrue), tcd)
    return _MS_BIAS[key]


def one_run(T, seed=None, collect_baselines=False):
    r = np.random.default_rng(seed)
    x = synth_stationary(S_test, T, r)
    est = ZeroSpecEstimator(band=BAND, mask=False)
    res = est.fit(x, T)
    truth = np.log(smoothed_psd(S_test, res.f))
    out = dict(f=res.f, zero=res.Lambda, truth=truth)
    if collect_baselines:
        fg = np.arange(BAND[0], BAND[1] + 1e-9, 0.125)
        P2, tcd = _spectrogram(x, T, fg)
        bias, _ = ms_bias(T, fg)
        out["welch"] = np.log(baselines.welch_h(x, 1 / 16, res.f, seg_units=8.0))
        out["minstat"] = np.interp(res.f, fg, np.log(baselines.min_stat(P2, tcd, bias=bias)))
        out["mmse"] = np.interp(res.f, fg, np.log(baselines.mmse_spp(P2, tcd)))
    return out


def example(T=512.0, n_mc=40):
    runs = [one_run(T, seed=1000 + i, collect_baselines=True) for i in range(n_mc)]
    f = runs[0]["f"]
    truth = runs[0]["truth"]
    curves = {k: np.array([r[k] for r in runs]) for k in ("zero", "welch", "minstat", "mmse")}
    stats = {k: dict(
        rms_db=float(np.median([np.sqrt(np.mean((c - truth) ** 2)) for c in v]) * DB),
        sup_db=float(np.median([np.max(np.abs(c - truth)) for c in v]) * DB))
        for k, v in curves.items()}
    np.savez(os.path.join(RES, "e2_example.npz"), f=f, truth=truth,
             zero=curves["zero"][0], welch=curves["welch"][0],
             minstat=curves["minstat"][0], mmse=curves["mmse"][0],
             zero_med=np.median(curves["zero"], 0),
             zero_lo=np.quantile(curves["zero"], 0.1, 0),
             zero_hi=np.quantile(curves["zero"], 0.9, 0),
             stats=str(stats))
    print("example done:", stats)


def rate(Ts=(128, 256, 512, 1024, 2048, 4096), n_mc=40):
    med, lo, hi = [], [], []
    for T in Ts:
        errs = []
        for i in range(n_mc if T < 4096 else max(16, n_mc // 2)):
            r = one_run(float(T), seed=int(T) * 100 + i)
            errs.append(np.max(np.abs(r["zero"] - r["truth"])))
        errs = np.array(errs)
        med.append(np.median(errs)); lo.append(np.quantile(errs, 0.25)); hi.append(np.quantile(errs, 0.75))
        print(f"T={T}: sup-err median {med[-1]:.3f}")
    Ts = np.array(Ts, float)
    slope = np.polyfit(np.log(Ts), np.log(med), 1)[0]
    np.savez(os.path.join(RES, "e2_rate.npz"), T=Ts, med=med, lo=lo, hi=hi, slope=slope)
    print(f"rate done: empirical slope = {slope:.3f} (theory -0.4)")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "example"):
        example()
    if which in ("all", "rate"):
        rate()
    print("E2 complete.")
