"""E5: detection rates of the masking rule and runtime (SM Sec. S-III-A).

(1) Detection probability of a persistent in-band line against the per-cell
    SNR gamma0, with the matched false-alarm probability (same decision
    region, no tone).
(2) Wall-clock time of the zero estimator and of Welch on one record.

Uses the signal model and SNR convention of run_e3_robustness.py.

Outputs: results/e5_roc.npz; prints a summary.
"""
import os, time
import numpy as np
from zerospec import GaussSTFT, synth_stationary, ZeroSpecEstimator
from zerospec import baselines
import run_e3_robustness as e3

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
DB = 10 / np.log(10)
F_TONE = e3.F_TONE
BAND, T, DT, NU_MAX, MARGIN = e3.BAND, e3.T, e3.DT, e3.NU_MAX, e3.MARGIN
HALF = 0.6  # decision region half-width (units) around the tone frequency


def tone_region_masked(r):
    # a persistent line depletes its whole frequency row across time (row test),
    # so it is masked at most times; a transient strong noise dweller masks only
    # a few time cells (level test).  Declare the line detected when the row is
    # masked over more than half of its time extent.
    masked = r.diagnostics["masked"]
    fc = r.diagnostics["fc"]
    rows = np.abs(fc - F_TONE) <= HALF
    if not rows.any():
        return False
    return float(masked[rows].mean()) > 0.5


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def roc(gammas, n_mc=60, n_fa=200):
    est = ZeroSpecEstimator(band=BAND, margin=MARGIN, mask=True)
    # false alarm: same decision region, pure noise, no tone
    fa = 0
    for i in range(n_fa):
        rng = np.random.default_rng(900_000 + i)
        x = synth_stationary(e3.S_noise, T, rng, nu_max=NU_MAX)
        fa += tone_region_masked(est.fit(x, T))
    p_fa = fa / n_fa
    pdet, lo, hi = [], [], []
    for g in gammas:
        k = 0
        for i in range(n_mc):
            rng = np.random.default_rng(int(g * 1000) + i)
            x = synth_stationary(e3.S_noise, T, rng, nu_max=NU_MAX)
            s, _ = e3.make_signal(g, 0, True, T, rng)   # tone only, no sweeps
            k += tone_region_masked(est.fit(x + s, T))
        p = k / n_mc
        l, h = wilson(k, n_mc)
        pdet.append(p); lo.append(l); hi.append(h)
        print(f"  gamma0={g:5.1f}: P_det={p:.3f} [{l:.2f},{h:.2f}]  (k={k}/{n_mc})")
    return np.array(pdet), np.array(lo), np.array(hi), p_fa


def crossing(gammas, pdet, level=0.9):
    for i in range(1, len(gammas)):
        if pdet[i - 1] < level <= pdet[i]:
            # linear interpolation in log-gamma
            x0, x1 = np.log(gammas[i - 1]), np.log(gammas[i])
            y0, y1 = pdet[i - 1], pdet[i]
            return float(np.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0)))
    return float(gammas[0]) if pdet[0] >= level else float("nan")


def runtime(reps=20):
    rng = np.random.default_rng(2024)
    x = synth_stationary(e3.S_noise, T, rng, nu_max=NU_MAX)
    fref = np.arange(BAND[0] + 0.3, BAND[1] - 0.3 + 1e-9, 0.05)
    est = ZeroSpecEstimator(band=BAND, margin=MARGIN, mask=True)
    est.fit(x, T); baselines.welch(x, DT, fref, seg_units=8.0)  # warmup
    tz = []
    for _ in range(reps):
        t0 = time.perf_counter(); est.fit(x, T); tz.append(time.perf_counter() - t0)
    tw = []
    for _ in range(reps):
        t0 = time.perf_counter(); baselines.welch(x, DT, fref, seg_units=8.0); tw.append(time.perf_counter() - t0)
    tz, tw = np.median(tz), np.median(tw)
    print(f"  zeros={tz*1e3:.1f} ms  welch={tw*1e3:.1f} ms  ratio={tz/tw:.1f}x  "
          f"(N={int(T/DT)} samples)")
    return tz, tw, int(T / DT)


if __name__ == "__main__":
    gammas = np.array([1., 2., 3., 4., 5., 7., 10., 14., 20., 30., 50.])
    print("ROC (persistent tone):")
    pdet, lo, hi, p_fa = roc(gammas)
    g90 = crossing(gammas, pdet, 0.9)
    print(f"false alarm P_fa={p_fa:.4f} ; P_det>=0.9 at gamma0~{g90:.1f}")
    print("Runtime:")
    tz, tw, N = runtime()
    np.savez(os.path.join(RES, "e5_roc.npz"),
             gammas=gammas, pdet=pdet, lo=lo, hi=hi, p_fa=p_fa, g90=g90,
             t_zeros=tz, t_welch=tw, n_samples=N)
    print("E5 complete.")
