"""E3c: noise PSD estimation during continuous speech (DEMAND + LibriSpeech), Fig. 5.

Analysis scale sigma_w = 16 ms (1 unit = 62.5 Hz): L = 2048 -> delta-f' = 1/8,
hop = 64 samples (4 ms) -> delta-t' = 1/4.  Band 125--7000 Hz (about 110 units,
subband calibration).  Ground truth: the spectrum of the noise segment used in
the mixture (Welch, smoothed to the analysis resolution).  Minimum statistics
is calibrated on a disjoint noise-only segment of the same recording.

Outputs: results/e3c_speech.npz (and e3c_example.npz for the figure)
"""
import os
import glob
import numpy as np
import scipy.signal as sg
import soundfile as sf
from zerospec.core import framed_stft, detect_zeros
from zerospec.estimator import ZeroSpecEstimator
from zerospec import baselines

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(RES, exist_ok=True)

FS = 16000
SIG_W = 1.0 / 62.5            # 16 ms; 1 unit = 62.5 Hz
L = 2048                      # delta f = 7.8125 Hz = 1/8 unit
HOP = 64                      # 4 ms = 1/4 unit
BAND_HZ = (125.0, 7000.0)
DUR = 42.0                    # mixture length (s)
DB = 10 / np.log(10)

_g_taps = (2.0 / SIG_W ** 2) ** 0.25 * np.exp(
    -np.pi * (np.arange(-L // 2, L // 2) / FS) ** 2 / SIG_W ** 2)


def smooth_to_analysis(f1s, S1s, fline):
    """P_ref(f) = 0.5 (S_1s * |ghat|^2)(f) in V-units."""
    ghat = np.fft.fft(np.fft.ifftshift(_g_taps)) / FS
    fg = np.arange(L) * FS / L
    gf = np.concatenate([fg[fg < FS / 2], fg[fg >= FS / 2] - FS])
    order = np.argsort(gf)
    gfo, g2o = gf[order], np.abs(ghat[order]) ** 2
    core = np.abs(gfo) < 8 * 62.5
    return np.array([0.5 * np.trapezoid(np.interp(f - gfo[core], f1s, S1s) * g2o[core],
                                        gfo[core]) for f in fline])


def load_noise(env):
    x, fs = sf.read(os.path.join(DATA, "demand", "extracted", env, "ch01.wav"))
    assert fs == FS
    return x


def load_speech(seed, need_s=DUR):
    files = sorted(glob.glob(os.path.join(
        DATA, "speech", "extracted", "LibriSpeech", "test-clean", "*", "*", "*.flac")))
    rng = np.random.default_rng(seed)
    out = []
    n = 0
    while n < int(need_s * FS):
        f = files[int(rng.integers(len(files)))]
        s, fs = sf.read(f)
        assert fs == FS
        out.append(s)
        n += len(s)
    return np.concatenate(out)[: int(need_s * FS)]


EULER_G = 0.5772156649015329


def _spread_table():
    """Numerically calibrate the lower-spread statistic (median - q25 of
    log|V|^2) on the exponential-lognormal family: returns interpolators
    spread -> Var(log P).  Anchored at sigma=0 by the exact Exp value."""
    rngc = np.random.default_rng(2718)
    sigs = np.arange(0.0, 2.51, 0.125)
    spreads = []
    n = 200_000
    e = np.log(-np.log(rngc.uniform(size=n)))   # log Exp(1) (negated CDF trick)
    zn = rngc.standard_normal(n)
    for s in sigs:
        x = e + s * zn
        spreads.append(np.median(x) - np.quantile(x, 0.25))
    return np.array(spreads), sigs ** 2


_SPREADS, _VARS = _spread_table()


def run_methods(y, fline, Pref, ms_bias_rows, fb_hz):
    V, f_hz, t_s = framed_stft(y, FS, SIG_W, L, HOP, BAND_HZ, margin_hz=70.0)
    fpn, tpn = f_hz * SIG_W, t_s / SIG_W
    est = ZeroSpecEstimator(band=(BAND_HZ[0] * SIG_W, BAND_HZ[1] * SIG_W),
                            margin=1.0, mask=True, t_margin=6.0)
    r = est.fit_stft_banded(V, fpn, tpn, T=tpn[-1], chunk=12.0, overlap=2.0)
    out = {"zeros": np.interp(fline, r.f / SIG_W, r.Lambda)}
    inb = (fpn >= BAND_HZ[0] * SIG_W) & (fpn <= BAND_HZ[1] * SIG_W)
    P2 = np.abs(V[inb, ::4].astype(np.complex128)) ** 2     # ~1-unit frames
    tcd = tpn[::4]
    # oracle-free arithmetic-mean correction: under within-record noise power
    # fluctuation the zero estimator returns the geometric-mean spectrum;
    # estimate Var(log P) per row from the lower quantile spread of log|V|^2
    # (robust to speech outliers; Exp component subtracted in quadrature)
    lp = np.log(np.maximum(P2, 1e-300))
    spread = np.median(lp, axis=1) - np.quantile(lp, 0.25, axis=1)
    sig2_logP = np.interp(spread, _SPREADS, _VARS)   # calibrated mapping
    out["zeros_arith"] = out["zeros"] + 0.5 * np.interp(fline, fb_hz, sig2_logP)
    f1s, S1s = sg.welch(y, fs=FS, nperseg=4096, noverlap=2048)
    out["welch_on"] = np.log(np.maximum(smooth_to_analysis(f1s, S1s, fline), 1e-30))
    out["minstat"] = np.interp(fline, fb_hz, np.log(np.maximum(
        baselines.min_stat_raw(P2, tcd) / ms_bias_rows, 1e-30)))
    out["mmse"] = np.interp(fline, fb_hz, np.log(np.maximum(
        baselines.mmse_spp(P2, tcd), 1e-30)))
    return out, r, (V, fpn, tpn)


def setup_env(env, fline):
    """Truths are the spectra of the noise segment used in the
    mixture (standard speech-enhancement practice; eliminates between-segment
    drift of nonstationary environments from the metric).  Min-stat is
    calibrated on a disjoint noise-only segment."""
    noise = load_noise(env)
    nseg = noise[int(60 * FS): int(60 * FS) + int(DUR * FS)]
    f1s, S1s = sg.welch(nseg, fs=FS, nperseg=8192, noverlap=4096)
    Pref = smooth_to_analysis(f1s, S1s, fline)
    # geometric truth <log P>_t of the same segment (Euler debias)
    Vg, fg_hz, _ = framed_stft(nseg, FS, SIG_W, L, HOP, BAND_HZ, margin_hz=70.0)
    fpn_g = fg_hz * SIG_W
    inb_g = (fpn_g >= BAND_HZ[0] * SIG_W) & (fpn_g <= BAND_HZ[1] * SIG_W)
    lgeo_rows = np.mean(np.log(np.abs(Vg[inb_g, ::4].astype(np.complex128)) ** 2
                               + 1e-300), axis=1) + EULER_G
    Lgeo = np.interp(fline, fg_hz[inb_g], lgeo_rows)
    # min-stat calibration on a disjoint noise-only 42 s
    cal = noise[int(280 * FS): int(280 * FS) + int(DUR * FS)]
    Vc, fc_hz, tc_s = framed_stft(cal, FS, SIG_W, L, HOP, BAND_HZ, margin_hz=70.0)
    fpn = fc_hz * SIG_W
    inb = (fpn >= BAND_HZ[0] * SIG_W) & (fpn <= BAND_HZ[1] * SIG_W)
    P2c = np.abs(Vc[inb, ::4].astype(np.complex128)) ** 2
    raw = baselines.min_stat_raw(P2c, (tc_s / SIG_W)[::4])
    fb_hz = fc_hz[inb]
    ms_bias_rows = raw / np.maximum(np.interp(fb_hz, fline, Pref), 1e-30)
    return nseg, Pref, Lgeo, ms_bias_rows, fb_hz


METHODS = ["zeros", "zeros_arith", "welch_on", "minstat", "mmse"]
# declared estimation target of each method: geometric or arithmetic mean
TARGET = dict(zeros="geo", zeros_arith="arith", welch_on="arith",
              minstat="arith", mmse="arith")


def main(n_draws=3):
    fline = np.arange(140.0, 6960.0, 4.0)
    envs = ["DKITCHEN", "PCAFETER"]
    snrs = [-5.0, 0.0, 5.0, 10.0]
    bias = {(e, s, m): [] for e in envs for s in snrs for m in METHODS}
    rmse = {(e, s, m): [] for e in envs for s in snrs for m in METHODS}
    bias_arith = {(e, s, m): [] for e in envs for s in snrs for m in METHODS}
    gap = {}
    example_saved = False
    for env in envs:
        nseg, Pref, Lgeo, ms_rows, fb_hz = setup_env(env, fline)
        lref = np.log(np.maximum(Pref, 1e-30))
        gap[env] = float(np.mean(lref - Lgeo))      # measured Jensen gap
        print(f"{env}: Jensen gap <logE-Elog> = {gap[env]*DB:.2f} dB", flush=True)
        for snr in snrs:
            for d in range(n_draws):
                sp = load_speech(seed=1000 + 17 * d)
                g = np.sqrt((sp ** 2).mean() / (nseg ** 2).mean()
                            / 10 ** (snr / 10))
                y = sp + g * nseg
                truth = dict(arith=lref + 2 * np.log(g),
                             geo=Lgeo + 2 * np.log(g))
                res, r, Vpack = run_methods(y, fline, Pref * g ** 2, ms_rows,
                                            fb_hz)
                for m in METHODS:
                    err = res[m] - truth[TARGET[m]]
                    bias[(env, snr, m)].append(np.mean(err))
                    rmse[(env, snr, m)].append(np.sqrt(np.mean(err ** 2)))
                    bias_arith[(env, snr, m)].append(
                        np.mean(res[m] - truth["arith"]))
                if not example_saved and env == "PCAFETER" and snr == 5.0:
                    V, fpn, tpn = Vpack
                    ncol = int(8.0 / SIG_W * SIG_W / (HOP / FS))  # 8 s of frames
                    ncol = int(8.0 / (HOP / FS))
                    zs, fcc, tcc = detect_zeros(
                        V[:, :ncol].astype(np.complex128), fpn, tpn[:ncol])
                    np.savez(os.path.join(RES, "e3c_example.npz"),
                             logP2=np.log10(np.abs(V[:, :ncol:2]) ** 2 + 1e-30),
                             f_hz=fpn / SIG_W, t_s=tpn[:ncol:2] * SIG_W,
                             zf_hz=fcc[np.nonzero(zs)[0]] / SIG_W,
                             zt_s=tcc[np.nonzero(zs)[1]] * SIG_W,
                             fline=fline, ltruth_arith=truth["arith"],
                             ltruth_geo=truth["geo"],
                             **{m: res[m] for m in METHODS})
                    example_saved = True
            msg = "  ".join(f"{m}={np.median(bias[(env, snr, m)])*DB:+.2f}"
                            for m in METHODS)
            print(f"{env} SNR={snr:+.0f}dB (vs own target, dB): {msg}", flush=True)
    np.savez(os.path.join(RES, "e3c_speech.npz"),
             envs=envs, snrs=snrs, fline=fline,
             gap_DKITCHEN=gap["DKITCHEN"], gap_PCAFETER=gap["PCAFETER"],
             **{f"bias_{e}_{int(s)}_{m}": np.array(bias[(e, s, m)])
                for e in envs for s in snrs for m in METHODS},
             **{f"rmse_{e}_{int(s)}_{m}": np.array(rmse[(e, s, m)])
                for e in envs for s in snrs for m in METHODS},
             **{f"biasarith_{e}_{int(s)}_{m}": np.array(bias_arith[(e, s, m)])
                for e in envs for s in snrs for m in METHODS})
    print("E3c complete.")


def quick():
    fline = np.arange(140.0, 6960.0, 4.0)
    env, snr = "PCAFETER", 5.0
    nseg, Pref, Lgeo, ms_rows, fb_hz = setup_env(env, fline)
    sp = load_speech(seed=1000)
    g = np.sqrt((sp ** 2).mean() / (nseg ** 2).mean() / 10 ** (snr / 10))
    y = sp + g * nseg
    truth = dict(arith=np.log(np.maximum(Pref, 1e-30)) + 2 * np.log(g),
                 geo=Lgeo + 2 * np.log(g))
    print(f"Jensen gap = {np.mean(truth['arith']-truth['geo'])*DB:.2f} dB")
    res, r, _ = run_methods(y, fline, Pref * g ** 2, ms_rows, fb_hz)
    for m in METHODS:
        err = res[m] - truth[TARGET[m]]
        ea = res[m] - truth["arith"]
        print(f"{m:12s} vs {TARGET[m]:5s}: bias={np.mean(err)*DB:+6.2f} dB "
              f"rms={np.sqrt(np.mean(err**2))*DB:5.2f} dB   "
              f"(vs arith: {np.mean(ea)*DB:+6.2f} dB)")
    print(f"bridged={r.bridged.sum()}/{len(r.bridged)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick()
    else:
        main()
