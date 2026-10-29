"""E3c-2: speech enhancement driven by each noise-PSD estimate (SM Table II).

Decision-directed Wiener enhancer (512/256 Hann STFT).  The noise periodogram
per enhancer bin is each method's PSD estimate mapped through a fixed
per-frequency factor kappa(f), calibrated once per environment on the disjoint
noise-only segment (the same data that minimum statistics uses).  Metrics:
STOI (pystoi), segmental-SNR improvement, and log-spectral distance to the
clean speech.

Outputs: results/e3c_enhance.npz
"""
import os
import numpy as np
import scipy.signal as sg
from pystoi import stoi as stoi_fn

import importlib
e3c = importlib.import_module("run_e3c_speech")

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FS = 16000
DB = 10 / np.log(10)

NFFT, HOPE = 512, 256
WIN = np.hanning(NFFT)


def stft_e(x):
    n = (len(x) - NFFT) // HOPE
    frames = np.stack([x[i * HOPE:i * HOPE + NFFT] * WIN for i in range(n)])
    return np.fft.rfft(frames, axis=1)


def istft_e(X, nsamp):
    out = np.zeros(nsamp)
    wsum = np.zeros(nsamp)
    fr = np.fft.irfft(X, axis=1)
    for i in range(fr.shape[0]):
        out[i * HOPE:i * HOPE + NFFT] += fr[i] * WIN
        wsum[i * HOPE:i * HOPE + NFFT] += WIN ** 2
    return out / np.maximum(wsum, 1e-8)


def wiener_dd(y, Nf, alpha=0.98, gmin=10 ** (-18 / 20)):
    """Decision-directed Wiener; Nf = noise periodogram per rfft bin."""
    Y = stft_e(y)
    P = np.abs(Y) ** 2
    G = np.ones_like(P)
    xi_prev = np.ones(Y.shape[1])
    Sprev = np.zeros(Y.shape[1])
    for i in range(Y.shape[0]):
        gam = np.minimum(P[i] / np.maximum(Nf, 1e-30), 1e4)
        xi = alpha * Sprev / np.maximum(Nf, 1e-30) + (1 - alpha) * np.maximum(gam - 1, 0)
        g = xi / (1 + xi)
        g = np.maximum(g, gmin)
        G[i] = g
        Sprev = (g ** 2) * P[i]
    return istft_e(G * Y, len(y))


def seg_snr_gain(clean, noisy, enhanced, frame=512, hop=256, lo=-10.0, hi=35.0):
    """Mean segmental SNR improvement over speech-active frames (dB)."""
    def seg(c, x):
        n = (len(c) - frame) // hop
        vals, act = [], []
        for i in range(n):
            s = c[i * hop:i * hop + frame]
            e = s - x[i * hop:i * hop + frame]
            ps, pe = (s ** 2).sum(), (e ** 2).sum()
            act.append(ps)
            vals.append(10 * np.log10(max(ps, 1e-12) / max(pe, 1e-12)))
        vals = np.clip(np.array(vals), lo, hi)
        act = np.array(act)
        keep = act > np.quantile(act, 0.4)          # speech-active frames
        return vals[keep].mean()
    return seg(clean, enhanced) - seg(clean, noisy)


def lsd(clean, enhanced, frame=512, hop=256):
    """Log-spectral distance to clean speech over active frames (dB)."""
    C = np.abs(stft_e(clean)) ** 2
    E = np.abs(stft_e(enhanced)) ** 2
    act = C.sum(axis=1)
    keep = act > np.quantile(act, 0.4)
    d = 10 * np.log10(np.maximum(C[keep], 1e-12)) \
        - 10 * np.log10(np.maximum(E[keep], 1e-12))
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=1))))


def kappa_factor(env, fline, Pref, fbins):
    """Per-bin unit factor: enhancer-domain noise periodogram over P_ref."""
    noise = e3c.load_noise(env)
    cal = noise[int(280 * FS): int(280 * FS) + int(e3c.DUR * FS)]
    N = np.abs(stft_e(cal)) ** 2
    Ebar = N.mean(axis=0)
    Pr = np.interp(fbins, fline, Pref)
    return Ebar / np.maximum(Pr, 1e-30)


def main(n_draws=3):
    fline = np.arange(140.0, 6960.0, 4.0)
    fbins = np.arange(NFFT // 2 + 1) * FS / NFFT
    envs = ["PCAFETER", "DKITCHEN"]
    snrs = [0.0, 5.0]
    methods = ["welch_on", "minstat", "mmse", "zeros_arith", "oracle"]
    stoi_res = {(e, s, m): [] for e in envs for s in snrs
                for m in methods + ["noisy"]}
    seg_res = {(e, s, m): [] for e in envs for s in snrs for m in methods}
    lsd_res = {(e, s, m): [] for e in envs for s in snrs
               for m in methods + ["noisy"]}
    for env in envs:
        nseg, Pref, Lgeo, ms_rows, fb_hz = e3c.setup_env(env, fline)
        kap = kappa_factor(env, fline, Pref, fbins)
        for snr in snrs:
            for d in range(n_draws):
                sp = e3c.load_speech(seed=1000 + 17 * d)
                g = np.sqrt((sp ** 2).mean() / (nseg ** 2).mean()
                            / 10 ** (snr / 10))
                y = sp + g * nseg
                res, _, _ = e3c.run_methods(y, fline, Pref * g ** 2, ms_rows,
                                            fb_hz)
                ests = {m: np.exp(np.interp(fbins, fline, res[m]))
                        for m in ("welch_on", "minstat", "mmse", "zeros_arith")}
                ests["oracle"] = np.interp(fbins, fline, Pref) * g ** 2
                seg = slice(int(2 * FS), int(40 * FS))
                stoi_res[(env, snr, "noisy")].append(
                    stoi_fn(sp[seg], y[seg], FS, extended=False))
                lsd_res[(env, snr, "noisy")].append(lsd(sp[seg], y[seg]))
                for m in methods:
                    xhat = wiener_dd(y, ests[m] * kap)
                    stoi_res[(env, snr, m)].append(
                        stoi_fn(sp[seg], xhat[seg], FS, extended=False))
                    seg_res[(env, snr, m)].append(
                        seg_snr_gain(sp[seg], y[seg], xhat[seg]))
                    lsd_res[(env, snr, m)].append(lsd(sp[seg], xhat[seg]))
            msg = "  ".join(
                f"{m}={np.median(stoi_res[(env, snr, m)]):.3f}"
                for m in ["noisy"] + methods)
            print(f"{env} SNR={snr:+.0f} STOI: {msg}", flush=True)
            msg2 = "  ".join(
                f"{m}={np.median(seg_res[(env, snr, m)]):+.2f}" for m in methods)
            print(f"{env} SNR={snr:+.0f} dSegSNR: {msg2}", flush=True)
            msg3 = "  ".join(
                f"{m}={np.median(lsd_res[(env, snr, m)]):.2f}"
                for m in ["noisy"] + methods)
            print(f"{env} SNR={snr:+.0f} LSD: {msg3}", flush=True)

    np.savez(os.path.join(RES, "e3c_enhance.npz"),
             envs=envs, snrs=snrs, methods=methods,
             **{f"stoi_{e}_{int(s)}_{m}": np.array(stoi_res[(e, s, m)])
                for e in envs for s in snrs for m in methods + ["noisy"]},
             **{f"seg_{e}_{int(s)}_{m}": np.array(seg_res[(e, s, m)])
                for e in envs for s in snrs for m in methods},
             **{f"lsd_{e}_{int(s)}_{m}": np.array(lsd_res[(e, s, m)])
                for e in envs for s in snrs for m in methods + ["noisy"]})
    print("E3c-enhance complete.")


if __name__ == "__main__":
    main()
