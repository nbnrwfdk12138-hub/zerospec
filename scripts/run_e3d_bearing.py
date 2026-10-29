"""E3d: broadband noise floor of variable-speed bearing vibration (SM Fig. 8).

University of Ottawa data set.  Pipeline: encoder -> instantaneous shaft speed;
vibration resampled to 20480 Hz; analysis at sigma_w = 1/32 s (1 unit = 32 Hz),
band 0.5--9 kHz (about 266 units, subband calibration).  Reference floor: cells
within 12 Hz of an order track k f_shaft(t), k <= 40, are excluded, the rest is
trimmed at the 99.5% row quantile, and arithmetic, geometric, and median floors
are formed over the retained cells.  The zero estimator sees only the raw
segment.

Usage: run once per record, `run_e3d_bearing.py H-A-1` (healthy, the default)
and `run_e3d_bearing.py I-A-1` (inner-race fault).
Outputs: results/e3d_bearing_<record>.npz, results/e3d_example_<record>.npz
"""
import os
import numpy as np
import scipy.io as sio
import scipy.signal as sg
from zerospec.core import framed_stft, detect_zeros
from zerospec.estimator import ZeroSpecEstimator
from zerospec import baselines

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data", "ottawa", "v43hmbwxpm-2")
os.makedirs(RES, exist_ok=True)

FS0 = 200_000
FS = 20_480                  # resample_poly(x, 128, 1250)
SIG_W = 1.0 / 32.0           # 1 unit = 32 Hz
L = 5120                     # delta f = 4 Hz = 1/8 unit
HOP = 80                     # 1/256 s = 1/8 unit
BAND_HZ = (500.0, 9000.0)
PPR = 1024                   # encoder pulses per revolution
DB = 10 / np.log(10)
EULER_G = 0.5772156649015329


def load_run(name="H-A-1"):
    sub = {"H": "1 Data collected from a healthy bearing",
           "I": "2 Data collected from a bearing with inner race fault"}[name[0]]
    m = sio.loadmat(os.path.join(DATA, sub, name + ".mat"))
    vib = m["Channel_1"].ravel()
    enc = m["Channel_2"].ravel()
    return vib, enc


def shaft_speed(enc, win_s=0.05):
    """Instantaneous shaft frequency from encoder zero-crossing rate."""
    e = enc - np.median(enc)
    n = len(e)
    w = int(win_s * FS0)
    t, f = [], []
    for i0 in range(0, n - w, w // 2):
        seg = e[i0:i0 + w]
        zc = np.count_nonzero(np.diff(seg > 0))
        t.append((i0 + w / 2) / FS0)
        f.append(zc / (2 * win_s) / PPR)       # zc counts both edges: zc/2 pulses per window
    t, f = np.array(t), np.array(f)
    # light smoothing
    k = 9
    fpad = np.pad(f, (k // 2, k // 2), mode="edge")   # edge padding: no zero-padding artefacts
    fs_ = np.convolve(fpad, np.ones(k) / k, mode="valid")
    return t, fs_


def reference_floor(V, f_hz, t_s, fsh_t, fsh_f, kmax=40, guard_hz=12.0, q=0.995):
    """Tacho-informed + trimmed oracle floor per row (arith and geo)."""
    P2 = np.abs(V.astype(np.complex128)) ** 2
    fsh = np.interp(t_s, fsh_t, fsh_f)
    excl = np.zeros(P2.shape, bool)
    for k in range(1, kmax + 1):
        trk = k * fsh
        d = np.abs(f_hz[:, None] - trk[None, :])
        excl |= d <= guard_hz
    arith = np.empty(len(f_hz))
    geo = np.empty(len(f_hz))
    med = np.empty(len(f_hz))
    for i in range(len(f_hz)):
        vals = P2[i, ~excl[i]]
        if vals.size < 50:
            arith[i] = np.nan
            geo[i] = np.nan
            med[i] = np.nan
            continue
        cut = np.quantile(vals, q)
        v = vals[vals <= cut]
        arith[i] = v.mean()
        geo[i] = np.mean(np.log(np.maximum(v, 1e-300))) + EULER_G
        # median floor: Exp-median debias; robust to impulsive bursts with
        # duty cycle below ~50% (the inter-impulse background)
        med[i] = np.median(vals) / np.log(2.0)
    return (np.log(np.maximum(arith, 1e-300)), geo,
            np.log(np.maximum(med, 1e-300)), excl)


def main(name="H-A-1"):
    vib, enc = load_run(name)
    fsh_t, fsh_f = shaft_speed(enc)
    print(f"{name}: shaft speed {fsh_f.min():.1f} -> {fsh_f.max():.1f} Hz "
          f"(median {np.median(fsh_f):.1f})", flush=True)
    x = sg.resample_poly(vib - vib.mean(), 128, 1250)
    V, f_hz, t_s = framed_stft(x, FS, SIG_W, L, HOP, BAND_HZ, margin_hz=40.0)
    fpn, tpn = f_hz * SIG_W, t_s / SIG_W

    # oracle reference on the same TF lattice
    inb = (f_hz >= BAND_HZ[0]) & (f_hz <= BAND_HZ[1])
    lref_a, lref_g, lref_m, excl = reference_floor(V[inb], f_hz[inb], t_s,
                                                   fsh_t, fsh_f)
    fb = f_hz[inb]

    # zero estimator on the raw segment
    est = ZeroSpecEstimator(band=(BAND_HZ[0] * SIG_W, BAND_HZ[1] * SIG_W),
                            margin=1.0, mask=True, t_margin=4.0)
    r = est.fit_stft_banded(V, fpn, tpn, T=tpn[-1], chunk=12.0, overlap=2.0)
    z_log = np.interp(fb, r.f / SIG_W, r.Lambda)

    # baselines
    P2 = np.abs(V[inb, ::8].astype(np.complex128)) ** 2     # 1-unit frames
    tcd = tpn[::8]
    f1s, S1s = sg.welch(x, fs=FS, nperseg=8192, noverlap=4096)
    # smooth Welch to analysis resolution (V-units factor 0.5 for real data)
    taps = (2.0 / SIG_W ** 2) ** 0.25 * np.exp(
        -np.pi * (np.arange(-L // 2, L // 2) / FS) ** 2 / SIG_W ** 2)
    ghat = np.fft.fft(np.fft.ifftshift(taps)) / FS
    fg = np.arange(L) * FS / L
    gf = np.concatenate([fg[fg < FS / 2], fg[fg >= FS / 2] - FS])
    order = np.argsort(gf)
    gfo, g2o = gf[order], np.abs(ghat[order]) ** 2
    core = np.abs(gfo) < 8 * 32.0
    welch_log = np.log(np.maximum(np.array(
        [0.5 * np.trapezoid(np.interp(f - gfo[core], f1s, S1s) * g2o[core],
                            gfo[core]) for f in fb]), 1e-300))
    # min-stat: generic white-noise factory calibration (no noise-only data
    # exists in deployment) and oracle calibration to the reference floor
    raw_ms = baselines.min_stat_raw(P2, tcd)
    ms_white = np.log(np.maximum(raw_ms / 0.5, 1e-300))
    ms_oracle = np.log(np.maximum(raw_ms, 1e-300)) - (
        np.log(np.maximum(raw_ms, 1e-300)) - lref_a).mean()
    mmse_log = np.log(np.maximum(baselines.mmse_spp(P2, tcd), 1e-300))

    ok = np.isfinite(lref_a) & np.isfinite(lref_g) & np.isfinite(lref_m)
    res = dict(zeros=(z_log, lref_m), welch_on=(welch_log, lref_a),
               minstat_white=(ms_white, lref_a), minstat_oracle=(ms_oracle, lref_a),
               mmse=(mmse_log, lref_m))
    print(f"burst excess <arith-median> = {np.mean(lref_a[ok]-lref_m[ok])*DB:.2f} dB"
          f"   <arith-geo> = {np.mean(lref_a[ok]-lref_g[ok])*DB:.2f} dB")
    out = {}
    for m, (v, tgt) in res.items():
        e = (v - tgt)[ok]
        out[m] = (float(np.mean(e)), float(np.sqrt(np.mean(e ** 2))))
        print(f"{m:14s} bias={out[m][0]*DB:+6.2f} dB  rms={out[m][1]*DB:5.2f} dB",
              flush=True)

    # mask-vs-tacho localization: exposure deficit per column against the
    # tacho-excluded fraction of the matching row
    expo = r.exposure
    excl_frac_rows = excl.mean(axis=1)
    excl_on_grid = np.interp(r.f / SIG_W, fb * SIG_W, excl_frac_rows)
    deficit = 1 - expo / max(expo.max(), 1e-12)
    if deficit.std() > 1e-9 and excl_on_grid.std() > 1e-9:
        corr = float(np.corrcoef(deficit, excl_on_grid)[0, 1])
    else:
        corr = float("nan")
    print(f"corr(mask-deficit, tacho-excluded fraction) = {corr:.2f}")

    np.savez(os.path.join(RES, f"e3d_bearing_{name}.npz"),
             fb=fb, lref_a=lref_a, lref_g=lref_g, lref_m=lref_m,
             zeros=z_log, welch=welch_log,
             ms_white=ms_white, ms_oracle=ms_oracle, mmse=mmse_log,
             fsh_t=fsh_t, fsh_f=fsh_f, corr=corr, name=name,
             **{f"m_{k}": np.array(v) for k, v in out.items()})

    # snapshot for the figure: first 6 s, decimated columns
    ncol = int(6.0 / (HOP / FS))
    zs, fcc, tcc = detect_zeros(V[:, :ncol:2].astype(np.complex128),
                                fpn, tpn[:ncol:2])
    np.savez(os.path.join(RES, f"e3d_example_{name}.npz"),
             logP2=np.log10(np.abs(V[:, :ncol:4]) ** 2 + 1e-30),
             f_hz=f_hz, t_s=t_s[:ncol:4],
             zf=fcc[np.nonzero(zs)[0]] / SIG_W, zt=tcc[np.nonzero(zs)[1]] * SIG_W,
             fsh_t=fsh_t, fsh_f=fsh_f,
             fb=fb, lref_a=lref_a, lref_g=lref_g, lref_m=lref_m,
             zeros=z_log, welch=welch_log)
    print("E3d complete.")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "H-A-1")
