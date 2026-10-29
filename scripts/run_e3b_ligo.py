"""E3b: on-source noise PSD estimation on LIGO strain (GW150914, H1), Fig. 4.

Analysis at sigma_w = 1/4 s (1 unit = 4 Hz): frames of 8192 samples (2 s) with
hop 128, so delta-f' = delta-t' = 1/8; band 64--512 Hz (112 units) with subband
calibration.  The estimator runs on the 64-s on-source segment that contains
the event; the reference is the Welch PSD of 896 s of off-source data, smoothed
to the same resolution.  Linear chirps (80 -> 400 Hz) of controlled per-cell
SNR are injected into the strain to measure each method's bias as the
contamination grows.  The event snapshot of Fig. 4(a) uses sigma_w = 1/16 s.

Outputs: results/e3b_ligo.npz, results/e3b_event_snapshot.npz
"""
import os
import numpy as np
import h5py
import scipy.signal as sg
from zerospec.estimator import ZeroSpecEstimator
from zerospec import baselines

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data", "ligo")
os.makedirs(RES, exist_ok=True)

FS = 4096
SIG_W = 1.0 / 4.0                   # analysis window scale (s): 1 unit = 4 Hz
GPS_EVENT = 1126259462.4
BAND_HZ = (64.0, 512.0)
DB = 10 / np.log(10)

L = 8192                            # frame length = 2 s  (delta f = 0.5 Hz = 1/8 unit)
HOP = 128                           # 1/32 s              (delta t' = 1/8 unit)


def load_strain():
    fn = [f for f in os.listdir(DATA) if f.startswith("H-H1") and f.endswith(".hdf5")][0]
    with h5py.File(os.path.join(DATA, fn), "r") as h:
        x = h["strain/Strain"][:]
        gps0 = float(h["meta/GPSstart"][()])
    assert np.isfinite(x).all(), "NaNs in strain"
    return x - x.mean(), gps0


def window_taps(sig_w=SIG_W, Lw=L):
    k = np.arange(-Lw // 2, Lw // 2)
    return (2.0 / sig_w ** 2) ** 0.25 * np.exp(-np.pi * (k / FS) ** 2 / sig_w ** 2)


W_TAPS = window_taps()
GHAT0 = W_TAPS.sum() / FS           # ghat_phys(0)


def stft_frames(x, i0, i1, Lw=L, hop=HOP, sig_w=SIG_W, margin_hz=8.0):
    """STFT of Eq. (1) of x[i0:i1]; returns (V[nf, nframes], f_hz, t_s)
    with t measured from sample i0."""
    C = Lw // 2
    taps = window_taps(sig_w, Lw)
    centers = np.arange(i0 + C, i1 - C, hop)
    frames = np.lib.stride_tricks.sliding_window_view(x, Lw)[centers - C]
    Y = np.fft.fft(np.fft.ifftshift(frames * taps[None, :], axes=1), axis=1) / FS
    f_hz = np.arange(Lw) * FS / Lw
    sel = (f_hz >= BAND_HZ[0] - margin_hz) & (f_hz <= BAND_HZ[1] + margin_hz)
    V = Y[:, sel].T.astype(np.complex128)
    t_s = (centers - i0) / FS
    phase = np.exp(-2j * np.pi * np.outer(f_hz[sel], t_s + i0 / FS))
    return V * phase, f_hz[sel], t_s


def to_normalized(V, f_hz, t_s):
    return V, f_hz * SIG_W, t_s / SIG_W


def welch_ref(x_off):
    """Off-source Welch, then smoothed to the analysis resolution:
    P_ref(f) = 0.5 * (S_1s * |ghat|^2)(f), in V-units."""
    f1s, S1s = sg.welch(x_off, fs=FS, nperseg=8192, noverlap=4096)
    ghat = np.fft.fft(np.fft.ifftshift(W_TAPS)) / FS
    fg = np.arange(L) * FS / L
    g2 = np.abs(ghat) ** 2
    # numerical convolution on the positive axis
    fline = np.arange(BAND_HZ[0] - 32, BAND_HZ[1] + 32, 1.0)
    Pref = np.empty_like(fline)
    gf = np.concatenate([fg[fg < FS / 2], fg[fg >= FS / 2] - FS])
    order = np.argsort(gf)
    gf, g2o = gf[order], g2[order]
    core = np.abs(gf) < 40
    for i, f in enumerate(fline):
        Sv = np.interp(f - gf[core], f1s, S1s)
        Pref[i] = 0.5 * np.trapezoid(Sv * g2o[core], gf[core])
    return fline, Pref


def make_injection(n, t0_s, gamma, fline, Pref, rng):
    """Linear sweep 80->400 Hz across the middle of the segment, amplitude
    tracking sqrt(gamma * P_ref(f(t)))."""
    t = np.arange(n) / FS
    T_seg = n / FS
    tt = (t - 4.0) / (T_seg - 12.0)
    tt = np.clip(tt, 0, 1)
    finst = 80.0 + (400.0 - 80.0) * tt
    act = (tt > 0) & (tt < 1)
    phase = 2 * np.pi * np.cumsum(finst) / FS
    A = np.sqrt(gamma * np.interp(finst, fline, Pref)) / GHAT0
    return np.where(act, A * np.cos(phase + rng.uniform(0, 2 * np.pi)), 0.0), (80.0, 400.0)


def run_estimators(xseg, fline, Pref):
    V, f_hz, t_s = stft_frames(xseg, 0, len(xseg))
    Vn, fpn, tpn = to_normalized(V, f_hz, t_s)
    est = ZeroSpecEstimator(band=(BAND_HZ[0] * SIG_W, BAND_HZ[1] * SIG_W),
                            margin=1.0, mask=True, t_margin=6.0)
    r = est.fit_stft_banded(Vn, fpn, tpn, T=tpn[-1], chunk=12.0, overlap=2.0)
    out_f_hz = r.f / SIG_W
    out = {"zeros": np.interp(fline, out_f_hz, r.Lambda)}
    # value baselines on the decimated spectrogram (1-unit frames)
    dec = 8
    P2 = np.abs(Vn[:, ::dec]) ** 2
    tcd = tpn[::dec]
    inb = (fpn >= BAND_HZ[0] * SIG_W) & (fpn <= BAND_HZ[1] * SIG_W)
    fb_hz = fpn[inb] / SIG_W
    out["welch_on"] = np.interp(fline, *_welch_on(xseg))
    out["welch_med"] = np.interp(fline, *_welch_on(xseg, average="median"))
    out["minstat"] = np.interp(fline, fb_hz,
                               baselines.min_stat_raw(P2[inb], tcd))
    out["mmse"] = np.interp(fline, fb_hz, baselines.mmse_spp(P2[inb], tcd))
    return out, r


def _welch_on(xseg, average="mean"):
    f1s, S1s = sg.welch(xseg, fs=FS, nperseg=8192, noverlap=4096,
                        average=average)
    ghat = np.fft.fft(np.fft.ifftshift(W_TAPS)) / FS
    fg = np.arange(L) * FS / L
    gf = np.concatenate([fg[fg < FS / 2], fg[fg >= FS / 2] - FS])
    order = np.argsort(gf)
    g2o = np.abs(ghat[order]) ** 2
    gfo = gf[order]
    core = np.abs(gfo) < 40
    fline = np.arange(BAND_HZ[0] - 16, BAND_HZ[1] + 16, 1.0)
    P = np.array([0.5 * np.trapezoid(np.interp(f - gfo[core], f1s, S1s) * g2o[core],
                                     gfo[core]) for f in fline])
    return fline, P


def main():
    x, gps0 = load_strain()
    i_evt = int((GPS_EVENT - gps0) * FS)

    # off-source reference: 896 s ending 128 s before the event
    off = x[i_evt - int(1024 * FS): i_evt - int(128 * FS)]
    fline, Pref = welch_ref(off)
    Lref = np.log(np.interp(fline, *_welch_on(off)))   # self-consistency curve

    # on-source segment: 64 s containing the event (event at +48 s)
    i0 = i_evt - int(48 * FS)
    on = x[i0: i0 + int(64 * FS)].copy()

    rng = np.random.default_rng(11)
    gammas = [0.0, 10.0, 30.0, 100.0, 300.0]
    curves = {}
    biases = {m: [] for m in ("zeros", "welch_on", "welch_med", "minstat", "mmse")}
    # offline calibration of min-stat from the off-source data
    V_off, fo, to = stft_frames(off[: int(64 * FS)], 0, int(64 * FS))
    Vn_o, fpn_o, tpn_o = to_normalized(V_off, fo, to)
    inb_o = (fpn_o >= BAND_HZ[0] * SIG_W) & (fpn_o <= BAND_HZ[1] * SIG_W)
    P2_off = np.abs(Vn_o[inb_o, ::8]) ** 2
    raw_off = baselines.min_stat_raw(P2_off, tpn_o[::8])
    Poff_rows = np.interp(fpn_o[inb_o] / SIG_W, fline, Pref)
    ms_bias_rows = raw_off / Poff_rows

    line_excl = None
    for g in gammas:
        inj, swept = make_injection(len(on), 4.0, g, fline, Pref, rng) if g > 0 \
            else (np.zeros_like(on), (80.0, 400.0))
        res, r = run_estimators(on + inj, fline, Pref)
        # apply off-source calibration to min-stat
        fb_hz = (fpn_o[inb_o]) / SIG_W
        res["minstat"] = res["minstat"] / np.interp(fline, fb_hz, ms_bias_rows)
        if line_excl is None:
            # rows the gamma=0 run bridges = persistent instrument lines: the
            # method declares "no measurement under the line"; exclude them
            # from the broadband-bias metric (they are displayed separately)
            bf = r.f[r.bridged] / SIG_W
            line_excl = np.zeros_like(fline, bool)
            for f0 in bf:
                line_excl |= np.abs(fline - f0) <= 2.5
            print(f"line rows excluded from metric: {line_excl.mean()*100:.1f}%")
        sw = (fline >= swept[0]) & (fline <= swept[1]) & ~line_excl
        for m in biases:
            est_log = res[m] if m == "zeros" else np.log(np.maximum(res[m], 1e-60))
            biases[m].append(np.mean((est_log - np.log(Pref))[sw]))
        if g in (0.0, 100.0, 300.0):
            curves[f"g{int(g)}"] = {m: (res[m] if m == "zeros"
                                        else np.log(np.maximum(res[m], 1e-60)))
                                    for m in res}
            if g == 100.0:
                curves["g100_diag"] = dict(bridged=r.bridged, fest=r.f / SIG_W)
        print(f"gamma={g}: " + "  ".join(
            f"{m}={biases[m][-1]*DB:+.2f}dB" for m in biases))

    np.savez(os.path.join(RES, "e3b_ligo.npz"),
             fline=fline, Pref=Pref, gammas=gammas, line_excl=line_excl,
             swept=np.array([80.0, 400.0]),
             **{f"bias_{m}": np.array(v) for m, v in biases.items()},
             **{f"curve_{k}_{m}": v for k, d in curves.items() if k.startswith("g")
                and not k.endswith("diag") for m, v in d.items()},
             diag_bridged=curves["g100_diag"]["bridged"],
             diag_fest=curves["g100_diag"]["fest"])

    # event snapshot: 6 s around GW150914 with a short window (1/16 s) so the
    # 0.2-s chirp is sharp; zeros detected in that window's normalized coords
    sw = 1.0 / 16.0
    j0 = i_evt - int(5 * FS)
    seg = x[j0: j0 + int(6 * FS)]
    Vs, fs_hz, ts_s = stft_frames(seg, 0, len(seg), Lw=2048, hop=32, sig_w=sw)
    from zerospec.core import detect_zeros
    zs, fc, tc = detect_zeros(Vs, fs_hz * sw, ts_s / sw)
    np.savez(os.path.join(RES, "e3b_event_snapshot.npz"),
             logP2=np.log10(np.abs(Vs) ** 2 + 1e-60),
             f_hz=fs_hz, t_s=ts_s, t_evt=(i_evt - j0) / FS,
             zf_hz=fc[np.nonzero(zs)[0]] / sw,
             zt_s=tc[np.nonzero(zs)[1]] * sw)
    print("E3b complete.")


if __name__ == "__main__":
    main()
