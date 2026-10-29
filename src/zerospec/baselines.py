"""Value-based baseline spectral estimators, all targeting the same object:
the (window-)smoothed PSD P(f) = (S * |ghat|^2)(f) in normalized units.

- welch:      segment-averaged periodogram (Hann segments; resolution about
              2/seg_units, finer than |ghat|^2).
- welch_h:    welch convolved with h = |ghat|^2, i.e. at the resolution of the
              estimand (used by all synthetic experiments).
- min_stat:   minimum statistics on the spectrogram, bias factor calibrated
              on white noise (the method's most favorable calibration).
- mmse_spp:   MMSE noise tracker with speech-presence probability
              (Gerkmann-Hendriks 2012 recursion, standard constants).
"""
from __future__ import annotations
import numpy as np


def welch(x: np.ndarray, dt: float, fline: np.ndarray, seg_units: float = 8.0,
          overlap: float = 0.5, average: str = "mean"):
    """Two-sided Welch on complex data; Hann window; resolution ~ 2/seg_units.

    average='median' takes the per-bin median over segment periodograms and
    debiases by ln 2 (exact for the Exp marginal), the standard robust variant.
    """
    L = int(round(seg_units / dt))
    step = max(1, int(round(L * (1 - overlap))))
    win = np.hanning(L)
    U = (win ** 2).sum() * dt
    fft_f = np.fft.fftfreq(L, d=dt)
    periodograms = []
    for s in range(0, len(x) - L + 1, step):
        X = np.fft.fft(x[s:s + L] * win) * dt
        periodograms.append(np.abs(X) ** 2 / U)
    P = np.array(periodograms)
    if average == "median":
        Pw = np.median(P, axis=0) / np.log(2.0)
    else:
        Pw = P.mean(axis=0)
    order = np.argsort(fft_f)
    return np.interp(fline, fft_f[order], Pw[order])


def welch_h(x: np.ndarray, dt: float, fline: np.ndarray, seg_units: float = 8.0,
            overlap: float = 0.5, average: str = "mean", pad: int = 4):
    """Welch estimate convolved with h(xi) = sqrt(2) exp(-2 pi xi^2), so that
    it estimates the window-smoothed spectrum S*h, the estimand of the paper.

    The segment periodograms are evaluated on a zero-padded FFT grid (spacing
    1/(pad*seg_units) normalized units), averaged (mean, or median debiased by
    ln 2), and the average is convolved with h by quadrature on that grid.
    The Hann spectral window adds a variance of about 0.004 units^2 to the
    0.080 of h, a negligible broadening.
    """
    L = int(round(seg_units / dt))
    step = max(1, int(round(L * (1 - overlap))))
    win = np.hanning(L)
    U = (win ** 2).sum() * dt
    nfft = pad * L
    fft_f = np.fft.fftfreq(nfft, d=dt)
    periodograms = []
    for s in range(0, len(x) - L + 1, step):
        X = np.fft.fft(x[s:s + L] * win, n=nfft) * dt
        periodograms.append(np.abs(X) ** 2 / U)
    P = np.array(periodograms)
    if average == "median":
        Pw = np.median(P, axis=0) / np.log(2.0)
    else:
        Pw = P.mean(axis=0)
    order = np.argsort(fft_f)
    nu, Pw = fft_f[order], Pw[order]
    dnu = nu[1] - nu[0]
    fl = np.atleast_1d(np.asarray(fline, dtype=float))
    h = np.sqrt(2.0) * np.exp(-2.0 * np.pi * (fl[:, None] - nu[None, :]) ** 2)
    return (h * Pw[None, :]).sum(axis=1) * dnu


def _smooth_spectrogram(P2: np.ndarray, tc: np.ndarray, tau_units: float = 1.0):
    """First-order recursive smoothing along time (axis=1)."""
    dt = tc[1] - tc[0]
    a = np.exp(-dt / tau_units)
    out = np.empty_like(P2)
    out[:, 0] = P2[:, 0]
    for j in range(1, P2.shape[1]):
        out[:, j] = a * out[:, j - 1] + (1 - a) * P2[:, j]
    return out


def min_stat_raw(P2: np.ndarray, tc: np.ndarray, win_units: float = 16.0):
    """Uncalibrated minimum-statistics track: median over time of the sliding
    minimum of the time-smoothed spectrogram, per frequency row."""
    dt = tc[1] - tc[0]
    W = max(2, int(round(win_units / dt)))
    Ps = _smooth_spectrogram(P2, tc)
    from numpy.lib.stride_tricks import sliding_window_view
    mins = sliding_window_view(Ps, W, axis=1).min(axis=-1)
    return np.median(mins, axis=1)


def min_stat(P2: np.ndarray, tc: np.ndarray, win_units: float = 16.0,
             bias=None):
    """Minimum-statistics estimate of P(f).

    `bias` is the per-row (or scalar) calibration factor E[raw]/P, ideally
    obtained from noise-only data with the same spectral color and the same
    spectrogram configuration (see `min_stat_oracle_bias`).  This oracle
    calibration is the baseline's most favorable case; without it the
    white-noise factor mis-calibrates colored noise by several dB.
    """
    raw = min_stat_raw(P2, tc, win_units)
    if bias is None:
        bias = 0.5  # rough white-noise default; prefer min_stat_oracle_bias
    return raw / bias


def min_stat_oracle_bias(noise_P2s, tc, P_true_rows, win_units: float = 16.0):
    """Per-row calibration factor E[raw]/P from noise-only spectrograms."""
    raws = np.array([min_stat_raw(p2, tc, win_units) for p2 in noise_P2s])
    return raws.mean(axis=0) / np.maximum(P_true_rows, 1e-300)


def mmse_spp(P2: np.ndarray, tc: np.ndarray, xi_h1_db: float = 15.0,
             alpha_n: float = 0.8, init_units: float = 4.0):
    """MMSE-SPP noise PSD tracker (Gerkmann-Hendriks 2012), per frequency row.

    Returns the time-median of the tracked noise PSD.
    """
    dt = tc[1] - tc[0]
    xi = 10 ** (xi_h1_db / 10)
    n_init = max(1, int(round(init_units / dt)))
    sig = P2[:, :n_init].mean(axis=1)
    pbar = np.full(P2.shape[0], 0.5)
    track = np.empty_like(P2)
    for j in range(P2.shape[1]):
        y = P2[:, j]
        gam = y / np.maximum(sig, 1e-300)
        p = 1.0 / (1.0 + (1.0 + xi) * np.exp(-np.minimum(gam * xi / (1.0 + xi), 700)))
        pbar = 0.9 * pbar + 0.1 * p
        p = np.where(pbar > 0.99, np.minimum(p, 0.99), p)
        est = (1 - p) * y + p * sig
        sig = alpha_n * sig + (1 - alpha_n) * est
        track[:, j] = sig
    return np.median(track, axis=1)
