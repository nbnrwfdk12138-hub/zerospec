"""STFT machinery, validated zero detector, and noise synthesis.

All in window-normalized coordinates. The zero detector uses mixed-gauge
edge differencing (t-edges from arg V, f-edges from arg(V e^{2i pi f t})),
which keeps every discrete phase step well below pi: validated against the
closed-form two-logon interference lattice and the white-noise density 1.
"""
from __future__ import annotations
import numpy as np
import scipy.signal as sg


class GaussSTFT:
    """Gaussian-window STFT of a uniformly sampled record on [0, T]."""

    def __init__(self, T: float, dt: float = 1 / 16):
        self.T, self.dt = float(T), float(dt)
        self.N = int(round(T / dt))
        self.tau = np.arange(self.N) * dt
        half = int(round(4.0 / dt))                       # +-4 truncation
        tg = np.arange(-half, half + 1) * dt
        self.window = 2.0 ** 0.25 * np.exp(-np.pi * tg ** 2)

    def stft(self, x: np.ndarray, fgrid: np.ndarray, tstep: int = 2):
        """V(t,f) on (fgrid x decimated time grid). Returns (V, tgrid)."""
        rows = []
        for f in np.asarray(fgrid):
            xf = x * np.exp(-2j * np.pi * f * self.tau)
            v = sg.fftconvolve(xf, self.window, mode="same") * self.dt
            rows.append(v[::tstep])
        return np.array(rows), self.tau[::tstep]


def detect_zeros(V: np.ndarray, fgrid: np.ndarray, tgrid: np.ndarray):
    """Plaquette winding counts; returns (bool mask on plaquettes, fc, tc).

    V is sampled on (fgrid x tgrid); both steps must be <= 1/4 normalized.
    """
    fg = np.asarray(fgrid)
    ph = np.angle(V)                                            # t-edges
    phm = np.angle(V * np.exp(2j * np.pi * np.outer(fg, tgrid)))  # f-edges
    wrap = lambda d: (d + np.pi) % (2 * np.pi) - np.pi
    d_t = wrap(np.diff(ph, axis=1))
    d_f = wrap(np.diff(phm, axis=0))
    w = d_t[:-1, :] + d_f[:, 1:] - d_t[1:, :] - d_f[:, :-1]
    zs = np.abs(w) > np.pi
    fc = 0.5 * (fg[:-1] + fg[1:])
    tc = 0.5 * (np.asarray(tgrid)[:-1] + np.asarray(tgrid)[1:])
    return zs, fc, tc


def framed_stft(x, fs, sig_w, L, hop, band_hz, margin_hz=0.0, dtype=np.complex64,
                block=4096):
    """Gaussian STFT of Eq. (1) for a real record, framed implementation.

    Returns (V[nf, nframes], f_hz, t_s).  Normalized coordinates are
    (f_hz * sig_w, t_s / sig_w); choose L, hop so that the normalized grid
    steps are <= 1/4.  Frames are processed in blocks to bound memory.
    """
    C = L // 2
    k = np.arange(-C, C)
    taps = (2.0 / sig_w ** 2) ** 0.25 * np.exp(-np.pi * (k / fs) ** 2 / sig_w ** 2)
    centers = np.arange(C, len(x) - C, hop)
    f_hz_all = np.arange(L) * fs / L
    sel = (f_hz_all >= band_hz[0] - margin_hz) & (f_hz_all <= band_hz[1] + margin_hz)
    f_hz = f_hz_all[sel]
    V = np.empty((sel.sum(), len(centers)), dtype=dtype)
    sw = np.lib.stride_tricks.sliding_window_view(x, L)
    for b0 in range(0, len(centers), block):
        cb = centers[b0:b0 + block]
        frames = sw[cb - C] * taps[None, :]
        Y = np.fft.fft(np.fft.ifftshift(frames, axes=1), axis=1)[:, sel] / fs
        t_b = cb / fs
        V[:, b0:b0 + len(cb)] = (Y.T * np.exp(-2j * np.pi * np.outer(f_hz, t_b))
                                 ).astype(dtype)
    return V, f_hz, centers / fs


# noise synthesis (circular Gaussian)

def synth_stationary(Sfunc, T: float, rng, dt: float = 1 / 16, nu_max: float = 6.0):
    """Circular stationary noise with PSD Sfunc on [-nu_max, nu_max].

    On the regular grid nu_m = -nu_max + m/T the spectral sum IS a length-N
    DFT (m*dnu*tau_k = mk/N), so synthesis is O(N log N) via FFT instead of
    the O(M N) dense matmul (which needs tens of GB at T >= 2048)."""
    N = int(round(T / dt))
    assert 2 * nu_max < 1.0 / dt, "synthesis band must fit below Nyquist"
    tau = np.arange(N) * dt
    dnu = 1.0 / T
    nus = np.arange(-nu_max, nu_max, dnu)
    M = len(nus)
    assert M <= N
    zeta = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) / np.sqrt(2)
    b = np.zeros(N, dtype=complex)
    b[:M] = np.sqrt(np.maximum(Sfunc(nus), 0.0) * dnu) * zeta
    return np.exp(-2j * np.pi * nu_max * tau) * (np.fft.ifft(b) * N)


def synth_modulated(sig2_func, T: float, rng, dt: float = 1 / 16):
    """Uniformly modulated circular white noise, local variance sig2_func(t)."""
    N = int(round(T / dt))
    tau = np.arange(N) * dt
    w = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2 * dt)
    return np.sqrt(sig2_func(tau)) * w
